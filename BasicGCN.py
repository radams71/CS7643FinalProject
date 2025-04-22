import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, aggr, GENConv
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer, CaptumExplainer
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients
import numpy as np
import random
from collections import defaultdict
import networkx as nx
from torch_geometric.utils import to_networkx

criterion = torch.nn.BCEWithLogitsLoss()
class GCN(torch.nn.Module):
    def __init__(self, hidden_size, num_tasks):
        super().__init__()
        aggregation = aggr.PowerMeanAggregation(p=-1)
        self.encoder = AtomEncoder(hidden_size)
        self.conv1 = GCNConv(hidden_size, hidden_size, aggr=aggregation)
        self.conv2 = GCNConv(hidden_size, hidden_size, aggr=aggregation)
        self.linear = torch.nn.Linear(hidden_size, num_tasks)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.encoder(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x


def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred.to(torch.float32), batch.y.to(torch.float32))
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)
    return result_dict

def draw_molecule(g, edge_mask=None, draw_edge_labels=False):
    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color='azure')

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_color='red')
    plt.show()

def to_molecule(data):
    elements = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]
    g = to_networkx(data, node_attrs=['x'])
    for u, data in g.nodes(data=True):
        data['name'] = elements[data['x'][0]-1]
        del data['x']
    return g

def main():
    logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    batch_size = 32
    hidden_size = 64
    learning_rate = 0.001
    epochs = 10

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda:0")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

    evaluator = Evaluator(name="ogbg-molhiv")
    print(dataset[0].x.dtype)
    model = GCN(hidden_size, dataset.num_tasks).to(device)
    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(1, epochs + 1):
        logging.info("=====Epoch {}".format(epoch))
        logging.info('Training...')

        epoch_loss = train(model, device, train_loader, optimizer)

        logging.info('Evaluating...')
        train_result = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
        valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
        test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]

        logging.info({'Train': train_result,
                     'Validation': valid_result,
                     'Test': test_result})

    def model_forward(edge_mask, data):
        batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
        edge_mask = edge_mask.unsqueeze(1).to(torch.float32)
        out = model(data.x, data.edge_index, batch, edge_mask)
        return out

    def explain(method, data, target=0):
        input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
        if method == 'ig':
            ig = IntegratedGradients(model_forward)
            mask = ig.attribute(input_mask, target=target,
                                additional_forward_args=(data,),
                                internal_batch_size=data.edge_index.shape[1])
        elif method == 'saliency':
            saliency = Saliency(model_forward)
            mask = saliency.attribute(input_mask, target=target,
                                      additional_forward_args=(data,))
        else:
            raise Exception('Unknown explanation method')

        edge_mask = np.abs(mask.cpu().detach().numpy())
        if edge_mask.max() > 0:  # avoid division by zero
            edge_mask = edge_mask / edge_mask.max()
        return edge_mask

    def aggregate_edge_directions(edge_mask, data):
        edge_mask_dict = defaultdict(float)
        for val, u, v in list(zip(edge_mask, *data.edge_index)):
            u, v = u.item(), v.item()
            if u > v:
                u, v = v, u
            edge_mask_dict[(u, v)] += val
        return edge_mask_dict

    data = dataset[0].to(device)
    mol = to_molecule(data)

    for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
        edge_mask = explain(method, data, target=0)
        edge_mask_dict = aggregate_edge_directions(edge_mask, data)
        plt.figure(figsize=(10, 5))
        plt.title(title)
        draw_molecule(mol, edge_mask_dict)




if __name__ == '__main__':
    main()