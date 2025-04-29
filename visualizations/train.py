from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv
from torch.nn.functional import relu, dropout
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from libauc.utils import set_all_seeds
import logging
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients, LayerIntegratedGradients, InterpretableEmbeddingBase, configure_interpretable_embedding_layer
from torch_geometric.utils import to_networkx
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from collections import defaultdict
from torch_geometric.nn.aggr import (
    Aggregation,
    MaxAggregation,
    MeanAggregation,
    MultiAggregation,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
    MulAggregation,
    PowerMeanAggregation
)

class GNN(torch.nn.Module):
    def __init__(self, hidden_size, aggr):
        super(GNN, self).__init__()
        self.embedding = AtomEncoder(hidden_size)
        self.conv1 = GCNConv(hidden_size, hidden_size, aggr=aggr)
        self.conv2 = GCNConv(hidden_size, hidden_size, aggr=aggr)
        self.conv3 = GCNConv(hidden_size, hidden_size, aggr=aggr)
        self.lin = Linear(hidden_size, 1)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = dropout(x, 0.2, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return torch.sigmoid(x)

def train(model, device, loader, optimizer, criterion):
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


def draw_molecule(g, edge_mask=None, draw_edge_labels=False, node_mask=None):
    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.spring_layout(g)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, edge_color=edge_color, width=widths, node_color=node_mask, cmap=plt.cm.Blues, font_color='red')
    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, edge_labels=edge_labels,
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
        data['name'] = elements[data['x'][0]]
        del data['x']
    return g

def main():
    logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Training Parameters
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 0.00001
    epochs = 100
    decay_epochs = [int(epochs * 0.33), int(epochs * 0.66)]
    seed = 0

    # Model Paramters
    aggregation = "median"
    t = 1.0
    p = 1.0
    hidden_size = 64


    set_all_seeds(seed)
    device = torch.device("cuda:0")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

    split_idx = dataset.get_idx_split()
    train_set = dataset[split_idx["train"]]
    valid_set = dataset[split_idx["valid"]]
    test_set = dataset[split_idx["test"]]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    evaluator = Evaluator(name="ogbg-molhiv")

    model = GNN(hidden_size, aggregation).to(device)
    logging.info(model)


    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_log = []
    valid_log = []
    test_log = []

    best_valid = 0
    final_test = 0
    for epoch in range(1, epochs + 1):
        logging.info("Epoch {}".format(epoch))
        logging.info('Training...')

        epoch_loss = train(model, device, train_loader, optimizer, criterion)

        logging.info('Evaluating...')
        train_result = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
        valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
        test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]

        logging.info({'Train': train_result,
                     'Validation': valid_result,
                     'Test': test_result,
                      "Train Loss": epoch_loss})
        train_log.append(train_result)
        valid_log.append(valid_result)
        test_log.append(test_result)
        if valid_result > best_valid:
            best_valid = valid_result
            final_test = test_result
            torch.save(model.state_dict(), 'best_model_'+ aggregation +'.pth')
    print(best_valid)
    print(final_test)

    def model_forward(node_mask, data):
        batch = torch.zeros(1, dtype=int).to(device)
        out = model(node_mask, data.edge_index, batch)
        return out

    # def explain(method, data, target=0):
    #     input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    #     if method == 'ig':
    #         ig = IntegratedGradients(model_forward)
    #         mask = ig.attribute(input_mask, target=target,
    #                             additional_forward_args=(data,),
    #                             internal_batch_size=data.edge_index.shape[1])
    #     elif method == 'saliency':
    #         saliency = Saliency(model_forward)
    #         mask = saliency.attribute(input_mask, target=target,
    #                                   additional_forward_args=(data,))
    #     else:
    #         raise Exception('Unknown explanation method')
    #
    #     edge_mask = np.abs(mask.cpu().detach().numpy())
    #     if edge_mask.max() > 0:  # avoid division by zero
    #         edge_mask = edge_mask / edge_mask.max()
    #     return edge_mask
    #
    # def aggregate_edge_directions(edge_mask, data):
    #     edge_mask_dict = defaultdict(float)
    #     for val, u, v in list(zip(edge_mask, *data.edge_index)):
    #         u, v = u.item(), v.item()
    #         if u > v:
    #             u, v = v, u
    #         edge_mask_dict[(u, v)] += val
    #     return edge_mask_dict
    #
    # data = dataset[0].to(device)
    # mol = to_molecule(data)
    #
    # for title, method in [('Integrated Gradients', 'ig')]:
    #     edge_mask = explain(method, data, target=0)
    #     edge_mask_dict = aggregate_edge_directions(edge_mask, data)
    #     plt.figure(figsize=(10, 5))
    #     plt.title(title)
    #     draw_molecule(mol, edge_mask_dict)
    data = dataset[0].to(device)
    intepretable_embedding = configure_interpretable_embedding_layer(model, "embedding")
    input_emb = intepretable_embedding.indices_to_embeddings(data.x).unsqueeze(0)
    ig = IntegratedGradients(model_forward)
    attributions = ig.attribute(inputs=input_emb,
                                       target=0,
                                       n_steps=50,
                                       additional_forward_args=(data,))
    mol = to_molecule(data)
    attributions = attributions.cpu().detach().squeeze(0)
    attributions = attributions.abs().sum(dim=1)
    attributions /= attributions.max()
    print(attributions)
    draw_molecule(mol, node_mask=attributions)





if __name__ == '__main__':
    main()

