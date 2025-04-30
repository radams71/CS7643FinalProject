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
from captum.attr import Saliency, IntegratedGradients, LayerIntegratedGradients, InterpretableEmbeddingBase, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from torch_geometric.utils import to_networkx
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from collections import defaultdict


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
    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, edge_color=edge_color, width=widths, node_color=node_mask, cmap=plt.cm.Blues, font_color='red', edge_cmap=plt.cm.Purples)
    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, edge_labels=edge_labels,
                                     font_color='red')


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
    aggregation = "sum"
    t = 1.0
    p = 1.0
    hidden_size = 64


    set_all_seeds(seed)
    device = torch.device("cuda:0")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    print(dataset.y.sum()/len(dataset.y))

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

    PATH = 'best_model_'+aggregation+'.pth'
    state_dict = torch.load(PATH)
    model.load_state_dict(state_dict)

    train_log = []
    valid_log = []
    test_log = []

    logging.info('Evaluating...')
    train_result = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
    valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
    test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]

    logging.info({'Train': train_result,
                'Validation': valid_result,
                'Test': test_result})
    train_log.append(train_result)
    valid_log.append(valid_result)
    test_log.append(test_result)

    def model_forward_node(node_mask, data):
        out = model(node_mask, data.edge_index, data.batch)
        return out

    def model_forward_edge(edge_mask, data):
        out = model(data.x, data.edge_index, data.batch, edge_mask)
        return out

    def aggregate_edge_directions(edge_mask, data):
        edge_mask_dict = defaultdict(float)
        for val, u, v in list(zip(edge_mask, *data.edge_index)):
            u, v = u.item(), v.item()
            if u > v:
                u, v = v, u
            edge_mask_dict[(u, v)] += val
        return edge_mask_dict

    data = DataLoader(test_set, batch_size=1, shuffle=False)
    for batch in data:
        if batch.y[0][0] == 1:
            break
    data = batch.to(device)
    print(model(data.x, data.edge_index, data.batch))
    ig = IntegratedGradients(model_forward_edge)
    edge_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    edge_attributions = ig.attribute(inputs=edge_mask,
                                     target=0,
                                     n_steps=100,
                                     additional_forward_args=(data,),
                                     internal_batch_size=data.edge_index.shape[1])
    edge_attributions = edge_attributions.cpu().detach().numpy()
    edge_attributions = np.abs(edge_attributions)
    edge_attributions /= np.max(edge_attributions)
    edge_attributions = aggregate_edge_directions(edge_attributions, data)

    interpretable_embedding = configure_interpretable_embedding_layer(model, "embedding")
    input_emb = interpretable_embedding.indices_to_embeddings(data.x)
    ig = IntegratedGradients(model_forward_node)
    node_attributions = ig.attribute(inputs=input_emb,
                                     target=0,
                                     n_steps=100,
                                     additional_forward_args=(data,),
                                     internal_batch_size=data.x.shape[1])
    mol = to_molecule(data)
    node_attributions = node_attributions.cpu().detach()
    node_attributions = node_attributions.abs().sum(dim=1)
    node_attributions /= node_attributions.max()
    print(node_attributions)

    plt.figure(figsize=(10, 10))
    draw_molecule(mol, edge_mask=edge_attributions, node_mask=node_attributions)
    plt.savefig(aggregation + '.png')
    plt.show()




if __name__ == '__main__':
    main()
