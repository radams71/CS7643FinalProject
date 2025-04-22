import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, aggr, GENConv, DeepGCNLayer
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn.aggr import LSTMAggregation
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients
import numpy as np
import random
from collections import defaultdict
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything
from torch.nn import LayerNorm, Linear, ReLU

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

criterion = AUCMLoss()
class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_tasks):
        super().__init__()
        aggregator = "sum"
        self.encoder = AtomEncoder(hidden_size)
        self.conv1 = GCNConv(hidden_size, hidden_size, aggr=aggregator)
        self.conv2 = GCNConv(hidden_size, hidden_size, aggr=aggregator)
        self.conv3 = GCNConv(hidden_size, hidden_size, aggr=aggregator)
        self.linear = torch.nn.Linear(hidden_size, num_tasks)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.encoder(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

class DeeperGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_layers, num_tasks):
        super().__init__()

        self.node_encoder = AtomEncoder(hidden_channels)
        self.edge_encoder = BondEncoder(hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.2,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.pool = global_mean_pool
        self.lin = Linear(hidden_channels, num_tasks)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.pool(x, batch)
        return self.lin(x)

def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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

    batch_size = 512
    hidden_size = 64
    num_layers = 7
    learning_rate = 0.1
    weight_decay = 0.00001
    margin = 1.0
    epoch_decay = 0.003
    decay_epochs = [50, 75]
    epochs = 100

    seed_everything(0)
    device = torch.device("cuda:0")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

    evaluator = Evaluator(name="ogbg-molhiv")

    model = DeeperGCN(dataset.num_features, dataset.num_edge_features, hidden_size, num_layers, dataset.num_tasks).to(device)
    logging.info(model)

    optimizer = PESG(model.parameters(), lr=learning_rate, weight_decay=weight_decay, loss_fn=criterion, margin=margin, epoch_decay = epoch_decay)

    train_log = []
    valid_log = []
    test_log = []

    for epoch in range(1, epochs + 1):
        if epoch in decay_epochs:
            optimizer.update_regularizer(decay_factor=10)
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
        train_log.append(train_result)
        valid_log.append(valid_result)
        test_log.append(test_result)

    plt.plot(train_log, label='Train')
    plt.plot(valid_log, label='Validation')
    plt.plot(test_log, label='Test')
    plt.show()

    # def model_forward(edge_mask, data):
    #     batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
    #     edge_mask = edge_mask.unsqueeze(1).to(torch.int64)
    #     out = model(data.x, data.edge_index, edge_mask, batch)
    #     return out
    #
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

    # data = dataset[0].to(device)
    # mol = to_molecule(data)
    #
    # for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
    #     edge_mask = explain(method, data, target=0)
    #     edge_mask_dict = aggregate_edge_directions(edge_mask, data)
    #     plt.figure(figsize=(10, 5))
    #     plt.title(title)
    #     draw_molecule(mol, edge_mask_dict)
    #
    # data = dataset[0].to(device)
    #
    # explainer = Explainer(
    #     model=model,
    #     algorithm=GNNExplainer(epochs=200),
    #     explanation_type='model',
    #     node_mask_type=None,
    #     edge_mask_type='object',
    #     model_config=dict(
    #         mode='binary_classification',
    #         task_level='graph',
    #         return_type='probs',
    #     ),
    # )
    # node_index = 10
    # explanation = explainer(data.x, data.edge_index, batch=torch.zeros(data.x.shape[0], dtype=int).to(device), edge_attr=data.edge_attr)
    # print(f'Generated explanations in {explanation.available_explanations}')
    #
    # path = 'feature_importance.png'
    # explanation.visualize_feature_importance(path)
    # print(f"Feature importance plot has been saved to '{path}'")
    #
    # path = 'subgraph.pdf'
    # explanation.visualize_graph(path)
    # print(f"Subgraph visualization plot has been saved to '{path}'")



if __name__ == '__main__':
    main()