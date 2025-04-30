#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: conv_gcn_enhanced.py
Author: Pedram Gharani
Created: 2025-04-22
Last Modified: 2025-04-29
Description:
    A Graph Convolutional Network (GCN) implementation for molecular property prediction on the ogbg-molhiv dataset. 
    This code defines a ConvGCN model with multi-head attention pooling, residual connections, and customizable aggregation types ("add", "mean", "max"). 
    It includes atom and edge embeddings, dropout, and a combined loss function (focal + weighted BCE) to handle imbalanced data. 
    The model is trained and evaluated using PyTorch Geometric, with visualizations for training metrics, ROC curves, and attention scores.


Usage:
    python conv_gcn_enhanced.py [--hidden_dims HIDDEN_DIMS] [--pooling_type POOLING] [--use_input_transform TRANSFORM] [--use_atom_embedding EMBED_ATOM] [--use_edge_attr EDGE_ATTR] 
                                [--aggr_type MODE] [--atom_embed_dim ATOM_DIM] [--edge_embed_dim EDGE_DIM] [--dropout_rate DROPOUT]


Arguments:
    --hidden_dims           HIDDEN_DIMS of network. Hidden dimensions for the 3 GCNConv layers. Allowed values: int int int. Default: 192 384 768'.
    --pooling_type          POOLING type. Graph-level pooling type: 'mean', 'sum', 'max', or 'attention'. choices=["mean", "sum", "max", "attention"]. Default: attention
    --use_input_transform   TRANSFORM input. Whether to use input transformation (0=False, 1=True). Allowed values: choices=[0, 1]. Default: 1
    --use_atom_embedding    EMBED_ATOM is used. Whether to use atom embedding (0=False, 1=True) Allowed values: choices=[0 , 1]. Default: 1
    --use_edge_attr         EDGE_ATTR is used. Whether to use edge attributes (0=False, 1=True). Allowed values: choices=[0 , 1]. Default: 1
    --aggr_type             MODE of aggregation. Message-passing aggregation type. Allowed values: choices=["add", "mean", "max"]. Default: add           
    --atom_embed_dim        ATOM_DIM value. Dimension of atom embedding. Allowed values: int. Default: 64
    --edge_embed_dim        EDGE_DIM value. Dimension of edge embedding. Allowed values: int. Default: 16
    --dropout_rate          DROPOUT rate. Allowed values: float. Default: 0.2 

"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool, BatchNorm
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.utils import degree, softmax, scatter
from visualize_results import plot_training_metrics, plot_roc_curve, plot_attention_scores
import argparse
import os
 
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        self.query = torch.nn.Linear(in_channels, in_channels)
        self.key = torch.nn.Linear(in_channels, in_channels)
        self.value = torch.nn.Linear(in_channels, in_channels)
        self.out = torch.nn.Linear(in_channels, in_channels)
 
    def forward(self, x, batch):
        num_nodes = x.size(0)
        q = self.query(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.key(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.value(x).view(num_nodes, self.num_heads, self.head_dim)
 
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)
        att_scores = softmax(scores, batch, num_nodes=num_nodes)
        out = torch.bmm(att_scores, v).view(num_nodes, -1)
        out = self.out(out)
        return out, att_scores.mean(dim=1)
 
class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, aggr="add"):
        super(CustomConv, self).__init__(aggr=aggr)
        self.edge_dim = edge_dim
        self.lin = torch.nn.Linear(in_channels, out_channels)
        if edge_dim is not None:
            self.edge_projection = torch.nn.Linear(edge_dim, in_channels)
 
    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
 
        if self.edge_dim is not None and edge_attr is not None:
            edge_attr = self.edge_projection(edge_attr)
            x = x + scatter(edge_attr, edge_index[1], dim=0, dim_size=x.size(0), reduce='mean')
 
        x = self.lin(x)
        return self.propagate(edge_index, x=x, norm=norm)
 
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
 
class ConvGCN(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dims, output_dim, use_edge_attr=False, use_input_transform=False, use_atom_embedding=True, pooling_type="mean", dropout_rate=0.2, aggr_type="add", atom_embed_dim=16, edge_embed_dim=16):
        super(ConvGCN, self).__init__()
        self.use_edge_attr = use_edge_attr
        self.use_input_transform = use_input_transform
        self.use_atom_embedding = use_atom_embedding
        self.pooling_type = pooling_type
        self.dropout_rate = dropout_rate
        self.aggr_type = aggr_type
        self.atom_embed_dim = atom_embed_dim
        self.edge_embed_dim = edge_embed_dim
        
        assert len(hidden_dims) == 3, "hidden_dims must contain exactly 3 values for the 3 GCNConv layers"
        
        if self.use_atom_embedding:
            max_atomic_num = 100
            self.atom_embedding = torch.nn.Embedding(max_atomic_num + 1, atom_embed_dim)
            self.other_features_transform = torch.nn.Linear(input_dim - 1, atom_embed_dim)
            self.combined_dim = atom_embed_dim + atom_embed_dim
        else:
            self.combined_dim = input_dim
        
        self.input_transform = torch.nn.Linear(self.combined_dim, hidden_dims[0]) if use_input_transform else None
        conv1_input_dim = self.combined_dim if not use_input_transform else hidden_dims[0]
        
        if use_edge_attr:
            self.edge_embedding = torch.nn.Linear(edge_dim, edge_embed_dim)
            conv1_edge_dim = edge_embed_dim
        else:
            conv1_edge_dim = None
        
        self.conv1 = CustomConv(conv1_input_dim, hidden_dims[0], edge_dim=conv1_edge_dim, aggr=aggr_type)
        self.bn1 = BatchNorm(hidden_dims[0])
        self.conv2 = CustomConv(hidden_dims[0], hidden_dims[1], edge_dim=conv1_edge_dim, aggr=aggr_type)
        self.bn2 = BatchNorm(hidden_dims[1])
        self.conv3 = CustomConv(hidden_dims[1], hidden_dims[2], edge_dim=conv1_edge_dim, aggr=aggr_type)
        self.bn3 = BatchNorm(hidden_dims[2])
        
        self.attention = MultiHeadAttention(hidden_dims[2], num_heads=4) if pooling_type == "attention" else None
        self.output = torch.nn.Linear(hidden_dims[2], output_dim)
 
    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = x.float()
        edge_index = edge_index.long()
        
        if self.use_atom_embedding:
            atomic_nums = x[:, 0].long()
            other_features = x[:, 1:]
            
            node_batch = batch
            other_features_mean = scatter(other_features, node_batch, dim=0, reduce='mean')
            other_features_var = scatter((other_features - other_features_mean[node_batch])**2, node_batch, dim=0, reduce='mean')
            other_features_std = torch.sqrt(other_features_var + 1e-8)
            other_features = (other_features - other_features_mean[node_batch]) / (other_features_std[node_batch] + 1e-8)
            
            atom_embed = self.atom_embedding(atomic_nums)
            other_embed = F.relu(self.other_features_transform(other_features))
            x = torch.cat([atom_embed, other_embed], dim=-1)
        else:
            x_mean = scatter(x, batch, dim=0, reduce='mean')
            x_var = scatter((x - x_mean[batch])**2, batch, dim=0, reduce='mean')
            x_std = torch.sqrt(x_var + 1e-8)
            x = (x - x_mean[batch]) / (x_std[batch] + 1e-8)
        
        if self.use_input_transform:
            x = F.relu(self.input_transform(x))
        
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = edge_attr.float()
            edge_batch = batch[edge_index[0]]
            edge_attr_mean = scatter(edge_attr, edge_batch, dim=0, reduce='mean')
            edge_attr_var = scatter((edge_attr - edge_attr_mean[edge_batch])**2, edge_batch, dim=0, reduce='mean')
            edge_attr_std = torch.sqrt(edge_attr_var + 1e-8)
            edge_attr = (edge_attr - edge_attr_mean[edge_batch]) / (edge_attr_std[edge_batch] + 1e-8)
            
            edge_attr = self.edge_embedding(edge_attr)
        else:
            edge_attr = None
 
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        att_scores = None
        if self.pooling_type == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling_type == "sum":
            x = global_add_pool(x, batch)
        elif self.pooling_type == "max":
            x = global_max_pool(x, batch)
        elif self.pooling_type == "attention":
            x, att_scores = self.attention(x, batch)
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        out = self.output(x)
        if return_attention:
            return out, att_scores
        return out
 
def focal_loss(logits, targets, alpha=0.25, gamma=1.5, label_smoothing=0.1):
    targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()
 
def weighted_bce_loss(logits, targets, pos_weight=10.0):
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor([pos_weight]).to(logits.device))
 
def combined_loss(logits, targets, alpha=0.25, gamma=1.5, label_smoothing=0.1, pos_weight=10.0, focal_weight=0.5):
    focal = focal_loss(logits, targets, alpha, gamma, label_smoothing)
    bce = weighted_bce_loss(logits, targets, pos_weight)
    return focal_weight * focal + (1 - focal_weight) * bce
 
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)
 
def evaluate(model, loader, evaluator, device, save_path=None, return_attention=False):
    model.eval()
    y_true, y_pred, att_scores_list = [], [], []
    for i, data in enumerate(loader):
        data = data.to(device)
        with torch.no_grad():
            if return_attention:
                out, att_scores = model(data, return_attention=True)
                att_scores_list.append((att_scores, data))
            else:
                out = model(data)
        out = torch.sigmoid(out)
        y_true.append(data.y.cpu())
        y_pred.append(out.cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    if save_path:
        # Extract attention scores from att_scores_list and convert to a list of numpy arrays
        att_scores_only = [att_scores.cpu().numpy() for att_scores, _ in att_scores_list] if return_attention else []
        torch.save({"y_true": y_true, "y_pred": y_pred, "attention_scores": att_scores_only}, save_path)
    result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]
    if return_attention:
        return result, att_scores_list
    return result
 
def run_experiment(use_edge_attr, use_input_transform, use_atom_embedding, pooling_type, hidden_dims, dropout_rate, aggr_type, atom_embed_dim, edge_embed_dim, device, dataset, split_idx):
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=1, shuffle=False)
 
    model = ConvGCN(
        input_dim=dataset.num_node_features,
        edge_dim=dataset[0].edge_attr.size(-1),
        hidden_dims=hidden_dims,
        output_dim=1,
        use_edge_attr=use_edge_attr,
        use_input_transform=use_input_transform,
        use_atom_embedding=use_atom_embedding,
        pooling_type=pooling_type,
        dropout_rate=dropout_rate,
        aggr_type=aggr_type,
        atom_embed_dim=atom_embed_dim,
        edge_embed_dim=edge_embed_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75, eta_min=1e-5)
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)
    criterion = combined_loss
    evaluator = Evaluator(name="ogbg-molhiv")
 
    train_losses, valid_aucs = [], []
    best_valid_auc = 0
    patience = 20
    patience_counter = 0
    suffix = f"edge_{'on' if use_edge_attr else 'off'}_transform_{'on' if use_input_transform else 'off'}_atom_embed_{'on' if use_atom_embedding else 'off'}_pool_{pooling_type}_dims_{'_'.join(map(str, hidden_dims))}_dropout_{dropout_rate}_aggr_{aggr_type}_atom_embed_dim_{atom_embed_dim}_edge_embed_dim_{edge_embed_dim}"
    for epoch in range(1, 101):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        valid_auc = evaluate(model, valid_loader, evaluator, device)
        if epoch <= 50:
            scheduler_cosine.step()
        else:
            scheduler_plateau.step(valid_auc)
        train_losses.append(train_loss)
        valid_aucs.append(valid_auc)
        print(f"ConvGCN ({suffix}), Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Valid AUC: {valid_auc:.4f}")
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            torch.save(model.state_dict(), f"conv_gcn_{suffix}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
 
    model.load_state_dict(torch.load(f"conv_gcn_{suffix}.pth"))
    eval_result = evaluate(
        model, test_loader, evaluator, device,
        save_path=f"conv_gcn_pred_{suffix}.pt",
        return_attention=(pooling_type == "attention")
    )
    if pooling_type == "attention":
        test_auc, att_scores_list = eval_result
    else:
        test_auc = eval_result
        att_scores_list = []
    
    print(f"ConvGCN ({suffix}), Test AUC: {test_auc:.4f}")
 
    
    plot_training_metrics(train_losses, valid_aucs, output_dir=f"plots_{suffix}")
    predictions = torch.load(f"conv_gcn_pred_{suffix}.pt")
    plot_roc_curve(predictions["y_true"], predictions["y_pred"], output_dir=f"plots_{suffix}")
    
    if pooling_type == "attention":
        for i, (att_scores, data) in enumerate(att_scores_list[:5]):
            data = data.to(device)
            y_true = data.y.cpu().numpy()[0].item()
            with torch.no_grad():
                y_pred = torch.sigmoid(model(data, return_attention=False)).cpu().detach().numpy()[0].item()
            plot_attention_scores(data, att_scores, y_true, y_pred, i, output_dir=f"plots_{suffix}")
 
def main():
    parser = argparse.ArgumentParser(description="Run ConvGCN with specified parameters on ogbg-molhiv dataset")
    parser.add_argument('--hidden_dims', type=int, nargs=3, default=[192, 384, 768],
                        help="Hidden dimensions for the 3 GCNConv layers (e.g., --hidden_dims 192 384 768)")
    parser.add_argument('--pooling_type', type=str, default="attention", choices=["mean", "sum", "max", "attention"],
                        help="Graph-level pooling type: 'mean', 'sum', 'max', or 'attention'")
    parser.add_argument('--use_input_transform', type=int, default=1, choices=[0, 1],
                        help="Whether to use input transformation (0=False, 1=True)")
    parser.add_argument('--use_atom_embedding', type=int, default=1, choices=[0, 1],
                        help="Whether to use atom embedding (0=False, 1=True)")
    parser.add_argument('--use_edge_attr', type=int, default=1, choices=[0, 1],
                        help="Whether to use edge attributes (0=False, 1=True)")
    parser.add_argument('--aggr_type', type=str, default="add", choices=["add", "mean", "max"],
                        help="Message-passing aggregation type: 'add', 'mean', or 'max'")
    parser.add_argument('--atom_embed_dim', type=int, default=64,
                        help="Dimension of atom embedding")
    parser.add_argument('--edge_embed_dim', type=int, default=16,
                        help="Dimension of edge embedding")
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help="Dropout rate")
 
    args = parser.parse_args()
 
    hidden_dims = args.hidden_dims
    pooling_type = args.pooling_type
    use_input_transform = bool(args.use_input_transform)
    use_atom_embedding = bool(args.use_atom_embedding)
    use_edge_attr = bool(args.use_edge_attr)
    aggr_type = args.aggr_type
    atom_embed_dim = args.atom_embed_dim
    edge_embed_dim = args.edge_embed_dim
    dropout_rate = args.dropout_rate
 
    device = torch.device("cpu")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx = dataset.get_idx_split()
 
    print(f"Running ConvGCN with hidden_dims={hidden_dims}, pooling_type={pooling_type}, use_input_transform={use_input_transform}, use_atom_embedding={use_atom_embedding}, use_edge_attr={use_edge_attr}, aggr_type={aggr_type}, atom_embed_dim={atom_embed_dim}, edge_embed_dim={edge_embed_dim}, dropout_rate={dropout_rate}")
 
    run_experiment(
        use_edge_attr=use_edge_attr,
        use_input_transform=use_input_transform,
        use_atom_embedding=use_atom_embedding,
        pooling_type=pooling_type,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        aggr_type=aggr_type,
        atom_embed_dim=atom_embed_dim,
        edge_embed_dim=edge_embed_dim,
        device=device,
        dataset=dataset,
        split_idx=split_idx
    )
 
if __name__ == "__main__":
    main()