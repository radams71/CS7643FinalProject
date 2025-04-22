import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
 
class ConvMPNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConvMPNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, output_dim)
 
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()  
        x = F.relu(self.conv1(x, edge_index))  
        x = F.relu(self.conv2(x, edge_index))  
        x = global_mean_pool(x, batch)  
        return torch.sigmoid(self.output(x))  
 
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
 
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)
 
def evaluate(model, loader, evaluator, device):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        y_true.append(data.y.cpu())
        y_pred.append(out.cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]
 
def main():
    device = torch.device("cpu")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    print(f"Number of node features: {dataset.num_node_features}")  # Should be 9
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
    model = ConvMPNN(
        input_dim=dataset.num_node_features,  # 9 for ogbg-molhiv
        hidden_dim=64,
        output_dim=1
    ).to(device)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    evaluator = Evaluator(name="ogbg-molhiv")
    best_valid_auc = 0
 
    for epoch in range(1, 251):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        valid_auc = evaluate(model, valid_loader, evaluator, device)
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Valid AUC: {valid_auc:.4f}")
 
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            torch.save(model.state_dict(), "best_model.pth")
 
    model.load_state_dict(torch.load("best_model.pth"))
    test_auc = evaluate(model, test_loader, evaluator, device)
    print(f"Best Test AUC: {test_auc:.4f}")
 
if __name__ == "__main__":
    main()