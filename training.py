"""
Adapted from: https://github.com/yzhuoning/DeepAUC_OGB_Challenge
"""

import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from libauc.utils import set_all_seeds
from libauc.losses import AUCMLoss, CompositionalAUCLoss
from libauc.optimizers import PESG, PDSCA
from torch.nn import BCELoss
from libauc.models import DeeperGCN
from libauc.sampler import DualSampler

def train(model, device, loader, optimizer, criterion):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
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
        pred = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)
    return result_dict

def main():
    logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Training Parameters
    loss_fn = "aucm"
    load_pretrained_model = False
    batch_size = 512
    learning_rate = 0.1
    weight_decay = 0.00001
    margin = 1.0
    epoch_decay = 0.002
    epochs = 100
    decay_epochs = [int(epochs * 0.5), int(epochs * 0.75)]
    sampling_rate = 0.3
    beta0 = 0.9
    beta1 = 0.999
    k = 3
    seed = 0

    # Model Paramters
    aggregation = "softmax"
    t = 1.0
    p = 2.0
    hidden_size = 256
    num_layers = 14
    dropout = 0.2


    set_all_seeds(seed)
    device = torch.device("cuda:0")
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")


    split_idx = dataset.get_idx_split()
    train_set = dataset[split_idx["train"]]
    valid_set = dataset[split_idx["valid"]]
    test_set = dataset[split_idx["test"]]
    sampler = DualSampler(train_set, batch_size, sampling_rate=sampling_rate, labels=train_set.y, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    evaluator = Evaluator(name="ogbg-molhiv")

    model = DeeperGCN(num_tasks=dataset.num_tasks,
                      emb_dim=hidden_size,
                      num_layers=num_layers,
                      dropout=dropout,
                      aggr=aggregation,
                      t=t,
                      learn_t=True,
                      p=p,
                      learn_p=True,
                      block="res+"
                      ).to(device)
    logging.info(model)

    if load_pretrained_model:
        PATH = 'pretrained_model.pth'
        state_dict = torch.load(PATH)
        msg = model.load_state_dict(state_dict, False)
        print(msg)

    if loss_fn == "aucm":
        criterion = AUCMLoss()
        optimizer = PESG(model.parameters(), lr=learning_rate, weight_decay=weight_decay, loss_fn=criterion, margin=margin, epoch_decay=epoch_decay)
    elif loss_fn == "ce":
        criterion = BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif loss_fn == "compauc":
        criterion = CompositionalAUCLoss(k=k)
        optimizer = PDSCA(model.parameters(), lr=learning_rate, weight_decay=weight_decay, loss_fn=criterion, beta1=beta0, beta2=beta1, margin=margin, epoch_decay=epoch_decay, version="v2")
    train_log = []
    valid_log = []
    test_log = []

    best_valid = 0
    final_test = 0
    for epoch in range(1, epochs + 1):
        if epoch in decay_epochs and loss_fn == "aucm":
            optimizer.update_regularizer(decay_factor=10)
        if epoch in decay_epochs and loss_fn == "compauc":
            optimizer.update_regularizer(decay_factor=10, decay_factor0=10)
        if epoch in decay_epochs and loss_fn == "ce":
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * param_group['lr']
        logging.info("Epoch {}".format(epoch))
        logging.info('Training...')

        epoch_loss = train(model, device, train_loader, optimizer, criterion)

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
        if valid_result > best_valid:
            best_valid = valid_result
            final_test = test_result
            torch.save(model.state_dict(), 'best_model.pth')
    print(best_valid)
    print(final_test)
    plt.plot(train_log, label='Train')
    plt.plot(valid_log, label='Validation')
    plt.plot(test_log, label='Test')
    plt.show()




if __name__ == '__main__':
    main()