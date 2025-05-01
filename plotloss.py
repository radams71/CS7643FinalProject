import matplotlib.pyplot as plt
import torch


train_log_compauc = torch.load("train_log_compauc.pt", weights_only=False)
valid_log_compauc = torch.load("valid_loss_compauc.pt", weights_only=False)
train_loss_compauc = torch.load("train_loss_compauc.pt", weights_only=False)
valid_loss_compauc = torch.load("valid_loss_compauc.pt", weights_only=False)

train_log_aucm = torch.load("train_log_aucm.pt", weights_only=False)
valid_log_aucm = torch.load("valid_log_aucm.pt", weights_only=False)
train_loss_aucm = torch.load("train_loss_aucm.pt", weights_only=False)
valid_loss_aucm = torch.load("valid_loss_aucm.pt", weights_only=False)

train_log_ce = torch.load("train_log_ce.pt", weights_only=False)
valid_log_ce = torch.load("valid_log_ce.pt", weights_only=False)
train_loss_ce = torch.load("train_loss_ce.pt", weights_only=False)
valid_loss_ce = torch.load("valid_loss_ce.pt", weights_only=False)


plt.plot(train_loss_compauc, label="train compauc")
plt.plot(train_loss_aucm, label="train aucm")
plt.plot(train_loss_ce, label="train ce")
# plt.plot(valid_log_compauc, label="valid compauc")
# plt.plot(valid_log_aucm, label="valid aucm")
# plt.plot(valid_log_ce, label="valid ce")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("loss_curves.png")