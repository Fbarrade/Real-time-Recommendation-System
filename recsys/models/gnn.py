import torch, torch.nn as nn 
import torch.nn.functional as F

import torch_geometric as tg 
import torch_geometric.nn as gnn 
from torch_geometric.loader import DataLoader as GDataLoader

from dataclasses import dataclass


__all__ = ["GNNModel", "GNNConfigs", "train_gnn_model"]



@dataclass
class GNNConfigs:
    in_channels: int =  495
    hidden_channels: int = 16 
    out_channels: int = 1


class GNNModel(torch.nn.Module):
    def __init__(self, configs: GNNConfigs):
        super(GNNModel, self).__init__()
        self.configs = configs 

        self.conv1 = gnn.GCNConv(self.configs.in_channels, self.configs.hidden_channels)
        self.conv2 = gnn.GCNConv(self.configs.hidden_channels, self.configs.hidden_channels)
        self.fc = torch.nn.Linear(self.configs.hidden_channels * 2, self.configs.out_channels)

    def forward(self, data: tg.data.Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        edge_pred = self.fc(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1))
        return edge_pred.squeeze()


def train_gnn_model(
    model: GNNModel, trainset, run
):
    opt = torch.optim.Adam(model.parameters(), lr=run.config.get("lr"))
    loss_fn = nn.MSELoss()

    for epoch in range(run.config.get("epochs")):

        model.train()

        total_loss = 0
      
        opt.zero_grad()
        out = model(trainset)

        loss = loss_fn(out, trainset.edge_attr.view(-1, 1))
        loss.backward()

        total_loss += loss.item()

        run.log({"Train/Loss": total_loss})

        opt.step()

        print(f'Epoch {epoch + 1}, Loss: {total_loss}')


    

