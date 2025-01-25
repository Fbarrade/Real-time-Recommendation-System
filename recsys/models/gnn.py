import torch, torch.nn as nn 
import torch.nn.functional as F

import torch_geometric as tg 
import torch_geometric.nn as gnn 
from torch_geometric.loader import DataLoader as GDataLoader

from dataclasses import dataclass


__all__ = ["GNNModel", "GNNConfigs", "train_gnn_model", "train_step_optuna"]



@dataclass
class GNNConfigs:
    in_channels: int =  495
    hidden_channels: int = 16 
    out_channels: int = 1

    num_gnn_layers: int = 3
    num_fc_layers: int = 2


class GNNModel(torch.nn.Module):
    def __init__(self, configs: GNNConfigs):
        super(GNNModel, self).__init__()
        self.configs = configs 

        self.gnn = nn.ModuleList()
        self.fc = nn.ModuleList()

        for i in range(self.configs.num_gnn_layers):
            if i == 0: 
                self.add_gnn_block(is_first=True)
            else:
                self.add_gnn_block()

        for j in range(self.configs.num_fc_layers):
            if j == 0:
                if self.configs.num_fc_layers != 1:
                    self.add_fc_block(is_first=True)
                else:
                    self.add_fc_block(is_first=True, is_last=True)
            elif j == self.configs.num_fc_layers - 1:
                self.add_fc_block(is_last=True)
            else:
                self.add_fc_block()

    def add_gnn_block(self, is_first: bool = False):
        
        in_channels = (self.configs.in_channels if is_first else self.configs.hidden_channels)
        
        self.gnn.append(
            gnn.GCNConv(in_channels, self.configs.hidden_channels)
        )
        self.gnn.append(nn.ReLU())

    def add_fc_block(self, is_first: bool = False, is_last: bool = False):

        in_channels = self.configs.hidden_channels
        if is_first: in_channels *= 2

        if not is_last:
            self.fc.append(
                nn.Linear(in_channels, self.configs.hidden_channels)
            )
            self.fc.append(nn.ReLU())
        else:
            self.fc.append(nn.Linear(in_channels, self.configs.out_channels))


    def forward(self, data: tg.data.Data):
        x, edge_index = data.x, data.edge_index

        for gnn_l in self.gnn:
            if isinstance(gnn_l, gnn.GCNConv):
                x = gnn_l(x, edge_index)
            else:
                x = gnn_l(x)

        x = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        for fc_l in self.fc:
            x = fc_l(x)

        return x.squeeze()


def train_gnn_model(
    model: GNNModel, trainset, testset, run
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

        validate_gnn_model(
            model=model, testset=testset, run=run, is_val= False
        )
        opt.step()
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')

@torch.no_grad()
def validate_gnn_model(
    model: GNNModel, testset, run, is_val = False
):
    loss_fn = nn.MSELoss()

    model.eval()
    out = model(testset)
    loss = loss_fn(out, testset.edge_attr.view(-1, 1))

    run.log({f"{'Val' if is_val else 'Test'}/Loss": loss.item()})


def train_step_optuna(trial, trainset):

    n_gnn_layers = trial.suggest_int("n_gnn_layers", 1, 3)
    n_fc_layers = trial.suggest_int("n_fc_layers", 1, 2)
    num_iters = trial.suggest_int("num_iters", 1, 20)

    hidden_channels = trial.suggest_int("hidden_channels", 4, 128)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model = GNNModel(
        configs= GNNConfigs(
            hidden_channels=hidden_channels, num_gnn_layers= n_gnn_layers, num_fc_layers=n_fc_layers
        )
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    
    total_loss = 0

    for itr in range(num_iters):
        opt.zero_grad()
        out = model(trainset)

        loss = loss_fn(out, trainset.edge_attr.view(-1, 1))
        loss.backward()
        opt.step()

        total_loss += loss.item()

    return total_loss





    

