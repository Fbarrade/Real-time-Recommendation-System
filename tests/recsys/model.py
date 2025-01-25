from recsys.models import GNNConfigs, GNNModel
import torch 

from torch_geometric.data import Data

model = GNNModel(
    configs=GNNConfigs(
        num_gnn_layers=3,
        num_fc_layers=2
    ))

x = torch.rand(100, 495)

index = torch.randint(2, (2, 4))

data = Data(x, index)

out = model(data)

print(out)

