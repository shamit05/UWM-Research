from visualize_featureids import gen_graph
import os
import pathlib

import numpy as np
from torch_geometric.utils import from_networkx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

DIR_LOC = pathlib.Path(__file__).parents[1] # /Research Directory
GEN_STRUCTURES_FILE_BASE = os.path.join(DIR_LOC, "generated_microstructures", "FeatureData_FakeMatl_")

def network_to_pyg_data(file):
    G = gen_graph(file)

    pyg_graph = from_networkx(G, group_node_attrs=["pos", "size", "rot"], group_edge_attrs=["weight"])
    pyg_graph.y = pyg_graph.surfaceFeature
    del pyg_graph.surfaceFeature
    # Split the data
    train_ratio = 0.2
    num_nodes = pyg_graph.x.shape[0]
    num_train = int(num_nodes * train_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_mask = torch.full_like(pyg_graph.y, False, dtype=bool)
    train_mask[idx[:num_train]] = True
    test_mask = torch.full_like(pyg_graph.y, False, dtype=bool)
    test_mask[idx[num_train:]] = True

    data = pyg_graph
    data.train_mask = train_mask
    data.test_mask = test_mask

    return data

data_batch = []

for i in range(0, 9):
    file = GEN_STRUCTURES_FILE_BASE + str(i) + ".csv"
    print("Loading graph " + str(i) + "...")
    data_batch.append(network_to_pyg_data(file))

# loader to combine data
print("Combining data...")

loader = DataLoader(data_batch, batch_size=32)
data = next(iter(loader))

# GCN model with 2 layers
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, int(data.y.max() + 1))

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device)

model = Net().to(device)

torch.manual_seed(42)

optimizer_name = "Adam"
lr = 1e-1
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
epochs = 200

def train():
  model.train()
  optimizer.zero_grad()
  F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
  optimizer.step()

@torch.no_grad()
def test():
  model.eval()
  logits = model()
  mask1 = data['train_mask']
  pred1 = logits[mask1].max(1)[1]
  acc1 = pred1.eq(data.y[mask1]).sum().item() / mask1.sum().item()
  mask = data['test_mask']
  pred = logits[mask].max(1)[1]
  acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
  return acc1,acc

print("Training...")
for epoch in range(1, epochs):
    print("Epoch " + str(epoch) + "...")
    train()
    train_acc, test_acc = test()
    print('Train Accuracy: %s' % train_acc)

train_acc,test_acc = test()

print('#' * 70)
print('Train Accuracy: %s' %train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)