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
    pyg_graph.y = pyg_graph["surfaceFeature"]
    del pyg_graph["surfaceFeature"]
    pyg_graph.y = pyg_graph.y.type(torch.LongTensor)

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
for i in range(0, 30):
    file = GEN_STRUCTURES_FILE_BASE + str(i) + ".csv"
    print("Loading graph " + str(i) + "...")
    data_batch.append(network_to_pyg_data(file))

print(data_batch)
# loader to combine data
print("Combining data...")

loader = DataLoader(data_batch, batch_size=32)
data = next(iter(loader))

# GCN model with 2 layers
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 32)
        self.conv2 = GCNConv(32, int(data.y.max() + 2))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    print("Epoch: " + str(epoch))
    print(f'Loss: {loss:.4f}, Accuracy: {acc:.4f}')