from visualize_featureids import gen_graph
import os
import pathlib

import numpy as np
from torch_geometric.utils import from_networkx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

import time
import ray
from ray import tune
from ray.tune.tune_config import TuneConfig
from ray.air.config import RunConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

DIR_LOC = pathlib.Path(__file__).parents[1]  # /Research Directory
GEN_STRUCTURES_FILE_BASE = os.path.join(DIR_LOC, "generated_microstructures", "FeatureData_FakeMatl_")


def network_to_pyg_data(file):
    G = gen_graph(file)
    pyg_graph = from_networkx(G, group_node_attrs=["pos"], group_edge_attrs=["weight"])
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


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_neurons, network_size):
        super(GCN, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(num_features, num_neurons))
        for _ in range(network_size - 1):
            self.conv_layers.append(GCNConv(num_neurons, num_neurons))
        self.conv_layers.append(GCNConv(num_neurons, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.conv_layers[:-1]:
            x = conv(x, edge_index)
            x = x.relu()

        x = self.conv_layers[-1](x, edge_index)

        return x.log_softmax(dim=1)


def train_gcn(config, data):
    num_neurons = int(config["num_neurons"].sample())
    network_size = int(config["network_size"].sample())
    lr = float(config["lr"].sample())
    weight_decay = float(config["weight_decay"].sample())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(data.num_node_features, int(data.y.max() + 2), num_neurons, network_size).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    # Report the accuracy metric to Ray Tune
    tune.report(accuracy=acc)

    return {"accuracy": acc}


def tune_hyperparameters(config, metric, mode):
    data_batch = []
    for i in range(0,99):
        # print("Loading graph " + str(i) + "...")
        file = GEN_STRUCTURES_FILE_BASE + str(i) + ".csv"
        data_batch.append(network_to_pyg_data(file))

    loader = DataLoader(data_batch, batch_size=32)
    data = next(iter(loader))

    # Perform hyperparameter tuning using Ray Tune
    reporter = CLIReporter(metric_columns=["accuracy"])
    scheduler = ASHAScheduler(metric=metric, mode=mode, max_t=500, grace_period=20)
    trainable_with_resources = tune.with_resources(lambda trainable: train_gcn(config, data), {"cpu": 8, "gpu": 1})

    # store results using tune.Tuner
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space= {
            "params": config
        },
        tune_config=TuneConfig(
            num_samples=10,
            # time_budget_s=600.0,
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            storage_path = "./ray_results",  # Specify a directory to store results
            progress_reporter=reporter,
        ),
    )
    result = tuner.fit()
    best_trial = result.get_best_result("accuracy", mode="max", scope="last")
    best_hyperparameters = best_trial.config
    best_accuracy = best_trial.metrics

    print("Best hyperparameters found:")
    print(best_hyperparameters)
    print("Best accuracy found:", best_accuracy)



if __name__ == "__main__":
    print('CUDA' if torch.cuda.is_available() else 'CPU')
    ray.init()

    # Define the search space for hyperparameters
    config = {
        "num_neurons": tune.choice([32, 64, 128]),
        "network_size": tune.choice([2, 3, 4]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-3)
    }

    # Define the metric and mode for the ASHAScheduler
    metric = "accuracy"
    mode = "max"

    # Perform hyperparameter tuning
    tune_hyperparameters(config, metric, mode)

    ray.shutdown()