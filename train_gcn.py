import torch
import argparse, time
import numpy as np

from models.GCN import GCN
from utils.train_utils import evaluate
from models.GAT import GAT

from dgl.data import load_data, register_data_args
from dgl.data.citation_graph import load_cora, load_citeseer, load_pubmed, load_synthetic

if __name__ == "__main__":
    import wandb
    import dgl

    exp_name = 'GCN_citeceer'
    data = load_citeseer()

    graph = dgl.DGLGraph(data.graph)
    features = torch.Tensor(data.features)
    train_mask = torch.LongTensor(data.train_mask)
    test_mask = torch.LongTensor(data.test_mask)
    val_mask = torch.LongTensor(data.val_mask)
    labels = torch.LongTensor(data.labels)
    num_labels = data.num_labels

    # model arguments

    input_dim = features.shape[1]
    n_hidden = 64
    n_layers = 1

    model = GCN(input_dim, n_hidden, num_labels, n_layers)
    # model = GAT(input_dim, n_hidden, num_labels, n_layers, heads=3)

    # train arguments

    lr = 1e-3
    n_epochs = 200

    loss_ftn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dur = []

    wandb.init(project='kaistai607', name=exp_name)
    wandb.watch(model)

    n_edges = graph.number_of_edges()

    for epoch in range(n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
            # forward
        logits = model(graph, features)
        loss = loss_ftn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, graph, features, labels, val_mask)
        wandb.log({'epoch': epoch,
                   'time': np.mean(dur),
                   'loss': loss.item(),
                   'acc': acc,

                   })

    acc = evaluate(model, graph, features, labels, test_mask)
    wandb.log({'test_acc': acc

               })
