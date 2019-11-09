import wandb
import dgl
import networkx as nx
from dgl.data.citation_graph import load_citeseer
import time
import torch
from utils.train_utils import evaluate
from models.GAT import GAT
import numpy as np

if __name__ == "__main__":
    exp_name = "GAT_citeseer"

    data = load_citeseer()

    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))

    graph = dgl.DGLGraph(g)
    graph.add_edges(graph.nodes(), graph.nodes())

    features = torch.Tensor(data.features)
    train_mask = torch.LongTensor(data.train_mask).bool()
    test_mask = torch.LongTensor(data.test_mask).bool()
    val_mask = torch.LongTensor(data.val_mask).bool()
    labels = torch.LongTensor(data.labels)
    num_labels = data.num_labels

    # model arguments

    input_dim = features.shape[1]
    n_hidden = 64
    n_layers = 3

    model = GAT(input_dim=input_dim, num_hidden=n_hidden, num_classes=num_labels, num_layers=n_layers)

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
