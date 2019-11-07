# import wandb
import dgl
import networkx as nx
from dgl.data.citation_graph import load_citeseer
import time
import torch
from utils.train_utils import evaluate
from models.GAT import GAT
import numpy as np

if __name__ == "__main__":

    data = load_citeseer()

    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))

    graph = dgl.DGLGraph(g)
    graph.add_edges(graph.nodes(), graph.nodes())

    features = torch.Tensor(data.features)
    train_mask = torch.LongTensor(data.train_mask)
    test_mask = torch.LongTensor(data.test_mask)
    val_mask = torch.LongTensor(data.val_mask)
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

    # wandb.init(project='ai607', name="GCN_cora")
    # wandb.watch(model)

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
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, graph, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
