import torch
import argparse, time
import numpy as np

from models.GCN import GCN
from models.GAT import GAT

# from utils.load_data import load_cora_data
from dgl.data import load_data, register_data_args
from dgl.data.citation_graph import load_cora, load_citeseer, load_pubmed, load_synthetic


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


if __name__ == "__main__":
    import wandb
    import dgl

    data = load_cora()

    graph = dgl.DGLGraph(data.graph)
    features = torch.Tensor(data.features)
    train_mask = torch.LongTensor(data.train_mask)
    test_mask = torch.LongTensor(data.test_mask)
    val_mask = torch.LongTensor(data.val_mask)
    labels = torch.LongTensor(data.labels)
    num_labels = data.num_labels

    # model arguments

    input_dim = features.shape[-1]
    n_hidden = 64
    n_layers = 3

    model = GCN(input_dim, n_hidden, num_labels, n_layers)

    # train arguments

    lr = 1e-3
    n_epochs = 200

    loss_ftn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dur = []

    wandb.init(project='ai607', name="GCN_cora")
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
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, graph, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))
