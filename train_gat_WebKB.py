import wandb
import dgl
import networkx as nx
from dgl.data.citation_graph import load_citeseer
import time
import torch
from models.GAT import GAT
import numpy as np
from torch import nn
from torch.nn import init
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity

import dgl
import networkx as nx
import scipy.sparse as sp
from dgl import DGLGraph

class WebKBset(object):
    r"""Cora citation network dataset. Nodes mean author and edges mean citation
    relationships.
    """

    def __init__(self, name):
        self.name = name
        self._load()

    def _load(self):
        idx_features_labels = np.genfromtxt("data/{}.content".format(self.name), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = _encode_onehot(idx_features_labels[:, -1])
        self.num_labels = labels.shape[1]
        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=str)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("data/{}.cites".format(self.name), dtype=np.dtype(str))
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        self.graph = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())

        features = _normalize(features)
        self.features = np.array(features.todense())
        self.labels = np.where(labels)[1]
        self.train_mask = _sample_mask(range(int(0.5 * labels.shape[0])), labels.shape[0])

        self.val_mask = _sample_mask(range(int(0.5 * labels.shape[0]) + 1, int(0.8 * labels.shape[0])), labels.shape[0])

        self.test_mask = _sample_mask(range(int(0.8 * labels.shape[0]) + 1, int(labels.shape[0])), labels.shape[0])

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        g = DGLGraph(self.graph)
        g.ndata['train_mask'] = self.train_mask
        g.ndata['val_mask'] = self.val_mask
        g.ndata['test_mask'] = self.test_mask
        g.ndata['label'] = self.labels
        g.ndata['feat'] = self.features
        return g

    def __len__(self):
        return 1


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

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
    exp_name = "GAT_texas"

    #filename = "cornell"
    filename="texas"
    # filename="washington"
    # filename="wisconsin"
    data = WebKBset(filename)

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

    wandb.init(project='ai607', name=exp_name)
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
