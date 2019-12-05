import torch
import argparse, time
import numpy as np

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
        labels = self._encode_onehot(idx_features_labels[:, -1])
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

        features = self._normalize(features)
        self.features = np.array(features.todense())
        self.labels = np.where(labels)[1]
        self.train_mask = self._sample_mask(range(int(0.5 * labels.shape[0])), labels.shape[0])

        self.val_mask = self._sample_mask(range(int(0.5 * labels.shape[0]) + 1, int(0.8 * labels.shape[0])),
                                          labels.shape[0])

        self.test_mask = self._sample_mask(range(int(0.8 * labels.shape[0]) + 1, int(labels.shape[0])), labels.shape[0])

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

    @staticmethod
    def _sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return mask

    @staticmethod
    def _normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    @staticmethod
    def _encode_onehot(labels):
        classes = list(sorted(set(labels)))
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    @staticmethod
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
    # filename = "cornell"
    filename = "texas"
    # filename="washington"
    # filename="wisconsin"
    data = WebKBset(filename)
    use_intermediate_embedding = True
    use_linear_comb = False

    graph = dgl.DGLGraph(data.graph)
    features = torch.Tensor(data.features)
    train_mask = torch.LongTensor(data.train_mask).bool()
    test_mask = torch.LongTensor(data.test_mask).bool()
    val_mask = torch.LongTensor(data.val_mask).bool()
    labels = torch.LongTensor(data.labels)
    num_labels = data.num_labels
