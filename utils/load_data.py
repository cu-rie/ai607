import torch
import dgl
from dgl.data import citation_graph as citegrh
import networkx as nx


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = dgl.DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    num_labels = 7
    return g, features, labels, mask, num_labels


if __name__ == "__main__":
    g, feat, label, mask, num_labels = load_cora_data()
