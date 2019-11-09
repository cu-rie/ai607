import torch
import torch.nn as nn
from models.Ours_layer import OurLayer


class OurNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_nodes,
                 use_linear_comb=True,
                 use_intermediate_embedding=True,
                 activation=torch.relu,
                 dropout=0.5):
        super(OurNetwork, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            OurLayer(input_dim, n_hidden, n_nodes, activation=activation, use_linear_comb=use_linear_comb))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                OurLayer(n_hidden, n_hidden, n_nodes, activation=activation, use_linear_comb=use_linear_comb))
        # output layer
        self.layers.append(OurLayer(n_hidden, n_classes, n_nodes, use_linear_comb=use_linear_comb))
        self.dropout = nn.Dropout(p=dropout)

        self.use_intermediate_embedding = use_intermediate_embedding
        if use_intermediate_embedding:
            embedding_dim = [input_dim] + [n_hidden] * n_layers + [n_classes]
            self.intermediate_weight = torch.nn.Parameter(torch.Tensor(embedding_dim), requires_grad=True)

    def forward(self, graph, features):
        if self.use_intermediate_embedding:
            intermediate_embedding = []
            h = features
            intermediate_embedding.append(h)
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(graph, h)
                intermediate_embedding.append()

        else:
            h = features
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(graph, h)
        return h


if __name__ == "__main__":
    import dgl
    from dgl.data.citation_graph import load_citeseer

    data = load_citeseer()

    graph = dgl.DGLGraph(data.graph)
    features = torch.Tensor(data.features)
    train_mask = torch.LongTensor(data.train_mask).bool()
    test_mask = torch.LongTensor(data.test_mask).bool()
    val_mask = torch.LongTensor(data.val_mask).bool()
    labels = torch.LongTensor(data.labels)
    num_labels = data.num_labels

    # model arguments

    input_dim = features.shape[1]
    n_hidden = 64
    n_layers = 1

    model = OurNetwork(input_dim, n_hidden, num_labels, n_layers, graph.number_of_nodes())
    model(graph, features)