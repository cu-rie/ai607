import torch
import torch.nn as nn
from models.layers.graphconv import GraphConv


class GCN(nn.Module):
    def __init__(self,
                 input_dim,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation=torch.relu,
                 dropout=0.):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(input_dim, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, features):

        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h
