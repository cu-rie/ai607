import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GAT(nn.Module):
    def __init__(self,
                 input_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads=3,
                 activation=torch.relu,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(
            GATConv(input_dim, num_hidden, heads, feat_drop, attn_drop, negative_slope, residual, activation))

        for _ in range(num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConv(
                num_hidden*heads, num_hidden, heads, feat_drop, attn_drop, negative_slope, residual, activation))
        # output projection
        self.layers.append(GATConv(
            num_hidden * heads, num_classes, heads, feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, graph, inputs):
        h = inputs
        for l in range(len(self.layers) - 1):
            h = self.layers[l](graph, h).flatten(1)
        # output projection
        logits = self.layers[-1](graph, h).mean(1)
        return logits


