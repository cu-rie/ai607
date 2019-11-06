import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np

gcn_msg = fn.copy_src(src='feat', out='msg')
gcn_reduce = fn.sum(msg='msg', out='feat')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        feat = self.linear(node.data['feat'])
        feat = self.activation(feat)
        return {'feat': feat}


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(input_dim, output_dim, activation)

    def forward(self, g, feature):
        g.ndata['feat'] = feature
        g.update_all(message_func=gcn_msg, reduce_func=gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('feat')


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=[64, 64], hidden_activation=F.relu, out_activation=F.relu):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()

        layer = GCNLayer(input_dim, hidden[0], hidden_activation)
        self.layers.append(layer)
        for i in range(len(hidden) - 1):
            layer = GCNLayer(hidden[i], hidden[i + 1], hidden_activation)
            self.layers.append(layer)
        layer = GCNLayer(hidden[-1], output_dim, out_activation)
        self.layers.append(layer)

    def forward(self, g, features):
        for l in self.layers:
            features = l(g, features)
        return features


if __name__ == "__main__":
    from utils.load_data import load_cora_data

    g, input_features, labels, mask, num_labels = load_cora_data()
    net = GCN(input_dim=input_features.shape[-1], output_dim=7)
    lr = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    dur = []
    for epoch in range(30):
        if epoch >= 3:
            t0 = time.time()

        logits = net(g, input_features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))
