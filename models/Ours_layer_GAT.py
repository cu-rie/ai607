import torch
from torch import nn
from torch.nn import init

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity


class OurLayer_GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_nodes,
                 num_heads=3,
                 use_linear_comb=True,
                 activation=None):
        super(OurLayer_GAT, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.use_linear_comb = use_linear_comb
        if use_linear_comb:
            if self._in_feats > self._out_feats:
                self.linear_comb = nn.Parameter(torch.ones(n_nodes) * 0.5, requires_grad=True)
            else:
                self.linear_comb = nn.Parameter(torch.ones(n_nodes) * 0.5, requires_grad=True)

        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        # self.feat_drop = nn.Dropout(feat_drop)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, feat):
        h = feat
        graph = graph.local_var()
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = edge_softmax(graph, e)
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         self.reduce_func)
        rst = graph.ndata['ft']
        # linear_comb

        # activation
        if self._activation is not None:
            rst = self._activation(rst)
        return rst

    def reduce_func(self, nodes):
        msg = nodes.mailbox['m'].sum(1)
        hidden = nodes.data['ft']
        if self.use_linear_comb:
            lin = self.linear_comb[nodes.nodes()].unsqueeze(1)
            out = (1 - lin) * msg + lin * hidden
        else:
            out = msg
        return {'out': out}
