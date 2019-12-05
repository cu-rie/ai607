import torch
from torch import nn
from torch.nn import init

import dgl.function as fn


class OurLayer_GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_nodes,

                 use_linear_comb=1,
                 norm=True,
                 bias=True,
                 activation=None):
        super(OurLayer_GCN, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.use_linear_comb = use_linear_comb
        if use_linear_comb==1:
            self.linear_comb = nn.Parameter(torch.ones(n_nodes) * 0.5, requires_grad=True)
        else:
            self.register_parameter('linear_comb', None)

        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat):
        graph.ndata['h'] = feat
        if self._norm:
            norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = torch.matmul(feat, self.weight)
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             self.reduce_func)
            # fn.sum(msg='m', out='h'))
            rst = graph.ndata.pop('out')
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             self.reduce_func)
            # fn.sum(msg='m', out='h'))
            rst = graph.ndata.pop('out')
            rst = torch.matmul(rst, self.weight)

        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def reduce_func(self, nodes):
        msg = nodes.mailbox['m'].sum(1)
        hidden = nodes.data['h']
        if self.linear_comb is not None:
            lin = self.linear_comb[nodes.nodes()].unsqueeze(1)
            out = (1 - lin) * msg + lin * hidden
        else:
            out = msg
        return {'out': out}
