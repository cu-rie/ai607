import torch
import torch.nn as nn
from models.layers.OurLayer_GCN import OurLayer_GCN
import torch.nn.functional as F

class OurNetwork_GCN(nn.Module):
    def __init__(self,
                 input_dim,
                 n_hidden,
                 n_classes,
                 n_layers,
                 n_nodes,
                 pooling_opt,
                 use_linear_comb=0,
                 use_intermediate_embedding=1,
                 hidden_activation=torch.relu,
                 out_activation=torch.relu,
                 dropout=0.5,
                 ):
        super(OurNetwork_GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = n_layers
        self.n_hidden = n_hidden
        self.pooling_opt = pooling_opt
        if self.pooling_opt == 2:
            self.attention_layer = nn.Linear(n_hidden, 1)
        self.layers.append(
            OurLayer_GCN(input_dim, n_hidden, n_nodes, activation=hidden_activation, use_linear_comb=use_linear_comb))
        for i in range(n_layers):
            self.layers.append(
                OurLayer_GCN(n_hidden, n_hidden, n_nodes, activation=hidden_activation,
                             use_linear_comb=use_linear_comb))
        # self.layers.append(
        #     OurLayer_GCN(n_hidden, n_classes, n_nodes, activation=out_activation, use_linear_comb=use_linear_comb))

        self.dropout = nn.Dropout(p=dropout)
        self.use_intermediate_embedding = use_intermediate_embedding
        if use_intermediate_embedding==1:
            self.output_layer = nn.Linear(n_hidden, n_classes)
            #self.out_layer = nn.Linear(n_layers + 1, n_classes)

    def forward(self, graph, features):
        if self.use_intermediate_embedding==1:
            intermediate_embeddings = []
            h = features
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(graph, h)  # shape = (# nodes, hidden)
                intermediate_embeddings.append(h)

            stacked_intermediate_embeddings = torch.stack(intermediate_embeddings)
            if self.pooling_opt == 0: ##avg
                pooled_h = torch.mean(stacked_intermediate_embeddings, 0)
            elif self.pooling_opt == 1: ##max
                pooled_h, _ = torch.max(stacked_intermediate_embeddings, 0)
            elif self.pooling_opt == 2:  ##attention
                attention = F.softmax(self.attention_layer(stacked_intermediate_embeddings), 0).repeat(1, 1,self.n_hidden)
                pooled_h = torch.sum(torch.mul(stacked_intermediate_embeddings, attention), 0)
            elif self.pooling_opt == 3: ##sum
                pooled_h = torch.sum(stacked_intermediate_embeddings, 0)
            else:
                print('please check your pooling option. No options now.')
            out = self.output_layer(pooled_h)
        else:
            h = features
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(graph, h)
            out = self.output_layer(h)
        return out


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
    n_layers = 2

    model = OurNetwork(input_dim, n_hidden, num_labels, n_layers, graph.number_of_nodes())
    model(graph, features)
