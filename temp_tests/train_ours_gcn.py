import torch
import argparse, time
import numpy as np

from models.OurNetwork_GCN import OurNetwork_GCN
from utils.train_utils import evaluate
from dgl.data import load_data, register_data_args
from dgl.data.citation_graph import load_cora, load_citeseer, load_pubmed, load_synthetic

if __name__ == "__main__":
    import wandb
    import dgl

    # exp_name = 'DEBUG'
    exp_name = 'Ours_citeseer_intermediate_embedding'
    data = load_citeseer()
    use_intermediate_embedding = True
    use_linear_comb = False

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
    n_layers = 3

    model = OurNetwork_GCN(input_dim, n_hidden, num_labels, n_layers, n_nodes=graph.number_of_nodes(),
                           use_intermediate_embedding=use_intermediate_embedding, use_linear_comb=use_linear_comb)

    # train arguments

    lr = 1e-3
    n_epochs = 200

    loss_ftn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dur = []

    wandb.init(project='gnnproject_ai607', name=exp_name)
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
