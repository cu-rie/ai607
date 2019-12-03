import torch
import argparse, time
import numpy as np
import os
import wandb
import dgl
import networkx as nx
from models.OurNetwork_GAT import OurNetwork_GAT
from models.OurNetwork_GCN import OurNetwork_GCN
from models.GAT import GAT
from models.GCN import GCN
from utils.train_utils import evaluate
from dgl.data import load_data, register_data_args
from dgl.data.citation_graph import load_cora, load_citeseer, load_pubmed, load_synthetic

def main(args):
    print(args)

    ## We define experiment name as below :
    exp_name = '%s_%s_%s_%s' % (args.exp_name, args.model_category, args.model, args.data)

    ## Make a directory for saving models.
    if not os.path.exists(args.saved_dir):
        os.mkdir(args.saved_dir)
    saved_exp_dir = os.path.join(args.saved_dir, exp_name)
    if not os.path.exists(saved_exp_dir):
        os.mkdir(saved_exp_dir)

    ## Load dataset from args.data
    if args.data == 'citeseer':
        data = load_citeseer()
    elif args.data == 'cora':
        data = load_cora()
    elif args.data == 'pubmed':
        data = load_pubmed()
    elif args.data == 'synthetic':
        data = load_synthetic()
    elif args.data in ['cornell', "texas", "washington", "wisconsin"]:
        data = WebKBset(args.data)
    else:
        print('Please specify the name of dataset.')


    ## Load graph structure based on models. Especially different at GAT only.
    if args.model != 'GAT':
        graph = dgl.DGLGraph(data.graph)
    else:
        g = data.graph
        g.remove_edges_from(nx.selfloop_edges(g))
        graph = dgl.DGLGraph(g)
        graph.add_edges(graph.nodes(), graph.nodes())

    ## Prepare dataset
    features = torch.Tensor(data.features)
    train_mask = torch.LongTensor(data.train_mask).bool()
    test_mask = torch.LongTensor(data.test_mask).bool()
    val_mask = torch.LongTensor(data.val_mask).bool()
    labels = torch.LongTensor(data.labels)
    num_labels = data.num_labels
    input_dim = features.shape[1]

    ## Specify Models
    if args.model_category == 'ours':
        if args.model == 'GAT':
            model = OurNetwork_GAT(input_dim, args.n_hidden, num_labels, args.n_layers, pooling_opt=args.pooling_opt, n_nodes=graph.number_of_nodes(),
                                   num_heads=args.num_heads, use_intermediate_embedding=True, use_linear_comb=False)
        elif args.model == 'GCN':
            model = OurNetwork_GCN(input_dim, args.n_hidden, num_labels, args.n_layers, pooling_opt=args.pooling_opt, n_nodes=graph.number_of_nodes(),
                                   use_intermediate_embedding=True, use_linear_comb=False)
        else:
            print('We dont have any other model:(')
    else:
        if args.model == 'GAT':
            model = GAT(input_dim=input_dim, num_hidden=args.n_hidden, num_classes=num_labels, num_layers=args.n_layers)
        elif args.model == 'GCN':
            model = GCN(input_dim=input_dim, num_hidden=args.n_hidden, num_classes=num_labels, num_layers=args.n_layers)
        else:
            print('We dont have any other model:(')

    ## Training Settings
    loss_ftn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dur = []
    wandb.init(project='gnnproject_ai607', name=exp_name)
    wandb.watch(model)

    print('Training Starts !')
    bad_cnt, max_dev = 0, 0
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= args.disp:
            t0 = time.time()

        logits = model(graph, features)
        loss = loss_ftn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.disp:
            dur.append(time.time() - t0)

        val_acc = evaluate(model, graph, features, labels, val_mask)
        wandb.log({'epoch': epoch, 'time': np.mean(dur), 'loss': loss.item(), 'acc': val_acc})
        print('Epoch : %s | Time : %.4f | Loss : %.4f | Valid Accuracy : %.4f' % (
        epoch, np.mean(dur), loss.item(), val_acc))

        ## Model save at each epoch
        saved_model_dir = os.path.join(saved_exp_dir, str(epoch) + '_model.pth')
        torch.save(model, saved_model_dir)

        if val_acc > max_dev:
            max_dev = val_acc
            max_epoch = epoch
            bad_cnt = 0
        else:
            bad_cnt += 1
        if bad_cnt > args.patience:
            print('Early Stopped!')
            break

    ## load the best model and test.
    model_loaded = torch.load(os.path.join(saved_exp_dir, str(max_epoch) + '_model.pth'))
    test_acc = evaluate(model_loaded, graph, features, labels, test_mask)
    wandb.log({'test_acc': test_acc})
    print('Final Test Accuracy : %.4f' % test_acc)


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='1114', type=str)
    parser.add_argument('--data', default='citeseer', help='cora, citeseer, pubmed, synthetic, cornell, texas, washington, wisconsin', type=str)
    parser.add_argument('--use_intermediate_embedding', default=True, type=str)
    parser.add_argument('--use_linear_comb', default=False, type=str)
    parser.add_argument('--n_hidden', default=64, type=int)
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--num_heads', default=3, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--disp', default=3, type=int)
    parser.add_argument('--saved_dir', default='saved_model', type=str)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--model_category', default='ours', help='ours, base', type=str)
    parser.add_argument('--model', default='GCN', help='GAT, GCN', type=str)
    parser.add_argument('--pooling_opt', default=2, help='0:avg,1:max,2:attention,3:sum', type=int)
    args = parser.parse_args()

    main(args)
