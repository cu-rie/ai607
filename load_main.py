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
from data.webkb import WebKBset
from dgl.data import load_data, register_data_args
from dgl.data.citation_graph import load_cora, load_citeseer, load_pubmed, load_synthetic
import pandas as pd


def main(args):
    print(args)

    if not os.path.exists(args.result):
        os.mkdir(args.result)

    ## We define experiment name as below :
    exp_name = '%s_%s_%s_%s_%s_%s_%s_%s' % (
        args.exp_name, args.data, args.model_category, args.model, args.use_intermediate_embedding,
        args.use_linear_comb,
        args.n_hidden, args.num_repeats)

    ## Make a directory for saving models.
    saved_exp_dir = os.path.join(args.saved_dir, exp_name)

    for epoch in range(args.n_epochs):
        model = torch.load(os.path.join(saved_exp_dir, str(199) + '_model.pth'))


if __name__ == '__main__':
    datas = ['texas', 'citeseer']
    inters = [0, 1]
    lins = [0, 1]
    hiddens = [64, 128]
    cats = ['ours', 'base']
    models = ['GCN', 'GAT']
    reps = [0, 1, 2, 3, 4]

    data = 'texas'
    inter = 1
    lin = 1
    hidden = 64
    cat = 'ours'
    mod = 'GCN'
    rep = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='1206', type=str)
    parser.add_argument('--result', default='Results_csv', type=str)
    parser.add_argument('--data', default=data,
                        help='cora, citeseer, pubmed, synthetic, cornell, texas, washington, wisconsin',
                        type=str)
    parser.add_argument('--use_intermediate_embedding', default=inter, type=str)
    parser.add_argument('--use_linear_comb', default=lin, type=str)
    parser.add_argument('--n_hidden', default=hidden, type=int)
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--num_heads', default=3, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--disp', default=3, type=int)
    parser.add_argument('--saved_dir', default='saved_model', type=str)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--model_category', default=cat, help='ours, base', type=str)
    parser.add_argument('--model', default=mod, help='GAT, GCN', type=str)
    parser.add_argument('--pooling_opt', default=2,
                        help='0:avg,1:max,2:attention,3:sum',
                        type=int)
    parser.add_argument('--num_repeats', default=rep, type=int)
    args = parser.parse_args()

    main(args)
