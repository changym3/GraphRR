
import time
import datetime
from collections import defaultdict
from itertools import product
from imp import reload

import dill
from dotmap import DotMap

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 200)
# pd.set_option('display.min_rows', 100)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.neighbors import kneighbors_graph

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)

import dgl
import dgl.nn
import dgl.function as fn
from dgl.nn import RelGraphConv, GATConv, SAGEConv
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax

# from my_nn import GraphConv, HadamardConv, QKV
import utils
from utils.data import MyData, NegDataset, Dataset, MyLoader
from utils.model import ScorePredictor, BatchedMLP, BatchedModel, BatchedHomoModel, ReciprocalModel, BatchGraphLoader
from utils.config import set_args, fix_args
from utils.evaluation import Evaluator, EarlyStopping
from parser import get_parser, get_args

args = get_args()

het_g, het_node_feat_dict, het_edge_feat_dict = utils.data.load_graph(args.graph_hist_range, bidirected=False, simple=False)
# het_g = utils.data.load_reciprocal_graph(args.graph_hist_range, bidirected=True)

label_train = MyData('train', args.label, args.label_train_range)
label_test = MyData('test', args.label, args.label_test_range)
print(label_train.pair_df[args.label].value_counts())
label_ids = torch.cat([label_train.user_ids, label_test.user_ids], dim=-1).unique()

g = het_g.edge_type_subgraph(args.edge_types)


if args.conv_type == 'Hetero':
    args.multi_connection = 'linear' # 'linear', 'none''
    args.rel_conv_type = 'GCN'
    args.rel_combine = 'transform' # 'mean', 'trans_query'
    args.loss_batch_size = 128
#     g = dgl.to_bidirected(dgl.to_simple(g, copy_ndata=True, copy_edata=True), copy_ndata=True)
elif args.conv_type == 'Reciprocal' or 'Ablation_' in args.conv_type:
#         args.multi_connection = 'sum' # 'linear', 'none', 'sum'
#         args.rel_conv_type = 'GAT'
#         args.rel_combine = 'query'    #'mean', 'trans_query', 'query'
    args.loss_batch_size = 64
    args.sampling_batch_size = 64 
elif args.conv_type == 'RGCN':
    args.multi_connection = 'none' # 'linear', 'none'
    args.loss_batch_size = 128
    args.inference_batch_size = 32
    args.sampling_batch_size = 32
elif args.conv_type == 'MLP':
    args.loss_batch_size = 1024000
else:
    args.multi_connection = 'none' # 'linear', 'none'
    args.loss_batch_size = 1024*4
    args.inference_batch_size = 128
    args.sampling_batch_size = 128




device = torch.device(f'cuda:{args.gpu}')

if args.conv_type == 'Reciprocal':
    model = ReciprocalModel(args).to(device)
elif 'Ablation_' in  args.conv_type:
    args.conv_type = args.conv_type[9:]
    model = ReciprocalModel(args).to(device)
else:
    model = BatchedModel(args).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
stopper = EarlyStopping(patience=10, spec=args.conv_type, verbose=args.verbose)

pos_weight = None
res = defaultdict(list)
eval_obj = Evaluator(args, args.eval_metrics)


dataloader = MyLoader(label_train, args.loss_batch_size, args.loss_type)
data_train_eval = label_train.pairs
data_test_eval = label_test.pairs
print(f'The dataloader have {len(dataloader)} iterations.')


start_time = time.time()

for e in range(1000):
    cnt = 0
    for batch in dataloader:
        nids = batch.unique()
        embs = model(g, nids)
        loss = model.compute_loss(embs, batch, args.loss_type)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if cnt % 10 == 0:
            with torch.no_grad():
                embs = model.inference(g, label_ids)
                eval_train_res = eval_obj.predict_and_evaluate(model, embs, data_train_eval, verbose=args.verbose, specs='Training')
                eval_test_res = eval_obj.predict_and_evaluate(model, embs, data_test_eval, verbose=args.verbose, specs='Testing')
                early_stop = stopper.step(loss.detach().item(), eval_test_res, model)
            res['train'].append(eval_train_res)
            res['test'].append(eval_test_res)
            if args.verbose:
                print(f'Epoch {e}, {cnt}/{len(dataloader)}.')
        cnt += 1
        
        if early_stop:
            break
    if early_stop:
        break
    torch.cuda.empty_cache()
end_time = time.time()
print('using time:', end_time-start_time, 'seconds.')
# for u in stopper.best_res:
#     print(f'{u:.4f}')
# return stopper.best_res


metric_names = ['auc', 'gauc', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'mrr@3', 'mrr@5', 'mrr@10', 'hits@3', 'hits@5', 'hits@10', 'f1', 'precision', 'recall', 'logloss']
for u, name in zip(stopper.best_res, metric_names):
    print(f'{name:10}{u:.4f}')