import torch
# from .data import PairLoader
from .config import args

import numpy as np
from sklearn import metrics
from collections import defaultdict
import datetime


class Evaluator(object):
    def __init__(self, args, metrics=None):
        self.args = args
        if metrics:
            self.args.eval_metrics = metrics
        self.device = torch.device(f'cuda:{args.gpu}')
    
    def predict_and_evaluate(self, model, embs, pairs, verbose=False, specs='Results'):
        y_truth, y_score = self.predict(model, embs, pairs)
        eval_res = self.evaluate(y_truth, y_score, pairs, verbose, specs)
        return eval_res
    
    def evaluate(self, y_truth, y_score, pairs, verbose=False, specs='Results'):
        res = self.classification_metrics(y_truth, y_score, uids=pairs[..., 0])
        if verbose:
            print(f'{specs}: {self.to_string(res)}.')
        return res
    
    @torch.no_grad()
    def predict(self, model, embs, pairs):
        batch_size = self.args.eval_batch_size
        ys_list = []
        yt_list = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start: start+batch_size]
            us, vs, ls = batch[:, 0], batch[:, 1], batch[:, 2]
            y_score = model.predict_score(embs, us, vs)
            ys_list.append(y_score)
            yt_list.append(ls)
        y_score = torch.cat(ys_list, dim=-1)
        y_truth = torch.cat(yt_list, dim=-1)
        return y_truth, y_score.cpu()
    
    def obtain_user_groups(self, y_truth, y_prob, uids):
        y_truth, y_prob, uids = y_truth.numpy(), y_prob.numpy(), uids.numpy()
        group_prob = defaultdict(list)
        group_truth = defaultdict(list)
        for lbl, prb, usr in zip(y_truth, y_prob, uids):
            group_prob[usr].append(prb)
            group_truth[usr].append(lbl)
#         group_flag = defaultdict(lambda: False)
#         for user_id in set(uids):
#             truths = group_truth[user_id]
#             flag = False
#             for i in range(len(truths) - 1):
#                 if truths[i] != truths[i + 1]:
#                     flag = True
#                     break
#             group_flag[user_id] = flag
        return group_truth, group_prob
    
    
    def classification_metrics(self, y_truth, y_score, uids):
        y_prob = y_score.sigmoid()
        fpr, tpr, thresholds = metrics.roc_curve(y_truth, y_prob)
        thrsh = thresholds[np.argmax(tpr - fpr)]
        y_pred = (y_prob > thrsh)
        
        group_truth, group_prob = self.obtain_user_groups(y_truth, y_score, uids)
        self.group_truth = group_truth
        self.group_prob = group_prob
        
        res = []
        names = []
        for mtr in self.args.eval_metrics:
            if mtr == 'auc':
                res.append(metrics.roc_auc_score(y_truth, y_prob))
                names.append('auc')
            elif mtr == 'f1':
                res.append(metrics.f1_score(y_truth, y_pred))
                names.append('f1')
            elif mtr == 'precision':
                res.append(metrics.precision_score(y_truth, y_pred))
                names.append('precision')
            elif mtr == 'recall':
                res.append(metrics.recall_score(y_truth, y_pred))
                names.append('recall')
            elif mtr == 'neg_loss':
                res.append(metrics.log_loss(y_truth, y_prob))
                names.append('logloss')
            elif mtr == 'gauc':
#                 user2auc, user2impr, group_auc = self.gauc_score(y_truth, y_prob, uids)
                gauc = self.gauc(group_truth, group_prob)
                res.append(gauc)
                names.append('gauc')
            elif mtr == 'ndcg':
                ndcg = self.ndcg(group_truth, group_prob, 3)
                res.append(ndcg)
                names.append('ndcg@3')
                ndcg = self.ndcg(group_truth, group_prob, 5)
                res.append(ndcg)
                names.append('ndcg@5')
                ndcg = self.ndcg(group_truth, group_prob, 10)
                res.append(ndcg)
                names.append('ndcg@10')
            elif mtr == 'mrr':
                s = self.mrr(group_truth, group_prob, 3)
                res.append(s)
                names.append('mrr@3')
                s = self.mrr(group_truth, group_prob, 5)
                res.append(s)
                names.append('mrr@5')
                s = self.mrr(group_truth, group_prob, 10)
                res.append(s)
                names.append('mrr@10')
            elif mtr == 'hits':
                s = self.hits(group_truth, group_prob, 3)
                res.append(s)
                names.append('hits@3')
                s = self.hits(group_truth, group_prob, 5)
                res.append(s)
                names.append('hits@5')
                s = self.hits(group_truth, group_prob, 10)
                res.append(s)
                names.append('hits@10')
            else:
                raise Exception(f'{mtr} is unknown metrics.')
        self.eval_metrics = names
        return np.array(res)
    
    def hit_score(self, y_truth, y_prob, k):
        topk_idx = y_prob.argsort()[-k:][::-1]
        topk_lbl = y_truth[topk_idx]
        n_hits = topk_lbl.sum()
        hit_rate = n_hits / len(y_prob)
        return hit_rate
        
    def hits(self, group_truth, group_prob, k):
        hits = 0
        for u in group_truth:
            hits += self.hit_score(np.asarray(group_truth[u]), np.asarray(group_prob[u]), k)
        return hits / len(group_truth)
    
    def mrr_score(self, y_truth, y_prob, k):
        topk_idx = y_prob.argsort()[-k:][::-1]
        topk_lbl = y_truth[topk_idx]
        first_pos_idx = topk_lbl.argmax(axis=-1)
        return 1 / (first_pos_idx+1)
        
        
    def mrr(self, group_truth, group_prob, k):
        mrrs = 0
        for u in group_truth:
            mrrs += self.mrr_score(np.asarray(group_truth[u]), np.asarray(group_prob[u]), k)
        return mrrs / len(group_truth)
    
    
    def ndcg(self, group_truth, group_prob, k):
#         max_len = max(len(user_score[k]) for k in uids)
#         scores = np.asarray([np.pad(user_score[k], (0, max_len - len(user_score[k])), 'constant', constant_values=0) for k in user_score])
        ndcgs = 0
        for u in group_truth:
            ndcgs += metrics.ndcg_score([group_truth[u]], [group_prob[u]], k=k)
        return ndcgs / len(group_truth)
    
    def gauc(self, group_truth, group_prob):
        aucs = 0
        for u in group_truth:
            aucs += metrics.roc_auc_score(group_truth[u], group_prob[u])
        return aucs / len(group_truth)
        
#     def gauc_score(self, y_truth, y_prob, uids):
#         y_truth, y_prob, uids = y_truth.numpy(), y_prob.numpy(), uids.numpy()
#         group_prob = defaultdict(lambda: [])
#         group_truth = defaultdict(lambda: [])
#         for lbl, prb, usr in zip(y_truth, y_prob, uids):
#             group_prob[usr].append(prb)
#             group_truth[usr].append(lbl)
    
#         group_flag = defaultdict(lambda: False)
#         for user_id in set(uids):
#             truths = group_truth[user_id]
#             flag = False
#             for i in range(len(truths) - 1):
#                 if truths[i] != truths[i + 1]:
#                     flag = True
#                     break
#             group_flag[user_id] = flag
# #         return group_flag, group_prob, group_truth
#         impression_total = 0
#         total_auc = 0
#         user2auc = {}
#         user2impr = {}
#         for usr in group_flag:
#             if group_flag[usr]:
#                 auc = metrics.roc_auc_score(np.asarray(group_truth[usr]), np.asarray(group_prob[usr]))
#                 user2auc[usr] = auc
#                 user2impr[usr] = len(group_truth[usr])
#                 total_auc += auc * len(group_truth[usr])
#                 impression_total += len(group_truth[usr])
#         group_auc = total_auc / impression_total
#         return user2auc, user2impr, group_auc
    
    def to_string(self, res, sep=', '):
        s = ''
        for mtr,v in zip(self.eval_metrics, res):
            s += f'{mtr}={v:.4f}{sep}'
        return s

    
class EarlyStopping(object):
    def __init__(self, patience=10, spec='', verbose=True):
        dt = datetime.datetime.now()
        self.filename = './early_stop/{}_{}_{:02d}-{:02d}-{:02d}-{:06d}.pth'.format(
            spec, dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)
        self.patience = patience
        self.counter = 0
        self.best_res = None
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def step(self, loss, eval_res, model):
        if self.best_loss is None:
            self.best_res = eval_res
            self.best_loss = loss
#             self.save_checkpoint(model)
#         elif (loss > self.best_loss) and (acc < self.best_acc):
        elif (eval_res[1] < self.best_res[1]):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
#             if (loss <= self.best_loss) and (acc >= self.best_acc):
#             if (acc >= self.best_acc):
#                 self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_res = eval_res
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))