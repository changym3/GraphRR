import math
import itertools
from dotmap import DotMap

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn
import dgl.function as fn
from dgl.nn import RelGraphConv, GATConv, SAGEConv
from .layer import GraphConv, Transformer, Attention, HeteroGraphConv, NGCFConv, LightConv

torch.set_default_dtype(torch.float32)


class ScorePredictor(nn.Module):
    def __init__(self, args, hidden=None):
        super().__init__()
        self.args = args
        if hidden is None:
            self.hidden = self.args.dim_embs
        else:
            self.hidden = hidden
        
        if self.args.predictor == 'dot':
            pass
        elif self.args.predictor == '1-linear':
            self.predictor = nn.Linear(self.hidden * 2, 1)
        elif self.args.predictor == '2-linear':
            self.predictor = nn.Sequential(nn.Linear(self.hidden*2, self.hidden),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden, 1))
        elif self.args.predictor == '3-linear':
            self.predictor = nn.Sequential(nn.Linear(self.hidden*2, self.hidden),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden, self.hidden),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden, 1))
        elif self.args.predictor == '4-linear':
            self.predictor = nn.Sequential(nn.Linear(self.hidden*2, self.hidden),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden, self.hidden),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden, self.hidden),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden, 1))
        self.reset_parameters()
            
    def reset_parameters(self):
        pass
#         if 'linear' in self.predictor_type:
#             for m in self.predictor.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight)
                    
    def forward(self, u_e, u_v):
        if self.args.predictor == 'dot':
            score = u_e.mul(u_v).sum(dim=-1)
        else:
            x = torch.cat([u_e, u_v], dim=-1)
            score = self.predictor(x).flatten()
        return score     
    
    
class MultiFeature(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.feat_dict = args.feat_dict
        self.attr2feat = nn.ModuleDict({k: nn.Linear(v, args.dim_hiddens)
                                        for k,v in self.feat_dict.items()})
        self.input_embs = nn.Embedding(args.num_nodes, args.dim_hiddens)
        if 'mean' in args.multi_feat:
            self.feat_attn = Attention(args.dim_hiddens, args.dim_query, 'mean')
        elif 'attn' in args.multi_feat:
            if 'raw' in args.multi_feat:
                self.feat_attn = Attention(args.dim_hiddens, args.dim_query, args.feat_combine, 4)
            elif 'emb' in args.multi_feat:
                self.feat_attn = Attention(args.dim_hiddens, args.dim_query, args.feat_combine, 5)
        
    def forward(self, ndata_dict):
        multi_feat = self.get_multi_feat(ndata_dict)
        if len(multi_feat) == 1:
            g_feat = multi_feat.squeeze(0)
        else:
            g_feat = self.feat_attn(multi_feat)
        return g_feat
    
    def get_emb_feat(self, ndata_dict):
        if dgl.NID in ndata_dict:
            return [self.input_embs(ndata_dict[dgl.NID])]
        else:
            return [self.input_embs.weight]
    
    def get_ori_feat(self, ndata_dict, feat_names):
        return [self.attr2feat[x](ndata_dict[x]) for x in feat_names]
    
    def get_multi_feat(self, ndata_dict):
        ori_feat_names = ['competitive', 'profile', 'engagement', 'social']
        f = self.args.multi_feat
        if f in ori_feat_names or f == 'all':
            feat_list = self.get_ori_feat(ndata_dict, [f])
        elif f == 'emb':
            feat_list = self.get_emb_feat(ndata_dict)
        elif f in ['mean', 'raw_attn']:
            feat_list = self.get_ori_feat(ndata_dict, ori_feat_names)
        elif f == 'emb_attn':
            feat_list = self.get_ori_feat(ndata_dict, ori_feat_names)
            feat_list.extend(self.get_emb_feat(ndata_dict))
        feat = torch.stack(feat_list, dim=0)
        return feat
    
    @torch.no_grad()
    def inference_attention(self, ndata_dict):
        multi_feat = self.get_multi_feat(ndata_dict)
        return self.feat_attn.inference_attention(multi_feat)
    
    
class BatchedMLP(nn.Module):
    def __init__(self, args, **params):
        super().__init__()
        self.args = DotMap(args.toDict())
        for k,v in params.items():
            self.args[k] = v
        args = self.args
        self.device = torch.device(f'cuda:{args.gpu}')
        
        self.feat_trans = nn.Linear(args.dim_features, args.dim_hiddens)
        self.convs = nn.ModuleList()
        for _ in range(self.args.conv_depth):
            self.convs.append(nn.Linear(args.dim_hiddens, args.dim_hiddens))
                
        if args.act_type == 'relu':
            self.fn_act = nn.ReLU()
        elif args.act_type == 'elu':
            self.fn_act = nn.ELU()
        elif args.act_type == 'sigmoid':
            self.fn_act = nn.Sigmoid()
        elif args.act_type == 'leaky':
            self.fn_act = nn.LeakyReLU()
        elif args.act_type == 'tanh':
            self.fn_act = nn.Tanh()
            
    def forward(self, g, nids):
        x = g.ndata['all']
        if self.args.conv_depth == 0:
            return x
        else:
            y = torch.zeros(self.args.num_nodes, self.args.dim_embs)
            h = x[nids].to(self.device)
            h = self.feat_trans(h)
            for i in range(self.args.conv_depth):
                h = self.fn_act(h)
                h = self.convs[i](h)
            y[nids] = h.cpu()
            return y

    @torch.no_grad()
    def inference(self, g, nids):
        return self.forward(g, nids)

    
class BatchedHomoModel(nn.Module):
    def __init__(self, args, **params):
        super(BatchedHomoModel, self).__init__()
        self.args = DotMap(args.toDict())
        for k,v in params.items():
            self.args[k] = v
        args = self.args
        self.device = torch.device(f'cuda:{args.gpu}')
        
        if args.act_type == 'leaky':
            self.fn_act = F.leaky_relu
        
        self.convs = nn.ModuleList([])
        for _ in range(self.args.conv_depth):
            if args.conv_type == 'GCN':
                conv = GraphConv(args.dim_hiddens, args.dim_hiddens, norm=args.norm, allow_zero_in_degree=True, residual=args.residual)
            elif args.conv_type == 'SAGE':
                conv = SAGEConv(args.dim_hiddens, args.dim_hiddens, 'mean')
            elif args.conv_type == 'GAT':
                assert args.dim_hiddens % args.num_heads == 0 and args.dim_embs % args.num_heads == 0
                conv = dgl.nn.GATConv(args.dim_hiddens, args.dim_hiddens // args.num_heads, num_heads=args.num_heads, feat_drop=args.feat_drop, attn_drop=args.attn_drop, allow_zero_in_degree=True)
            elif args.conv_type == 'RGCN':
                conv = RelGraphConv(args.dim_hiddens, args.dim_hiddens, len(args.edge_types), regularizer='basis', num_bases=4, self_loop=args.residual)
            elif args.conv_type == 'NGCF':
                conv = NGCFConv(args.dim_hiddens, args.dim_hiddens, norm='both')
            elif args.conv_type == 'LGC':
                conv = LightConv(args.dim_hiddens, args.dim_hiddens, norm='both')
                self.args.multi_connection = 'linear'
                self.fn_act = lambda x: x
            elif args.conv_type == 'Transformer':
                conv = Transformer(args.dim_hiddens, args.dim_hiddens, args.num_heads, args.residual)
            elif args.conv_type == 'Hetero' or args.conv_type == 'Reciprocal':
                conv = HeteroGraphConv(args, args.edge_types)
            self.convs.append(conv)
            
        self.concat_weight = nn.Linear((args.conv_depth + 1) * args.dim_hiddens, args.dim_embs, bias=False)
        
        if self.args.conv_depth == 1:
            self.sampler_nodes = [15]
            self.infer_nodes = [30]
        elif self.args.conv_depth == 2:
            self.sampler_nodes = [10, 10]
            self.infer_nodes = [10, 10]
        elif self.args.conv_depth == 3:
            self.sampler_nodes = [10, 5, 5]
        elif self.args.conv_depth == 4:
            self.sampler_nodes = [10, 5, 5, 5]
        
            
        if args.heads_fusion == 'concat':
            self.fuse_fn = lambda x: x.view(len(x), -1)
        elif self.args.heads_fusion == 'mean':
            self.fuse_fn = lambda x: torch.sum(x, dim=1) / self.num_heads
    
        self.multi_feat = MultiFeature(args)
        self.reset_parameters()
        
    def reset_parameters(self):
        pass
#         torch.nn.init.xavier_uniform_(self.self_weight.weight.data)

    def calc_from_blocks(self, blocks, conv_idx):
        if len(conv_idx) == 0:
            return self.multi_feat(blocks[0].dstdata)
        else:
            h = self.multi_feat(blocks[0].srcdata)
            for b, idx in zip(blocks, conv_idx):
                if self.args.conv_type == 'RGCN':
                    h = self.fn_act(h)
                    h = self.convs[idx](b, h, b.edata[dgl.ETYPE])
                elif self.args.conv_type == 'GAT':
                    h = self.fn_act(h)
                    h = self.fuse_fn(self.convs[idx](b, h))
                else:
                    h = self.fn_act(h)
                    h = self.convs[idx](b, h)
            return h

    def calc_from_loader(self, loader):
        if self.args.multi_connection == 'sum':
            output_node_list = []
            output_emb_list = []
            for input_nodes, output_nodes, blocks in loader:
                blocks = [b.to(self.device) for b in blocks]
                if self.args.conv_depth == 1:
                    h0 = self.calc_from_blocks(blocks[-1:], [])
                    h1 = self.calc_from_blocks(blocks, [0])
                    h = h0 + h1
                if self.args.conv_depth == 2:
                    h0 = self.calc_from_blocks(blocks[-1:], [])
                    h1 = self.calc_from_blocks(blocks[-1:], [1])
                    h2 = self.calc_from_blocks(blocks, [0, 1])
                    h = h0 + h1 + h2
                if self.args.conv_depth == 3:
                    h0 = self.calc_from_blocks(blocks[-1:], [])
                    h1 = self.calc_from_blocks(blocks[-1:], [1])
                    h2 = self.calc_from_blocks(blocks[-2:], [1, 2])
                    h3 = self.calc_from_blocks(blocks, [0, 1, 2])
                    h = h0 + h1 + h2 + h3
                output_node_list.append(output_nodes)
                output_emb_list.append(h.cpu())
                torch.cuda.empty_cache()
            output_nodes = torch.cat(output_node_list, dim=0)
            output_embs = torch.cat(output_emb_list, dim=0)
        elif self.args.multi_connection == 'linear':
            output_node_list = []
            output_emb_list = []
            for input_nodes, output_nodes, blocks in loader:
                blocks = [b.to(self.device) for b in blocks]
                if self.args.conv_depth == 1:
                    h0 = self.calc_from_blocks(blocks[-1:], [])
                    h1 = self.calc_from_blocks(blocks, [0])
                    h = torch.cat([h0, h1], dim=-1)
                if self.args.conv_depth == 2:
                    h0 = self.calc_from_blocks(blocks[-1:], [])
                    h1 = self.calc_from_blocks(blocks[-1:], [1])
                    h2 = self.calc_from_blocks(blocks, [0, 1])
                    h = torch.cat([h0, h1, h2], dim=-1)
                if self.args.conv_depth == 3:
                    h0 = self.calc_from_blocks(blocks[-1:], [])
                    h1 = self.calc_from_blocks(blocks[-1:], [1])
                    h2 = self.calc_from_blocks(blocks[-2:], [1, 2])
                    h3 = self.calc_from_blocks(blocks, [0, 1, 2])
                    h = torch.cat([h0, h1, h2, h3], dim=-1)
                h = self.concat_weight(h)
                output_node_list.append(output_nodes)
                output_emb_list.append(h.cpu())
                torch.cuda.empty_cache()
            output_nodes = torch.cat(output_node_list, dim=0)
            output_embs = torch.cat(output_emb_list, dim=0)
        elif self.args.multi_connection == 'none':
            output_node_list = []
            output_emb_list = []
            for input_nodes, output_nodes, blocks in loader:
                blocks = [b.to(self.device) for b in blocks]
                if self.args.conv_depth == 1:
                    h = self.calc_from_blocks(blocks, [0])
                elif self.args.conv_depth == 2:
                    h = self.calc_from_blocks(blocks, [0, 1])
                elif self.args.conv_depth == 3:
                    h = self.calc_from_blocks(blocks, [0, 1, 2])
                output_node_list.append(output_nodes)
                output_emb_list.append(h.cpu())
                torch.cuda.empty_cache()
            output_nodes = torch.cat(output_node_list, dim=0)
            output_embs = torch.cat(output_emb_list, dim=0)
        return output_nodes, output_embs

    def calc_embs(self, loader):
        output_nodes, output_embs = self.calc_from_loader(loader)
        embs = torch.zeros(self.args.num_nodes, self.args.dim_embs)
        embs[output_nodes] = output_embs
        return embs

    def get_dataloader(self, g, nids, sampler_type):
        if sampler_type in ['in', 'out']:
            dataloader = BatchGraphLoader(g, nids, self.sampler_nodes[:self.args.aug_layer], sampler_type, shuffle=False,
                                          batch_size=self.args.sampling_batch_size, topk=self.args.topk)
        elif sampler_type in ['bidirected']:
            dataloader = dgl.dataloading.NodeDataLoader(g, nids,
                                                        dgl.dataloading.MultiLayerNeighborSampler(self.sampler_nodes),
                                                        batch_size=self.args.sampling_batch_size, num_workers=self.args.num_workers,
                                                        shuffle=False, drop_last=False)
        return dataloader
    
    def forward(self, g, nids):
        nids = nids[torch.randperm(len(nids))]
        
        if self.args.conv_type == 'Reciprocal':
            g = dgl.to_simple(g, copy_ndata=True, copy_edata=True)
            dataloader = self.get_dataloader(g, nids, self.args.sampler_type)
        else:
#                 g = dgl.to_simple(g, copy_ndata=True, copy_edata=True)
            if self.args.sampler_type == 'bidirected':
#                 g = dgl.to_bidirected(g, copy_ndata=True)
                g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
            if self.args.conv_type != 'Hetero':
                g = dgl.to_homogeneous(g, ndata=['all'])
            # to_homo must be after to_bidirected
            dataloader = self.get_dataloader(g, nids, self.args.sampler_type)
        embs = self.calc_embs(dataloader)
        return embs
    
    @torch.no_grad()
    def inference(self, g, nids):
        return self.forward(g, nids)
        
    
    
        
class BatchedModel(nn.Module):
    def __init__(self, args):
        super().__init__()       
        self.args = DotMap(args.toDict())
        self.device = torch.device(f'cuda:{args.gpu}')
        
        if self.args.conv_type == 'MLP':
            self.emb_model = BatchedMLP(args)
            if self.args.conv_depth == 0:
                self.link_predictor = ScorePredictor(args, args.dim_features)
            else:
                self.link_predictor = ScorePredictor(args)
        else:
            self.emb_model = BatchedHomoModel(args)
            self.link_predictor = ScorePredictor(args)
        self.device = torch.device(f'cuda:{args.gpu}')
            
    def forward(self, g, nids):
        embs = self.emb_model(g, nids)
        return embs

    def predict_score(self, embs, us, vs):
        us_emb = embs[us].to(self.device)
        vs_emb = embs[vs].to(self.device)
        score = self.link_predictor(us_emb, vs_emb)
        return score
    
    def predict(self, embs, us, vs):
        score = self.predict_score(embs, us, vs)
        return score.sigmoid()
    
    def compute_loss(self, embs, data_batch, loss_type, pos_weight=None):
        if loss_type == 'BCE':
            us, vs, ls = data_batch.T
            scores = self.predict_score(embs, us, vs)
            ls = ls.float().to(self.device)
            loss = F.binary_cross_entropy_with_logits(scores, ls, pos_weight=pos_weight)
        elif loss_type == 'BPR':
            us, ps, ns = data_batch.T
            pos_score = self.predict_score(embs, us, ps)
            neg_score = self.predict_score(embs, us, ns)
            loss = - (pos_score - neg_score).sigmoid().log().mean()
        elif loss_type == 'MY':
            us, vs, ls = data_batch.T
            scores_1 = self.predict_score(embs, us, vs)
            ls_1 = ls.float().to(self.device)
            scores_2 = self.predict_score(embs, us, vs)
            ls_2 = torch.ones_like(scores_1).float().to(self.device)
            scores = torch.cat([scores_1, scores_2], dim=-1)
            ls = torch.cat([ls_1, ls_2], dim=-1)
            loss = F.binary_cross_entropy_with_logits(scores, ls, pos_weight=pos_weight)            
        return loss
    
    @torch.no_grad()
    def inference(self, g, nids):
        embs = self.forward(g, nids)
        return embs
    
    @torch.no_grad()
    def inference_feat_attn(self, g):
        # return C N 1
        return self.emb_model.multi_feat.inference_attention(g.to(self.device).srcdata)
    
    @torch.no_grad()
    def inference_rel_attn(self, g, ):
        # return R N 1
        return self.emb_model.convs[0].inference_attention(g, g.to(self.device).srcdata)
    
    
class ReciprocalModel(nn.Module):
    def __init__(self, args):
        super().__init__()       
        self.args = DotMap(args.toDict())
        self.device = torch.device(f'cuda:{args.gpu}')
        self.emb_models = nn.ModuleDict()
        self.graph_list = ['prfr', 'attr', 'sim']
        self.emb_models['prfr'] = BatchedHomoModel(args, sampler_type='out')
        self.emb_models['attr'] = BatchedHomoModel(args, sampler_type='in')
        self.emb_models['sim'] = BatchedHomoModel(args, sampler_type='bidirected')
#         for k in args.rec_edge_types:
#             self.emb_models[k] = BatchedHomoModel(args, edge_types=args.rec_edge_types[k], sampler_type='')
        
        if self.args.recp_combine in ['mean', 'query', 'trans_query', 'transform']:
            self.attention =  Attention(args.dim_hiddens, args.dim_query, self.args.recp_combine, len(self.graph_list))
        self.link_predictor = ScorePredictor(args)

    def forward(self, g, nids):
        embs_list = [self.emb_models[k](g, nids)[nids].to(self.device) for k in self.graph_list]
        embs = torch.zeros(self.args.num_nodes, self.args.dim_embs, device=self.device)
        embs[nids] = self.recp_combine(embs_list)
        return embs

    def recp_combine(self, embs_list):
        if self.args.recp_combine in ['mean', 'query', 'trans_query', 'transform']:
            embs = torch.stack(embs_list, dim=0)
            out_embs = self.attention(embs)
        elif self.args.recp_combine == 'prfr':
            out_embs = embs_list[0]
        elif self.args.recp_combine == 'attr':
            out_embs = embs_list[1]
        elif self.args.recp_combine == 'sim':
            out_embs = embs_list[2]
        return out_embs
    
    def predict_score(self, embs, us, vs):
        us_emb = embs[us].to(self.device)
        vs_emb = embs[vs].to(self.device)
        score = self.link_predictor(us_emb, vs_emb)
        return score
    
    def predict(self, embs, us, vs):
        score = self.predict_score(embs, us, vs)
        return score.sigmoid()
    
    def compute_loss(self, embs, data_batch, loss_type, pos_weight=None):
        if loss_type == 'BCE':
            us, vs, ls = data_batch.T
            scores = self.predict_score(embs, us, vs)
            ls = ls.float().to(self.device)
            loss = F.binary_cross_entropy_with_logits(scores, ls, pos_weight=pos_weight)
        elif loss_type == 'BPR':
            us, ps, ns = data_batch.T
            pos_score = self.predict_score(embs, us, ps)
            neg_score = self.predict_score(embs, us, ns)
            loss = - (pos_score - neg_score).sigmoid().log().mean()
        return loss
            
    @torch.no_grad()
    def inference(self, g, nids):
        embs = self.forward(g, nids)
        return embs
    
    @torch.no_grad()
    def inference_feat_attn(self, g):
        # return C N 1
        return self.emb_model.multi_feat.inference_attention(g.to(self.device).srcdata)
    
    @torch.no_grad()
    def inference_rel_attn(self, g, ):
        # return R N 1
        return self.emb_model.convs[0].inference_attention(g, g.to(self.device).srcdata)
    

class BatchGraphLoader():
    def __init__(self, g, nids, fanouts, direction, batch_size, shuffle, topk):
        self.g = g
        self.etypes = self.g.canonical_etypes
        self.nids = nids
        self.fanouts = fanouts
        self.direction = direction
        self.num_layers = len(fanouts)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.topk = topk
        
    def gen_direct_graph(self, batch):
        if self.direction == 'in':
            g = dgl.sampling.sample_neighbors(self.g, batch, self.fanouts[0], edge_dir='in')
        elif self.direction == 'out':
            g = dgl.sampling.sample_neighbors(self.g, batch, self.fanouts[0], edge_dir='out')
            g = dgl.reverse(g, copy_edata=True)
        return g
    
    def gen_complete_graph(self, g):
        data_dict = {}
        for ut, et, ut in self.etypes:
            pairs_list = []
            srcs, dsts = g.edges(etype=et)
            for u in dsts.unique():
                seeds = srcs[u == dsts]
                pairs = torch.cartesian_prod(seeds, seeds)
                pairs_list.append(pairs)
            if len(pairs_list) > 0:
                et_pairs = torch.cat(pairs_list, dim=0)
                data_dict[ut, et, ut] = tuple(et_pairs.T)
            else:
                data_dict[ut, et, ut] = ([], [])
        c_g = dgl.heterograph(data_dict, num_nodes_dict={ut: self.g.number_of_nodes()})
#         c_g.ndata['all'] = self.g.ndata['all']
        return c_g
    
    def gen_topk_graph(self, g, topk):
        c_g = self.gen_complete_graph(g)
        c_g.ndata['all'] = g.ndata['all']
        for et in self.etypes:
            c_g.apply_edges(fn.u_sub_v('all', 'all', 'diff'), etype=et)
            c_g.edges[et].data['distance'] = c_g.edges[et].data['diff'].square().sum(dim=-1)
        topk_g = dgl.sampling.select_topk(c_g, topk, 'distance',  ascending=True)
        return topk_g

    def gen_blocks(self, batch):
        g = self.gen_direct_graph(batch)
#         seed_nodes = {et: g.edges(etype=et)[0] for _, et, _ in self.etypes}
        topk_g = self.gen_topk_graph(g, self.topk)
        
        b = dgl.to_block(g, batch)
        blocks = [b]
        for idx in range(1, self.num_layers):
            seed_nodes = b.srcdata[dgl.NID]
            g = dgl.sampling.sample_neighbors(topk_g, seed_nodes, self.fanouts[idx], edge_dir='in')
            b = dgl.to_block(g, seed_nodes)
            b.srcdata['all'] = self.g.ndata['all'][b.srcdata[dgl.NID]]
            b.dstdata['all'] = self.g.ndata['all'][b.dstdata[dgl.NID]]
            blocks.append(b)
        return blocks[::-1]
#             for ut, et, ut in self.etypes:
#                 b.edges[et].data['ego_feat'] = self.g.ndata['all'][u].unsqueeze(0).repeat(b.num_edges(et), 1)
    
    def generate_batch(self):
        if self.shuffle:
            nids = self.nids[torch.randperm(len(self.nids))]
        else:
            nids = self.nids
        start = 0
        for start in range(0, len(nids), self.batch_size):
            batch_user = nids[start: start+self.batch_size]
            blocks = self.gen_blocks(batch_user)
            input_nodes = blocks[0].srcdata[dgl.NID]
            output_nodes = blocks[-1].dstdata[dgl.NID]
#             print(1)
            yield input_nodes, output_nodes, blocks
            
    def __iter__(self):
        return self.generate_batch()
    