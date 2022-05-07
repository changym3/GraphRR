"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import pandas as pd

import torch
import torch as th
from torch import nn
from torch.nn import init
from torch.functional import F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax
from dgl.nn import utils


    

class Attention(nn.Module):
    def __init__(self, dim_hiddens, dim_query, attn_type, num_channels=None):
        super().__init__()
        self.dim_hiddens = dim_hiddens
        self.dim_query = dim_query
        self.attn_type = attn_type
        self.num_channels = num_channels
        self.num_heads = 4
        if attn_type == 'query':
            self.attention = nn.Sequential(nn.Linear(dim_hiddens, dim_query), nn.Tanh(), nn.Linear(dim_query, 1, bias=False))
        elif attn_type == 'trans_query':
            assert num_channels is not None
            self.transformations = nn.ModuleList()
            for i in range(self.num_channels):
                self.transformations.append(nn.Linear(dim_hiddens, dim_query))
            self.attention = nn.Sequential(nn.Tanh(), nn.Linear(dim_query, 1, bias=False))
        elif attn_type == 'query_key':
            out_heads = dim_hiddens // self.num_heads
            self.Qs = nn.ModuleList()
            self.Ks = nn.ModuleList()
            self.Vs = nn.ModuleList()
            for i in range(self.num_heads):
                self.Qs.append(nn.Linear(self.dim_hiddens, out_heads))
                self.Ks.append(nn.Linear(self.dim_hiddens, out_heads))
                self.Vs.append(nn.Linear(self.dim_hiddens, out_heads))
        elif attn_type == 'transform':
            self.transformation = nn.Linear(dim_hiddens * num_channels, dim_hiddens)
    
    def compute_attention(self, x):
        # x \in (C, N, d)
        if self.attn_type == 'query':
            attn = F.softmax(self.attention(x), dim=0)
        elif self.attn_type == 'trans_query':
            assert len(x) == self.num_channels
            x_list = [self.transformations[i](x[i]) for i in range(self.num_channels)]
            x = torch.stack(x_list, dim=0)
            attn = F.softmax(self.attention(x), dim=0)
        return attn
    
    @torch.no_grad()
    def inference_attention(self, x):
        return self.compute_attention(x)
    
    def forward(self, x):
        # x \in (C, N, d)
        if self.attn_type in ['query', 'trans_query']:
            attn = self.compute_attention(x)
            res = (attn * x).sum(dim=0)
        elif self.attn_type == 'mean':
            res = x.mean(dim=0)
        elif self.attn_type == 'transform':
            # (C, N, d) to (N, C, d) to (N, C*d)
            x = x.permute(1, 0, 2).reshape(-1, self.num_channels * self.dim_hiddens)
            res = self.transformation(x)
        return res
    
#         elif self.attn_type == 'query_key':
#             rst_list = []
#             for i in range(self.num_heads):
#                 q_mat = self.Qs[i](x)
#                 k_mat = self.Ks[i](x)
#                 v_mat = self.Vs[i](x)
#                 attn = (q_mat * k_mat).sum(dim=-1, keepdim=True)
#                 rst_list.append((attn * v_mat).sum(dim=0))
#             res = torch.cat(rst_list, dim=-1)




class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 residual=True,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
#         if norm not in ('none', 'both', 'right'):
        if norm not in ('both', 'right', 'none'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self._residual = residual
        
        if self._residual:
            self.loop_weight = nn.Linear(in_feats, out_feats, bias=False)
        
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):

        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if self._residual:
                loop_message = self.loop_weight(feat_dst)
            if graph.num_edges() == 0:
                return feat_dst                
                
            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight
                
            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)
                
            if self._residual:
                rst = rst + loop_message

            return rst

    
    
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
                
            rst = rst.view(len(rst), -1)
                
            return rst      

        

    
            
class Transformer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, residual):
        super().__init__()
        self._in_feats = in_feats
        self._num_heads = num_heads
        self._out_feats = out_feats
        self._out_heads = out_feats // num_heads
        self._residual = residual
        
        self.W_self = nn.Linear(self._in_feats, self._out_feats)
        self.Qs = nn.ModuleList()
        self.Ks = nn.ModuleList()
        self.Vs = nn.ModuleList()
        for i in range(self._num_heads):
            self.Qs.append(nn.Linear(self._in_feats, self._out_heads))
            self.Ks.append(nn.Linear(self._in_feats, self._out_heads))
            self.Vs.append(nn.Linear(self._in_feats, self._out_heads))
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.Qs:
            init.xavier_uniform_(m.weight)
        for m in self.Ks:
            init.xavier_uniform_(m.weight)
        for m in self.Vs:
            init.xavier_uniform_(m.weight)
            
        init.xavier_uniform_(self.W_self.weight)
    
    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            rst_list = []
            for i in range(self._num_heads):
                q_mat = self.Qs[i](feat_dst)
                k_mat = self.Ks[i](feat_src)
                v_mat = self.Vs[i](feat_src)
                graph.srcdata.update({'k': k_mat, 'v': v_mat})
                graph.dstdata.update({'q': q_mat})
                graph.apply_edges(fn.u_mul_v('k', 'q', 'qk'))
                attn_score = graph.edata.pop('qk').sum(dim=1)
                graph.edata['attn'] = edge_softmax(graph, attn_score)
                graph.update_all(fn.u_mul_e('v', 'attn', 'm'),
                                fn.sum('m', 'h'))
    #             attn = (q_mat * k_mat).sum(dim=1, keepdim=True)
    #             v_mat = attn * v_mat
                rst_list.append(graph.dstdata['h'])
            rst = torch.cat(rst_list, dim=-1)
            if self._residual:
                rst = rst + self.W_self(feat_dst)
            return rst

        

        


        
class HeteroGraphConv(nn.Module):
    def __init__(self, args, edge_types):
        super().__init__()
        self.args = args
        if args.rel_conv_type == 'Transformer':
            self.mods = nn.ModuleDict({rel: 
                                       Transformer(args.dim_hiddens, args.dim_hiddens, args.num_heads, args.residual)
                                       for rel in edge_types})
        elif args.rel_conv_type == 'EdgeTransformer':
            self.mods = nn.ModuleDict({rel: 
                                       EdgeTransformer(args.dim_hiddens, args.dim_hiddens, args.dim_edge_feats, args.num_heads, args.residual)
                                       for rel in edge_types})
        elif args.rel_conv_type == 'GCN':
            self.mods = nn.ModuleDict({rel: 
                                       GraphConv(args.dim_hiddens, args.dim_hiddens, allow_zero_in_degree=True, residual=args.residual)
                                       for rel in edge_types})
        elif args.rel_conv_type == 'GAT':
            self.mods = nn.ModuleDict({rel: 
                                       GATConv(args.dim_hiddens, args.dim_hiddens // args.num_heads, num_heads=args.num_heads, allow_zero_in_degree=True, residual=args.residual)
                                       for rel in edge_types})
        elif args.rel_conv_type == 'LGC':
            self.mods = nn.ModuleDict({rel: 
                                       LightConv(args.dim_hiddens, args.dim_hiddens, norm='both')
                                       for rel in edge_types})
            
        self.attn = Attention(args.dim_hiddens, args.dim_query, args.rel_combine, len(edge_types))
        
        self.loop_trans = nn.Linear(args.dim_hiddens, args.dim_hiddens)
        
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)  
        
    def obtain_rel_embs(self, g, x):
        outputs = []
        loop_message = self.loop_trans(x[:g.number_of_dst_nodes()])
        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                dstdata = loop_message
            else:
                dstdata = self.mods[etype](rel_graph, x)
#                 if self.args.residual:
#                     dst_data = dst_data + loop_message               
            outputs.append(dstdata)
        rel_embs = torch.stack(outputs)
        return rel_embs
        
    def forward(self, g, x):
        rel_embs = self.obtain_rel_embs(g, x)
        res = self.attn(rel_embs)
        return res
    
    @torch.no_grad()
    def inference_attention(self, g, x):
        rel_embs = self.obtain_rel_embs(g, x)
        return self.attn.inference_attention(rel_embs)        
        
        
        

         
        
        

    
        
class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        etype = edges.data['id'][0]
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
        att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
        return {'a': att, 'v': val}
    
    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
        
    def forward(self, b, src_h, dst_h):
        if b.num_edges() == 0:
            return dst_h
        with b.local_scope():
            node_dict, edge_dict = b.node_dict, b.edge_dict
            for srctype, etype, dsttype in b.canonical_etypes:
                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]] 
                q_linear = self.q_linears[node_dict[dsttype]]

                b.srcdata['k'] = k_linear(src_h).view(-1, self.n_heads, self.d_k)
                b.srcdata['v'] = v_linear(src_h).view(-1, self.n_heads, self.d_k)
                b.dstdata['q'] = q_linear(dst_h).view(-1, self.n_heads, self.d_k)

                b.apply_edges(func=self.edge_attention, etype=etype)
            b.multi_update_all({etype : (self.message_func, self.reduce_func) \
                                for etype in edge_dict}, cross_reducer = 'mean')

            for ntype in b.dsttypes:
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                trans_out = self.a_linears[n_id](b.dstdata['t'])
                trans_out = trans_out * alpha + dst_h * (1-alpha)
                if self.use_norm:
                    out_h = self.drop(self.norms[n_id](trans_out))
                else:
                    out_h = self.drop(trans_out)
            return out_h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
    
    
    

class HGT(nn.Module):
    def __init__(self, args):
        super(HGT, self).__init__()
        self.args = DotMap(args.toDict())
        args = self.args
        self.feature_trans = nn.Linear(args.dim_features, args.dim_hiddens, bias=False)
        self.convs = nn.ModuleList()
# def preprocess_het_graph(g, args):
#     node_dict = {}
#     edge_dict = {}
#     for ntype in g.ntypes:
#         node_dict[ntype] = len(node_dict)
#     for etype in g.etypes:
#         edge_dict[etype] = len(edge_dict)
#         g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 
#     args.node_dict = node_dict
#     args.edge_dict = edge_dict
        for _ in range(args.conv_depth - 1):
            self.convs.append(HGTLayer(args.dim_hiddens, args.dim_hiddens, 1, args.num_edge_types, args.num_heads, dropout=args.dropout, use_norm=args.use_norm))
        self.convs.append(HGTLayer(args.dim_hiddens, args.dim_embs, 1, args.num_edge_types, args.num_heads, dropout=args.dropout, use_norm=args.use_norm))
        
        if self.args.conv_depth == 1:
            self.sampler_nodes = [5]
            self.sampler_inference = [10]
        elif self.args.conv_depth == 2:
            self.sampler_nodes = [3, 10]
            self.sampler_inference = [5, 10]
        elif self.args.conv_depth == 3:
            self.sampler_nodes = [5, 10, 10]
            self.sampler_inference = [10, 10, 10]
            
        nn.init.xavier_uniform_(self.feature_trans.weight)

    def forward(self, g, nids, device):
        y = torch.zeros(g.num_nodes(), self.args.dim_embs)          
        dataloader = dgl.dataloading.NodeDataLoader(g, nids, 
                                                    dgl.dataloading.MultiLayerNeighborSampler(self.sampler_nodes),
                                                    batch_size=self.args.sampling_batch_size,
                                                    num_workers=self.args.num_workers,
                                                    shuffle=True,
                                                    drop_last=False)
        for input_nodes, output_nodes, blocks in dataloader:
            # initial src hidden
            new_h = torch.tanh(self.feature_trans(blocks[0].srcdata['all'].to(device)))
#             new_h = self.feature_trans(blocks[-1].dstdata['feature'].to(device))
            for i in range(self.args.conv_depth):
                b = blocks[i].to(device)
                b.node_dict = self.args.node_dict
                b.edge_dict = self.args.edge_dict
                src_h = new_h
                dst_h = new_h[:b.num_dst_nodes()]
                new_h = self.convs[i](b, src_h, dst_h)
            y[output_nodes] = new_h.cpu()
        return y
    
    
    def inference(self, g, nids, device):
        y = torch.zeros(g.num_nodes(), self.args.dim_embs)          
        dataloader = dgl.dataloading.NodeDataLoader(g, nids, 
                                                    dgl.dataloading.MultiLayerNeighborSampler(self.sampler_inference),
                                                    batch_size=self.args.inference_batch_size,
                                                    num_workers=self.args.num_workers,
                                                    shuffle=True,
                                                    drop_last=False)
        for input_nodes, output_nodes, blocks in dataloader:
            # initial src hidden
            new_h = torch.tanh(self.feature_trans(blocks[0].srcdata['all'].to(device)))
#             new_h = self.feature_trans(blocks[-1].dstdata['feature'].to(device))
            for i in range(self.args.conv_depth):
                b = blocks[i].to(device)
                b.node_dict = self.args.node_dict
                b.edge_dict = self.args.edge_dict
                src_h = new_h
                dst_h = new_h[:b.num_dst_nodes()]
                new_h = self.convs[i](b, src_h, dst_h)
            y[output_nodes] = new_h.cpu()
        return y
    
    def __repr__(self):
        return '{}(n_inp={}, n_hid={}, n_out={}, n_layers={})'.format(
            self.__class__.__name__, self.n_inp, self.n_hid,
            self.n_out, self.n_layers)
    
        
class NGCFConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        
        self.msg_weight = nn.Linear(in_feats, out_feats)
        self.neighbor_weight = nn.Linear(in_feats, out_feats)        
        self.reset_parameters()
        
    def reset_parameters(self):
        init.xavier_uniform_(self.msg_weight.weight.data)
        init.xavier_uniform_(self.neighbor_weight.weight.data)

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm
            
            graph.srcdata.update({'el': feat_src})
            graph.dstdata.update({'er': feat_dst})

            # v = W1(u⊙v) + W2@u
            graph.apply_edges(fn.u_mul_v('el', 'er', 'e'))
            graph.edata['e_trans'] = self.msg_weight(graph.edata.pop('e'))
            graph.srcdata['u_trans'] = self.neighbor_weight(graph.srcdata['el'])
            graph.apply_edges(fn.u_add_e('u_trans', 'e_trans', 'msg'))
            graph.update_all(fn.copy_e('msg', 'm'),
                             fn.sum(msg='m', out='h'))
#                 graph.update_all(fn.u_add_e('u_trans', 'e_trans', 'm'),
#                             fn.sum('m', 'h'))

            rst = graph.dstdata['h']        
            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm
                
            
            loop_rst = self.neighbor_weight(graph.dstdata['er'])
            rst = rst + loop_rst
            return rst

        
class LightConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        
        self.msg_weight = nn.Linear(in_feats, out_feats)
        self.neighbor_weight = nn.Linear(in_feats, out_feats)        
        self.reset_parameters()
        
    def reset_parameters(self):
        init.xavier_uniform_(self.msg_weight.weight.data)
        init.xavier_uniform_(self.neighbor_weight.weight.data)

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm
            
            graph.srcdata.update({'el': feat_src})
            graph.dstdata.update({'er': feat_dst})

            # v = W1(u⊙v) + W2@u
            graph.update_all(fn.copy_u('el', 'm'),
                             fn.sum(msg='m', out='h'))
#                 graph.update_all(fn.u_add_e('u_trans', 'e_trans', 'm'),
#                             fn.sum('m', 'h'))

            rst = graph.dstdata['h']        
            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm
                
            
#             loop_rst = self.neighbor_weight(graph.dstdata['er'])
#             rst = rst + loop_rst
            return rst