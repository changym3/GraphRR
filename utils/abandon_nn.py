class SuperGAT(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, attn_type, residual):
        super().__init__()
        self._in_feats = in_feats
        self._num_heads = num_heads
        self._out_feats = out_feats
        self._out_heads = out_feats // num_heads
        self.attn_type = attn_type
        self._residual = residual
        self.negative_slope = 0.2
        self.scale_factor = math.sqrt(self._out_heads)
        
        self.W_self = nn.Linear(self._in_feats, self._out_feats)
        self.Ws = nn.ModuleList()
        self.As_l = nn.ModuleList()
        self.As_r = nn.ModuleList()
        for i in range(self._num_heads):
            self.Ws.append(nn.Linear(self._in_feats, self._out_heads))
            self.As_l.append(nn.Linear(self._out_heads, 1, bias=False))
            self.As_r.append(nn.Linear(self._out_heads, 1, bias=False))
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.Ws:
            init.xavier_uniform_(m.weight)
        for m in self.As_l:
            init.xavier_uniform_(m.weight)     
        for m in self.As_r:
            init.xavier_uniform_(m.weight)     
        init.xavier_uniform_(self.W_self.weight)
    
    def forward(self, graph, feat):
        self.attn_scores = []
        self.attn_labels = graph.edata['attn_labels']
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            rst_list = []
            for i in range(self._num_heads):
                feat_src_head = self.Ws[i](feat_src)
                feat_dst_head= self.Ws[i](feat_dst)
                graph.srcdata.update({'feat': feat_src_head})
                graph.dstdata.update({'feat': feat_dst_head})
                attn_score = self.compute_attention_score(graph, feat_src_head, feat_dst_head, i)
                self.attn_scores.append(attn_score)
                graph.edata['attn']  = edge_softmax(graph, F.leaky_relu(attn_score, self.negative_slope))
                graph.edata['attn_pos'] = graph.edata['attn'] * graph.edata['attn_labels']
                graph.update_all(fn.u_mul_e('feat', 'attn_pos', 'm'),
                                fn.sum('m', 'h'))
                rst_list.append(graph.dstdata['h'])
            rst = torch.cat(rst_list, dim=-1)
            if self._residual:
                rst = rst + self.W_self(feat_dst)
            return rst
        
    def compute_attention_score(self, graph, feat_src, feat_dst, head_idx):
        if self.attn_type == 'SD':
            graph.apply_edges(fn.u_mul_v('feat', 'feat', 'dp'))
            attn_score = graph.edata['dp'].sum(dim=-1) / self.scale_factor
        elif self.attn_type == 'MX':
            # dot-product
            graph.apply_edges(fn.u_mul_v('feat', 'feat', 'dp'))
            dp_score = graph.edata['dp'].sum(dim=-1)
            # cat-score
            graph.srcdata.update({'attn_l': self.As_l[head_idx](feat_src).sum(dim=-1)})
            graph.dstdata.update({'attn_r': self.As_r[head_idx](feat_dst).sum(dim=-1)})
            graph.apply_edges(fn.u_add_v('attn_l', 'attn_r', 'cat'))
            cat_score =  graph.edata['cat'].sum(dim=-1) 
            attn_score = dp_score.sigmoid() * cat_score
        return attn_score
    
    def attention_loss(self):
        loss = 0
#         print(self.attn_scores)
#         print(self.attn_labels)
        for i in range(self._num_heads):
            loss = loss + F.binary_cross_entropy_with_logits(self.attn_scores[i], self.attn_labels)
#             print(loss)
        loss = loss / self._num_heads
        return loss
        
        
class EdgeTransformer(nn.Module):
    def __init__(self, in_feats, out_feats, e_feats, num_heads, residual):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._e_feats = e_feats
        self._num_heads = num_heads
        self._out_heads = out_feats // num_heads
        self._residual = residual
        
        self.W_self = nn.Linear(self._in_feats, self._out_feats)
        self.Qs = nn.ModuleList()
        self.Ks = nn.ModuleList()
        self.Vs = nn.ModuleList()
        self.Es = nn.ModuleList()
        self.As = nn.ModuleList()
        for i in range(self._num_heads):
            self.Qs.append(nn.Linear(self._in_feats, self._out_heads))
            self.Ks.append(nn.Linear(self._in_feats, self._out_heads))
            self.Vs.append(nn.Linear(self._in_feats, self._out_heads))
            self.Es.append(nn.Linear(self._e_feats, self._out_heads))
            self.As.append(nn.Linear(self._out_heads, 1))
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.Qs:
            init.xavier_uniform_(m.weight)
        for m in self.Ks:
            init.xavier_uniform_(m.weight)
        for m in self.Vs:
            init.xavier_uniform_(m.weight)
        for m in self.Es:
            init.xavier_uniform_(m.weight)
        for m in self.As:
            init.xavier_uniform_(m.weight)    
        init.xavier_uniform_(self.W_self.weight)
    
    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            feat_edge = graph.edata['feature']
            rst_list = []
            for i in range(self._num_heads):
                q_mat = self.Qs[i](feat_dst)
                k_mat = self.Ks[i](feat_src)
                v_mat = self.Vs[i](feat_src)
                e_mat = self.Es[i](feat_edge)
                graph.srcdata.update({'k': k_mat, 'v': v_mat})
                graph.dstdata.update({'q': q_mat})
                graph.edata.update({'e': e_mat})
#                 e_mat.mm(torch.concat([q_mat, k_mat, v_mat], dim=-1))

#                 # Q.K.E
#                 graph.apply_edges(fn.u_mul_e('k', 'e', 'ke'))
#                 graph.apply_edges(fn.e_mul_v('ke', 'q', 'qke'))
#                 attn_score = graph.edata['qke'].sum(dim=1)
                # A(Q.K.E)
                graph.apply_edges(fn.u_mul_e('k', 'e', 'ke'))
                graph.apply_edges(fn.e_mul_v('ke', 'q', 'qke'))
                attn_score = self.As[i](graph.edata['qke'])
    
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
        
        
class RTE_Transformer(nn.Module):
    def __init__(self, args, in_feats, out_feats, e_feats, num_heads, residual):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._e_feats = e_feats
        self._num_heads = num_heads
        self._out_heads = out_feats // num_heads
        self._residual = residual
        
        self.W_self = nn.Linear(self._in_feats, self._out_feats)
        self.Qs = nn.ModuleList()
        self.Ks = nn.ModuleList()
        self.Vs = nn.ModuleList()
        self.Es = nn.ModuleList()
        self.As = nn.ModuleList()
        self.RTE = RelTemporalEncoding(args, self._out_feats)
        for i in range(self._num_heads):
            self.Qs.append(nn.Linear(self._in_feats, self._out_heads))
            self.Ks.append(nn.Linear(self._in_feats, self._out_heads))
            self.Vs.append(nn.Linear(self._in_feats, self._out_heads))
            self.Es.append(nn.Linear(self._e_feats, self._out_heads))
            self.As.append(nn.Linear(self._out_heads * 2, 1))
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.Qs:
            init.xavier_uniform_(m.weight)
        for m in self.Ks:
            init.xavier_uniform_(m.weight)
        for m in self.Vs:
            init.xavier_uniform_(m.weight)
        for m in self.Es:
            init.xavier_uniform_(m.weight)
        for m in self.As:
            init.xavier_uniform_(m.weight)    
        init.xavier_uniform_(self.W_self.weight)
    
    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.edata.update({'ts_emb': self.RTE(graph.edata['ts'])})
            graph.srcdata.update({'feat': feat_src})
            graph.apply_edges(fn.u_add_e('feat', 'ts_emb', 'tef')) # temperal enconded feat

            rst_list = []
            for i in range(self._num_heads):
                q_mat = self.Qs[i](feat_dst)
                graph.dstdata.update({'q': q_mat})
                
                v_mat = self.Vs[i](graph.edata['tef'])
                graph.edata['k_mat'] = self.Ks[i](graph.edata['tef'])
                graph.apply_edges(lambda e: {'q_mat': e.dst['q']})
                attn_score = F.leaky_relu(self.As[i](
                    torch.cat([graph.edata['k_mat'], graph.edata['q_mat']], dim=-1)))
                attn = edge_softmax(graph, attn_score)
                graph.edata['m'] = attn * v_mat
                graph.update_all(fn.copy_e('m', 'm'),
                                 fn.sum('m', 'h'))
                
#                 # A(Q.K.V)
#                 graph.apply_edges(fn.u_mul_e('k', 'e', 'ke'))
#                 graph.apply_edges(fn.e_mul_v('ke', 'q', 'qke'))
#                 attn_score = F.relu(self.As[i](graph.edata['qke']))
                
#                 graph.edata['attn'] = edge_softmax(graph, attn_score)
# #                 graph.update_all(fn.u_mul_e('v', 'attn', 'm'),
# #                                 fn.sum('m', 'h'))

#                 graph.apply_edges(fn.u_add_e('v', 'e', 'v_rte'))
#                 graph.edata['v_res']= graph.edata['attn'] * graph.edata['v_rte']
#                 graph.update_all(fn.copy_e('v_res', 'm'),
#                                  fn.sum('m', 'h'))
                

                rst_list.append(graph.dstdata['h'])
            rst = torch.cat(rst_list, dim=-1)
            if self._residual:
                rst = rst + self.W_self(feat_dst)
            return rst
        
        
class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, args, n_hid):
        super().__init__()
        self.args = args
        self.seq_start = self.ts2seq(pd.Timestamp(args.period_start))
        self.seq_end = self.ts2seq(pd.Timestamp(args.period_end))
        max_len = self.seq_end - self.seq_start
        
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
        
    def ts2seq(self, ts):
        if self.args.ts_freq == 'hour':
            return ts.day_of_year * 24 + ts.hour
            
    def forward(self, t):
#         assert (t >= self.seq_start).sum() > 0
        return self.lin(self.emb(t-self.seq_start))           



class OldHeteroGraphConv(nn.Module):
    def __init__(self, mods, dim_values, dim_query, agg_type='attn'):
        super().__init__()
        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
                
        self.dim_values = dim_values
        self.dim_query = dim_query
        
        self.attention = nn.ModuleDict()
        for k, _ in self.mods.items():
            self.attention[k] = nn.Sequential(nn.Linear(dim_values, dim_query), nn.Tanh(), nn.Linear(dim_query, 1, bias=False))
        
        self.agg_type = agg_type
        if agg_type == 'sum':
            self.agg_fn = th.sum
        elif agg_type == 'max':
            self.agg_fn = lambda inputs, dim: th.max(inputs, dim=dim)[0]
        elif agg_type == 'min':
            self.agg_fn = lambda inputs, dim: th.min(inputs, dim=dim)[0]
        elif agg_type == 'stack':
            self.agg_fn = th.stack
        
    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = []
        et_scores = []
        et_count = 0
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = inputs[:g.number_of_dst_nodes()]

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    et_scores.append(torch.zeros(g.number_of_dst_nodes(), 1).to(g.device))
                    outputs.append(torch.zeros(g.number_of_dst_nodes(), self.dim_values).to(g.device))
                    continue
                et_count += 1
                dstdata = self.mods[etype](rel_graph, (src_inputs, dst_inputs))
                outputs.append(dstdata)
                et_scores.append(self.attention[etype](dstdata))

        if len(outputs) == 0:
            out_embs = dst_inputs
        else:
            et_dst_data = torch.stack(outputs, dim=0)
            if self.agg_type == 'attn':
                attn = torch.softmax(torch.stack(et_scores, dim=0), dim=0)
                out_embs = (attn * et_dst_data).sum(dim=0)
            elif self.agg_type == 'attn_sum':
                attn = torch.softmax(torch.stack(et_scores, dim=0).mean(dim=1, keepdims=True), dim=0)
                out_embs = (attn * et_dst_data).sum(dim=0)
            elif self.agg_type == 'mean':
                out_embs = torch.sum(torch.stack(et_scores, dim=0), dim=0) / et_count
            else:
                out_embs = self.agg_fn(et_dst_data, dim=0)
        return out_embs, attn    
    