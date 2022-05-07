import dill
import dgl
import torch

from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd


def load_reciprocal_graph(hist_range, bidirected=False, simple=False):
    het_num_nodes_dict = {}
    with open("./cym_data/user_dict.pkl", "rb") as dill_file:
        dict_user2id = dill.load(dill_file)
    het_num_nodes_dict['user'] = len(dict_user2id)
    
    et_list = ['TeamInviteCoRefused', 'TeamInviteCoRefuse', 'TeamInviteCoAgree', 'TeamInviteAgree', 
               'FriendApplyCoRefused', 'FriendApplyCoRefuse', 'FriendApplyCoAgree', 'FriendApplyAgree']
    het_data_dict = {}
    for et in et_list:
        period_start = pd.Timestamp('2020' + hist_range[0])
        period_end = pd.Timestamp('2020' + hist_range[1]) + pd.Timedelta(1, unit='D')
        day = period_start
        edge_df_list = []
        while day < period_end:
            with open(f"./cym_data/{et}/{day.strftime('%Y-%m-%d')}.pkl", "rb") as dill_file:
                edge_df_list.append(dill.load(dill_file))
            day = day + pd.Timedelta(1, unit='D')
        edge_df = pd.concat(edge_df_list, axis=0)
        us = edge_df['src'].values
        vs = edge_df['dst'].values
        het_data_dict[('user', et, 'user')] = (us, vs)

    g = dgl.heterograph(data_dict=het_data_dict, num_nodes_dict=het_num_nodes_dict)
    het_node_feat_dict = setup_node_feat(g)
    
    if bidirected:
        g = dgl.to_bidirected(dgl.to_simple(g), copy_ndata=True)
        
    return g


def load_graph(hist_range, bidirected=False, simple=False):
    het_num_nodes_dict = {}
    with open("./cym_data/user_dict.pkl", "rb") as dill_file:
        dict_user2id = dill.load(dill_file)
    het_num_nodes_dict['user'] = len(dict_user2id)
    
    edge_df_list = []
    period_start = pd.Timestamp('2020' + hist_range[0])
    period_end = pd.Timestamp('2020' + hist_range[1]) + pd.Timedelta(1, unit='D')
    day = period_start
    while day < period_end:
        with open(f"./cym_data/g/g_{day.strftime('%Y-%m-%d')}.pkl", "rb") as dill_file:
            edge_df_list.append(dill.load(dill_file))
        day = day + pd.Timedelta(1, unit='D')
    edge_df = pd.concat(edge_df_list, axis=0)
    
    if bidirected:
        rev_edge_df = edge_df.rename({'role_id': 'target_id', 'target_id': 'role_id'}, axis=1)
        edge_df = pd.concat([edge_df, rev_edge_df], axis=0)
    if simple:
        edge_df = edge_df.groupby(['role_id', 'target_id', 'prefer_type']).mean().reset_index()
    
    het_data_dict = {}
    us = edge_df.role_id.values
    vs = edge_df.target_id.values
    for pf_type in edge_df.prefer_type.unique():
        pf_type_mask = (edge_df.prefer_type==pf_type).values
        het_data_dict[('user', pf_type, 'user')] = (us[pf_type_mask], vs[pf_type_mask])
    
    g = dgl.heterograph(data_dict=het_data_dict, num_nodes_dict=het_num_nodes_dict)
    het_node_feat_dict = setup_node_feat(g, feat_type='cate') # 'cate', 'num',  'all'
    het_edge_feat_dict = setup_edge_feat(g, edge_df)
    return g, het_node_feat_dict, het_node_feat_dict
    
    

    
def setup_node_feat(g, feat_type):
    if feat_type == 'cate':
        het_node_feat_dict = {}
        with open("./cym_data/user_feat_cate.pkl", "rb") as dill_file:
            mat = dill.load(dill_file)
            user_feat = mat['feat']
            cat2feat = mat['cat2feat']   
        het_node_feat_dict['user'] = user_feat

        for node_t in het_node_feat_dict:
            df = het_node_feat_dict[node_t]
            for f in ['competitive', 'profile', 'engagement', 'social']:
                g.nodes[node_t].data[f] = torch.from_numpy(df[cat2feat[f]].values).float()
            g.nodes[node_t].data['all'] = torch.from_numpy(df.values).float()    
        return het_node_feat_dict
    elif feat_type == 'num':
        het_node_feat_dict = {}
        with open("./cym_data/user_feat_num.pkl", "rb") as dill_file:
            df = dill.load(dill_file)  
        het_node_feat_dict['user'] = torch.from_numpy(df.values).float()    
        g.nodes['user'].data['all'] = torch.from_numpy(df.values).float()    
        return het_node_feat_dict
    elif feat_type == 'all':
        het_node_feat_dict = {}
        with open("./cym_data/user_feat_all.pkl", "rb") as dill_file:
            df = dill.load(dill_file)  
        het_node_feat_dict['user'] = df
        g.nodes['user'].data['all'] = torch.from_numpy(df.values).float()    
        return het_node_feat_dict

def setup_edge_feat(g, edge_df):
    e_fs = edge_df.loc[:, 'b1day_friend_tag': 'count_bad']
    het_edge_feat_dict = {}
    for pf_type in edge_df.prefer_type.unique():
        pf_type_mask = (edge_df.prefer_type==pf_type).values
        het_edge_feat_dict[('user', pf_type, 'user')] = e_fs[pf_type_mask]
    for edge_t in het_edge_feat_dict:
        g.edges[edge_t].data['feature'] = torch.from_numpy(het_edge_feat_dict[edge_t].values).float()
#             s = het_edge_feat_dict[edge_t]['t_when']
#             if ts_freq == 'hour':
#                 g.edges[edge_t].data['ts'] = torch.from_numpy((s.dt.day_of_year * 24 + s.dt.hour).values)
    return het_edge_feat_dict

    
class NegDataset(object):
    def __init__(self, dataset, args):
        super().__init__()
        df = dataset.pair_df
        pos_df = df[df[args.label] == 1]
        self.args = args
        self.g = dgl.graph((pos_df.src.tolist(), pos_df.dst.tolist()), num_nodes=428803)
        self.eid_loader = torch.utils.data.DataLoader(self.g.edges('eid'), batch_size=args.loss_batch_size, shuffle=True, num_workers=4)
        self.neg_sampler = dgl.dataloading.negative_sampler.Uniform(args.neg_k)
    
    def __iter__(self):
        return self.generator()
    
    def __len__(self):
        return len(self.eid_loader)
    
    def generator(self):
        for eids in self.eid_loader:
            us, ns = self.neg_sampler(self.g, eids)
            ps = self.g.find_edges(eids)[1].repeat_interleave(self.args.neg_k)
            yield us, ps, ns
    
    
class MyData(object):
    def __init__(self, spec, lbl, lbl_range=None):
        '''
        use self.pair_df to access the raw pair df.
        use self.tri_df to access the triplet df.
        use self.triplets to access the triplets.
        '''
        super().__init__()
        self.build(spec, lbl, lbl_range)        

    def build(self, spec, lbl, lbl_range):
        self.label = lbl
        self.raw_df = self.load_labels(spec, lbl, lbl_range)
        self.tri_df = self.build_triplet_df(self.raw_df, lbl)
        self.triplets = self.build_triplets(self.tri_df)
        self.pair_df = self.build_pair_df(self.tri_df, lbl)
        self.pairs = self.build_pairs(self.pair_df)
        self.user_ids = self.triplets.flatten().unique()
    
    def load_labels(self, spec, lbl, lbl_range):
        if lbl in ['walktogether_tag', 'active_tag', 'friend_tag', 'playagain_tag']:  
            with open(f"./cym_label/{spec}_{lbl}.pkl", "rb") as dill_file:
                df = dill.load(dill_file)
        elif lbl in ['TeamInviteAgree', 'FriendApplyAgree', 'TeamInvite', 'FriendApply']:
            lbl_start, lbl_end = lbl_range
            lbl_list = []
            for day in range(lbl_start, lbl_end+1):
                with open(f"./cym_label/{lbl}/2020-05-{day:02d}.pkl", "rb") as dill_file:
                    lbl_list.append(dill.load(dill_file))
            df = pd.concat(lbl_list, axis=0)
        return df
    
    def build_pair_df(self, tri_df, lbl):
        p_list = []
        n_list = []
        for _, u, ps, ns in tri_df.itertuples():
            p_list.extend(list(product([u], ps)))
            n_list.extend(list(product([u], ns)))
        labels = np.concatenate([np.ones(len(p_list)), np.zeros(len(n_list))]).astype(np.int8)
        data = np.array(p_list + n_list)
        df = pd.DataFrame(data, columns=['src', 'dst'])
        df[lbl] = labels
        return df
    
    def build_pairs(self, df):
        pairs = []
        for _,u,v,l in df.itertuples():
            pairs.append([u, v, l])
        return torch.as_tensor(pairs)
    
    def build_triplet_df(self, df, label):
        pos_dict = defaultdict(list)
        neg_dict = defaultdict(list)
        for u,v,y in zip(df['src'], df['dst'], df[label]):
            if y:
                pos_dict[u].append(v)
            else:
                neg_dict[u].append(v)
        user_ids = pd.Series(np.unique(df.src))
        res_df = pd.DataFrame({
            'user': user_ids,
            'pos': user_ids.map(pos_dict),
            'neg': user_ids.map(neg_dict)
        })
        if 'TeamInvite' in self.label:
            comp_mask = ((res_df['neg'].map(len) > 10) & (res_df['pos'].map(len) > 10))
        if 'FriendApply' in self.label:
            comp_mask = ((res_df['neg'].map(len) > 5) & (res_df['pos'].map(len) > 5))
        res_df = res_df[comp_mask].reset_index(drop=True)
        return res_df
    
    def build_triplets(self, tri_df):
        triplets = []
        for _, u,p,n in tri_df.itertuples():
            triplets.extend(list(product([u], p, n)))
        return torch.as_tensor(triplets)
    
    
    
class MyLoader(object):
    def __init__(self, data, batch_size, loss_type):
        self.data = data
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.user_list = np.unique(self.data.pair_df.src)
        self.length = len(self.user_list)
        self.ui_dict = self.build_interaction_dict(self.data.pair_df)
        
    def __iter__(self):
        return self.user_pair_generator(self.batch_size)
    
    def __len__(self):
        return self.length // self.batch_size + 1
    
    def build_interaction_dict(self, user_df):
        ui_dict = {}
        for u in self.user_list:
            batch_interactions = user_df[user_df.src == u]
            ui_dict[u] = torch.from_numpy(batch_interactions.values)
        return ui_dict
            
    def user_pair_generator(self, batch_size):
        idxs = np.random.permutation(range(len(self.user_list)))
        user_list = self.user_list[idxs]
        for start in range(0, self.length, batch_size): 
            batch_ui = [
                self.ui_dict[u] for u in user_list[start: start+batch_size]
            ]
            yield torch.cat(batch_ui, dim=0)
    
#     def user_triplet_generator(self, batch_size):
#         df = self.data.tri_df
#         idxs = np.random.permutation(range(self.length))
#         for start in range(0, self.length, batch_size): 
#             batch_data = df.iloc[idxs[start: start+batch_size]]
#             data = []
#             for _, u, ps, ns in batch_data.itertuples():
#                 data.extend(list(product([u], ps, ns)))
#             yield torch.as_tensor(data)    
            
            
            
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, loss_type):
        self.loss_type = loss_type
        self._data_obj = data
        
        self.eval_data = self._data_obj.pairs
        if loss_type == 'BCE':
            self.iter_data = self._data_obj.pairs
        elif loss_type == 'BPR':
            self.iter_data = self._data_obj.triplets
        elif loss_type == 'BPR_generate':
            self.iter_data = self._data_obj.tri_df
    
    def __getitem__(self, index):
        if self.loss_type == 'BCE':
            return self.iter_data[index]
        elif self.loss_type == 'BPR':
            return self.iter_data[index]
        elif self.loss_type == 'BPR_generate':
            u = self.iter_data.iloc[index].user
            p = np.random.choice(self.iter_data.iloc[index].pos, 1)
            ps = torch.as_tensor(p).repeat(10)
            ns = np.random.randint(len(self.iter_data), size=(len(ps),))
            ns = self.iter_data.user[ns].values
            ns = torch.as_tensor(ns)
            us = torch.as_tensor([u] * len(ps))
            return torch.stack([us, ps, ns], dim=1)
    
    def __len__(self):
        return len(self.iter_data)
    
    def __iter__(self):
        return self.generator()
    

    
# def load_signed_graph(hist_range, bidirected=False, simple=False):
#     het_num_nodes_dict = {}
#     with open("./cym_data/user_dict.pkl", "rb") as dill_file:
#         dict_user2id = dill.load(dill_file)
#     het_num_nodes_dict['user'] = len(dict_user2id)
    
#     edge_df_list = []
#     period_start = pd.Timestamp('2020' + hist_range[0])
#     period_end = pd.Timestamp('2020' + hist_range[1]) + pd.Timedelta(1, unit='D')
#     day = period_start
#     while day < period_end:
#         with open(f"./cym_data/signed_g/g_{day.strftime('%Y-%m-%d')}.pkl", "rb") as dill_file:
#             edge_df_list.append(dill.load(dill_file))
#         day = day + pd.Timedelta(1, unit='D')
#     edge_df = pd.concat(edge_df_list, axis=0)
# #     return edge_df
#     het_data_dict = {}
#     us = edge_df.role_id.values
#     vs = edge_df.target_id.values
#     het_data_dict[('user', 'FriendApply', 'user')] = (us, vs)
#     g = dgl.heterograph(data_dict=het_data_dict, num_nodes_dict=het_num_nodes_dict)
#     het_node_feat_dict = setup_node_feat(g)
#     g.edata['attn_labels'] = torch.from_numpy(edge_df['attn_labels'].values).float()
#     return g


def load_tj_graph(lbl):
    file_dir = '/home/changym/NetEase_txt/cym/cym_data/tom_jerry/1_8_12_16'
    file_name = file_dir + '/social/graph_user_reverse.pkl'
    with open(file_name, 'rb') as f:
        graph = dill.load(f, encoding='utf-8')
    if 'TeamInvite' in lbl:
        file_name = file_dir + '/user_feature_invite_select10.pkl'
    if 'FriendApply' in lbl:
        file_name = file_dir + '/user_feature_friend_select10.pkl'
#     file_name = file_dir + '/user_feature_all_numpy.pkl'
    with open(file_name, 'rb') as f:
        user_feat = dill.load(f, encoding='utf-8' )
    user_feat = user_feat.astype(np.float32)
    graph.ndata['all'] = torch.from_numpy(user_feat)
    return graph

def load_tj_label(lbl):
    # file_dir = '/data1/duerx/pkl_data_all/link_based/1_7_11_15'
    file_dir = '/home/changym/NetEase_txt/cym/cym_data/tom_jerry/1_8_12_16'
    file_name = file_dir + f'/{lbl}_label_filter_reverse.pkl'
    with open(file_name, 'rb') as f:
        train_label = dill.load(f, encoding='utf-8' )
        test_label = dill.load(f, encoding='utf-8' )
    train_label.columns = ['src', 'dst', lbl]
    test_label.columns = ['src', 'dst', lbl]
    
#     train_label = train_label.reset_index(drop=True)
#     for index, row in train_label.iterrows():
#         if row[lbl] == 0:
#             row['src'], row['dst'] = row['dst'], row['src']
#         train_label.iloc[index] = row

#     test_label = test_label.reset_index(drop=True)
#     for index, row in test_label.iterrows():
#         if row[lbl] == 0:
#             row['src'], row['dst'] = row['dst'], row['src']
#     test_label.iloc[index] = row
    
    train_label = TomData(train_label, lbl)
    test_label = TomData(test_label, lbl)
    return train_label, test_label


class TomData(object):
    def __init__(self, df, lbl):
        super().__init__()
        self.label = lbl
        self.build_from_df(df, lbl)        

#     def build(self, spec, lbl):
#         self.raw_df = self.load_labels(spec, lbl)
#         self.tri_df = self.build_triplet_df(self.raw_df, lbl)
#         self.triplets = self.build_triplets(self.tri_df)
#         self.pair_df = self.build_pair_df(self.tri_df, lbl)
#         self.pairs = self.build_pairs(self.pair_df)
#         self.user_ids = self.triplets.flatten().unique()
    
    def build_from_df(self, df, lbl):
        self.raw_df = df
        self.tri_df = self.build_triplet_df(self.raw_df, lbl)
        self.triplets = self.build_triplets(self.tri_df)
        self.pair_df = self.build_pair_df(self.tri_df, lbl)
        self.pairs = self.build_pairs(self.pair_df)
        self.user_ids = self.triplets.flatten().unique()
    
    def build_pair_df(self, tri_df, lbl):
        p_list = []
        n_list = []
        for _, u, ps, ns in tri_df.itertuples():
            p_list.extend(list(product([u], ps)))
            n_list.extend(list(product([u], ns)))
        labels = np.concatenate([np.ones(len(p_list)), np.zeros(len(n_list))]).astype(np.int8)
        data = np.array(p_list + n_list)
        df = pd.DataFrame(data, columns=['src', 'dst'])
        df[lbl] = labels
        return df
    
    def build_pairs(self, df):
        pairs = []
        for _,u,v,l in df.itertuples():
            pairs.append([u, v, l])
        return torch.as_tensor(pairs)
    
    def build_triplet_df(self, df, label):
        pos_dict = defaultdict(list)
        neg_dict = defaultdict(list)
        for u,v,y in zip(df['src'], df['dst'], df[label]):
            if y:
                pos_dict[u].append(v)
            else:
                neg_dict[u].append(v)
        user_ids = pd.Series(np.unique(df.src))
        res_df = pd.DataFrame({
            'user': user_ids,
            'pos': user_ids.map(pos_dict),
            'neg': user_ids.map(neg_dict)
        })
        if 'TeamInvite' in self.label:
            comp_mask = ((res_df['neg'].map(len) > 5) & (res_df['pos'].map(len) > 5))
        if 'FriendApply' in self.label:
            comp_mask = ((res_df['neg'].map(len) > 5) & (res_df['pos'].map(len) > 5))
        res_df = res_df[comp_mask].reset_index(drop=True)
        return res_df
    
    def build_triplets(self, tri_df):
        triplets = []
        for _, u,p,n in tri_df.itertuples():
            triplets.extend(list(product([u], p, n)))
        return torch.as_tensor(triplets)