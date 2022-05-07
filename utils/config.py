from dotmap import DotMap



args = DotMap()

# My Model Setting
args.conv_type = 'Hetero' # 'MLP', GAT', 'GCN', 'SAGE', 'RGCN', 'Transformer', 'EdgeTransformer', 'RTE'
# Hetero

args.rel_conv_type = 'Transformer'
args.rel_combine = 'query' # 'mean', 'trans_query'
# args.feat_combine = 'trans_query' # 'mean', 'trans_query'
args.recp_combine = 'transform' # 'transform'
args.multi_feat = 'all'
# 'emb', 'raw_attn', 'emb_attn', 'all', 'mean'
# 'competitive', 'profile', 'engagement', 'social', , 
args.multi_connection = 'none' # 'linear', 'none'
args.topk = 5
args.aug_layer = 2
args.edge_types =  ['FriendApply', 'TeamInvite', 'FriendApplyAgree', 'TeamInviteAgree', 'ZoneLeaveMessageGift']


args.sampler_type = 'in'
#'raw', 'in', 'out'


# args.edge_types = ['FriendApply', 'FriendApplyAgree', 'FriendApplyRefuse', 'GameSendLike', 'GameWatchGift', 'PrivateChat', 'TeamInvite', 'TeamInviteAgree', 'ZoneLeaveMessageGift']
args.graph_reversed = False

args.loss_type =  'BCE' # 'BCE'
args.conv_depth = 2

# Data Setting
args.num_nodes = 428803

args.label = 'TeamInvite'  
# 'walktogether_tag', 'active_tag', 'friend_tag', 'playagain_tag'
# 'TeamInvite', 'FriendApply' ,'TeamInviteAgree', 'FriendApplyAgree',
args.graph_hist_range = ('0427', '0511')
# '0412'
args.label_train_range = (12, 13)
args.label_test_range = (14, 14)




args.dim_hiddens = 64
args.dim_embs = 64
args.dim_query = 64

args.residual = False
args.act_type = 'leaky'
args.norm = 'right'
# GAT/Transformer setting
args.num_heads = 4
args.heads_fusion = 'concat'
args.edge_drop = 0
args.feat_drop = 0.4
args.attn_drop = 0.4
args.neg_k = 10

# Predictor Setting
args.predictor = '1-linear'

# Learning Setting
args.learning_rate = 0.01
args.loss_batch_size = 4096             # to calculated loss
# args.loss_batch_size = 128             # to calculated loss
args.inference_batch_size = 128       # the batch size for inferencing all/batched nodes embeddings
args.sampling_batch_size = 128

args.gpu = 0
args.num_workers = 0
args.verbose = True
args.epochs = 1


args.feat_dict = {
    'competitive': 7,
    'profile': 8,
    'engagement': 22,
    'social': 12,
    'all': 49
}
# args.feat_dict = {
#     'all': 10
# }
# args.feat_dict = {
#     'all': 61
# }
args.dim_features = args.feat_dict['all']
args.dim_edge_feats = 6

# Evaluation Setting
args.eval_batch_size = 102400 * 4
# args.eval_metrics = ['auc', 'gauc', 'precision', 'recall', 'f1', 'neg_loss']
args.eval_metrics = ['auc', 'gauc', 'ndcg', 'mrr', 'hits',  'f1', 'precision', 'recall', 'neg_loss']

# args.eval_metrics = ['auc']
#  'gauc'


def set_args(args, key, value):
    args[key] = value


def fix_args(args):
    if args.conv_type == 'MLP':
        args.conv_type = 'MLP'
        args.conv_depth = 2
        args.loss_batch_size = 1024000
    elif args.conv_type == 'LR':
        args.conv_type = 'MLP'
        args.conv_depth = 0
        args.loss_batch_size = 1024000
    elif args.conv_type == 'RGCN':
        args.loss_batch_size = 128
#     elif args.conv_type == 'Het':
#         args.loss_batch_size = 1024
#     else:
#         args.loss_batch_size = 10240

# set_args(args, 'edge_types',  ['FriendApplyAgree'])

