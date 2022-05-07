import argparse

from utils.config import fix_args

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--conv_depth', type=int, default=2)
    parser.add_argument('--loss_batch_size', type=int, default=10240)
    parser.add_argument('--inference_batch_size', type=int, default=128)
    parser.add_argument('--sampling_batch_size', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=1)

    parser.add_argument('--conv_type', type=str, default='GCN', choices=['MLP', 'GAT', 'GCN', 'SAGE', 'RGCN', 'Transformer', 'EdgeTransformer', 'RTE', 'SuperGAT', 'LGC', 'NGCF', 'Hetero', 'Reciprocal'])
    parser.add_argument('--loss_type', type=str, default='BCE', choices=['BCE', 'BPR'])
    parser.add_argument('--sampler_type', type=str, default='bidirected', choices=['bidirected', 'in', 'out'])
    parser.add_argument('--multi_feat', type=str, default='all', choices=['all'])
    parser.add_argument('--multi_connection', type=str, default='none', choices=['linear', 'none'])

    parser.add_argument('--label', type=str, default='TeamInvite', choices=['TeamInvite', 'FriendApply' ,'TeamInviteAgree', 'FriendApplyAgree'])
    parser.add_argument('--edge_types', nargs='+', type=str, default=['FriendApply', 'TeamInvite', 'FriendApplyAgree', 'TeamInviteAgree', 'ZoneLeaveMessageGift'])

    parser.add_argument('--verbose', type=int, default=1, choices=[1, 0])
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    args = fix_args(args)
    return args