import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run R-GAT.")
    parser.add_argument('--data_path', nargs='?', default='./data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ops_new', help='Choose a dataset from {FB15K-237, WN18RR, ops}.')
    parser.add_argument('--reverse', action='store_true', help='Reverse triples to test head and tail entities.')

    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epoch', type=int, default=3000, help='Number of epochs to train.')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--n_head', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--n_channel', type=int, default=32, help='Number of convolution channels.')
    parser.add_argument('--kernel', type=int, default=3, help='Kernel size of N*N.')
    parser.add_argument('--attention', nargs='?', default='GINN', help='The model of attention in {GINN, GAT, None}.')
    parser.add_argument('--score_func', nargs='?', default='ConvE', help='The model of scoring function in {ConvE, DistMult}')
    parser.add_argument('--reshape_size', type=int, default=10, help='The reshape size of ConvE.')

    parser.add_argument('--evaluation', type=int, default=20, help='Evaluation interval')
    parser.add_argument('--patience', type=int, default=20, help='How long to wait after last time validation improved')

    return parser.parse_args()
