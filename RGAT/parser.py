import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run R-GAT.")
    parser.add_argument('--data_path', nargs='?', default='./data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ops', help='Choose a dataset from {FB15K-237, ops}')

    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--n_head', type=int, default=4, help='Number of head attentions.')

    return parser.parse_args()
