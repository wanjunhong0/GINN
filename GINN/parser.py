import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run GINN.")
    parser.add_argument('--data_path', nargs='?', default='./data/', help='Input data path.')
    parser.add_argument('--model_path', nargs='?', default='checkpoint.pt', help='Saved model path.')
    parser.add_argument('--dataset', nargs='?', default='kinship', help='Choose a dataset from {FB15K-237, WN18RR, kinship}.')
    parser.add_argument('--reverse', action='store_true', help='Reverse triples to test head and tail entities.')

    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epoch', type=int, default=5000, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of sample per batch.')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--head', type=int, default=1, help='Number of head attentions.')
    parser.add_argument('--channel', type=int, default=32, help='Number of convolution channels.')
    parser.add_argument('--kernel', type=int, default=3, help='Kernel size of N*N.')
    parser.add_argument('--attention', nargs='?', default='GINN', help='The model of attention in {GINN, GAT, None}.')
    parser.add_argument('--score_func', nargs='?', default='ConvE', help='The model of scoring function in {ConvE, DistMult}')
    parser.add_argument('--reshape_size', type=int, default=10, help='The reshape size of ConvE.')

    parser.add_argument('--evaluation', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--patience', type=int, default=10, help='How long to wait after last time validation improved')
    parser.add_argument('--device', nargs='?', default='cuda:0', help='Which device to run on')

    return parser.parse_args()
