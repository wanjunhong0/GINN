import torch
from scipy.stats import rankdata 
import numpy as np


def label_smoothing(label, label_smoothing):
    return (1.0 - label_smoothing) * label + label_smoothing / label.max()

def prepare_ranking_input(data, n_entity):
    h = data[:, 0].unsqueeze(1).repeat(1, n_entity).view(-1, 1)
    r = data[:, 1].unsqueeze(1).repeat(1, n_entity).view(-1, 1)
    t = torch.LongTensor(range(n_entity)).view(-1, 1).repeat(data.shape[0], 1)
    return torch.cat([h, r, t], dim=1)

def get_ranking(score, data, predict):
    rank = rankdata(score, axis=1)
    if predict == 'tail':
        index = data[:, 2]
    if predict == 'head':
        index = data[:, 0]
    return rank[np.arange(len(rank)), index]

def topNhit(rank, n):
    return np.sum(rank <= n) / len(rank)