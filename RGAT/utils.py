import torch
from scipy.stats import rankdata 
import numpy as np


def label_smoothing(label, label_smoothing):
    return (1.0 - label_smoothing) * label + label_smoothing / label.max()

def get_ranking(score, data, predict='tail'):
    rank = rankdata(score, axis=1)
    if predict == 'tail':
        index = data[:, 2]
    if predict == 'head':
        index = data[:, 0]
    return rank[np.arange(len(rank)), index]

def topNhit(rank, n):
    return np.sum(rank <= n) / len(rank)