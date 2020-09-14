import torch
import numpy as np


def label_smoothing(label, label_smoothing):
    return (1.0 - label_smoothing) * label + (label_smoothing / label.shape[1])
    
def topNhit(rank, n):
    return np.sum(rank <= n) / len(rank)