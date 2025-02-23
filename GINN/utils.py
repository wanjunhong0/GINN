import torch
import numpy as np


def label_smoothing(label, rate):
    """
    Args:
        label (torch tensor): label
        rate (float): the rate of smoothing (0-1)

    Returns:
        (torch tensor): label after smoothing
    """
    return (1.0 - rate) * label + (rate / label.shape[1])


def rank_filter(score, filter, label, index):
    """Rank against all other candidate triple not appearing in the train, val and test

    Args:
        score (torch tensor): score matrix
        filter (torch tensor): one-hot filter matrix
        label (torch tensor): label

    Returns:
        (torch tensor): rank matrix after filter
    """
    filter_score = torch.mul(score, filter)
    rank = filter_score.argsort().argsort()
    filter_rank = torch.mul(rank, label).argsort().argsort()[index] - rank[index] + 1. # need float dtype to calculate mean
    return filter_rank


def topNhit(rank, n):
    """
    Args:
        rank (torch tensor): ranking matrix
        n (int): top n wanted

    Returns:
        (float): the rate of results within topN, (ranking <=3) / # of ranking
    """
    return (rank <= n).sum().item() / len(rank)


class EarlyStopping:
    """Early stops the training if validation metrics doesn't improve after a given patience."""
    def __init__(self, patience=10, mode='min', delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation improved.
            mode (str): Max or min is prefered improvement
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val, model):
        if self.mode == 'min':
            score = -val
        if self.mode == 'max':
            score = val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        self.val_min = val_loss
