import torch
import numpy as np
from scipy.stats import rankdata 


def label_smoothing(label, label_smoothing):
    """
    Args:
        label (torch tensor): label
        label_smoothing (float): the rate of smoothing (0-1)

    Returns:
        (torch tensor): label after smoothing
    """
    return (1.0 - label_smoothing) * label + (label_smoothing / label.shape[1])


def rank_filter(score, filter, label):
    """Rank against all other candidate triple not appearing in the train, val and test

    Args:
        score (torch tensor): score matrix
        filter (torch tensor): one-hot filter matrix
        label (torch tensor): label

    Returns:
        (numpy array): rank matrix after filter
    """
    filter_score = torch.mul(score, filter)
    multi = torch.mul(filter_score, label)
    rank = rankdata(-filter_score.detach().numpy(), axis=1) - rankdata(-multi.detach().numpy(), axis=1) + label.detach().numpy()
    return rank

    
def topNhit(rank, n):
    """
    Args:
        rank (numpy array): ranking matrix
        n (int): top n wanted 

    Returns:
        (float): the rate of results within topN, (ranking <=3) / # of ranking
    """
    return np.sum(rank <= n) / len(rank)


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
