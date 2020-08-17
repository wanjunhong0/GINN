import time
import argparse
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np

from RGAT.parser import parse_args
from RGAT.models import RGAT
from RGAT.load_data import Data
from RGAT.utils import *

# Settings
args = parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)

"""
===========================================================================
Loading data
===========================================================================
"""
data = Data(path=args.data_path + args.dataset)
n_entity = data.n_entity
n_relation = data.n_relation

train = torch.LongTensor(data.train.values)
val = torch.LongTensor(data.val.values)
test = torch.LongTensor(data.test.values)

label_train = torch.cat([torch.ones([train.shape[0]]), torch.zeros([train.shape[0]])])
label_train = label_smoothing(label_train, args.label_smoothing)
label_val = torch.ones([val.shape[0]])
label_test = torch.ones([test.shape[0]])



print('Loaded {0} dataset with {1} entities and {2} relations'.format(args.dataset, n_entity, n_relation))
"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = RGAT(n_entity=n_entity, n_relation=n_relation, dim=args.hidden, dropout=args.dropout, 
             n_head=args.n_head, n_channel=args.n_channel, kernel_size=args.kernel)
optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    train_neg = data.neg_sampling(data.train, 'tail')
    model.train()
    optimizer.zero_grad()
    output = model(train, torch.cat([train, train_neg], dim=0))
    loss_train = F.binary_cross_entropy(input=output, target=label_train)
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(train, val)
    loss_valid = F.binary_cross_entropy(input=output, target=label_val)

    val_ranking = prepare_ranking_input(val, n_entity)
    score = model(train, val_ranking).view(-1, n_entity)
    rank_val = get_ranking(-score.detach().numpy(), val, 'tail')

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | MRR = [val: {4:.4f}]'
          .format(epoch, time.time() - t, loss_train.item() ,loss_valid.item(), np.mean(np.power(rank_val, -1))))

"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output = model(train, test)
loss_test = F.binary_cross_entropy(input=output, target=label_test)

test_ranking = prepare_ranking_input(test, n_entity)
score = model(train, test_ranking).view(-1, n_entity)
rank_test = get_ranking(-score.detach().numpy(), test, 'tail')
print('======================Testing======================')
print('Loss = [test: {0:.4f}] | MRR = [test: {1:.4f}]'.format(loss_test.item(), np.mean(np.power(rank_test, -1))))
