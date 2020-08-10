import time
import argparse
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import rankdata 

from RGAT.parser import parse_args
from RGAT.models import RGAT
from RGAT.load_data import Data

# Settings
args = parse_args()
torch.manual_seed(args.seed)

"""
===========================================================================
Loading data
===========================================================================
"""
data = Data(path=args.data_path + args.dataset, seed=123)

graph = data.graph
train = data.train
label_train = data.label_train
valid = data.valid
label_valid = torch.ones([valid.shape[0]])
test = data.test
label_test = torch.ones([test.shape[0]])
n_entity = data.n_entity
n_relation = data.n_relation

print('Loaded {0} dataset with {1} entities and {2} relations'.format(args.dataset, n_entity, n_relation))
"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = RGAT(n_entity=n_entity, n_relation=n_relation, dim=args.hidden, dropout=args.dropout, n_head=args.n_head)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    model.train()
    optimizer.zero_grad()
    output = model(graph, train)
    loss_train = F.binary_cross_entropy(input=output, target=label_train)
    acc_train = accuracy_score(y_pred=np.where(output.detach().numpy() < 0.5, 0, 1), y_true=label_train)
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(graph, valid)
    loss_valid = F.binary_cross_entropy(input=output, target=label_valid)
    acc_valid = accuracy_score(y_pred=np.where(output.detach().numpy() < 0.5, 0, 1), y_true=label_valid)

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train.item() ,loss_valid.item(), acc_train, acc_valid))

"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output = model(graph, test)
loss_test = F.binary_cross_entropy(input=output, target=label_test)
acc_test = accuracy_score(y_pred=np.where(output.detach().numpy() < 0.5, 0, 1), y_true=label_test)
print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test.item(), acc_test))

test_h = test[:, 0].unsqueeze(1).repeat(1, n_entity).view(-1, 1)
test_r = test[:, 1].unsqueeze(1).repeat(1, n_entity).view(-1, 1)
test_t = torch.LongTensor(range(n_entity)).view(-1, 1).repeat(test.shape[0], 1)

score = model(train, torch.cat([test_h, test_r, test_t], dim=1)).view(-1, n_entity)
rank_matrix = rankdata(-score.detach().numpy(), axis=1)
rank = rank_matrix[np.arange(len(rank_matrix)), test[:, 2].numpy()]
print(np.mean(rank))
print(rank)