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
model = RGAT(n_entity=n_entity, n_relation=n_relation, dim=args.hidden, dropout=args.dropout, n_head=args.n_head)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epoch+1):
    t = time.time()
    # Training
    train_neg = data.neg_sampling(data.train, 'tail')
    model.train()
    optimizer.zero_grad()
    output = model(train, torch.cat([train, train_neg], dim=0))
    loss_train = F.binary_cross_entropy(input=output, target=label_train)
    acc_train = accuracy_score(y_pred=np.where(output.detach().numpy() < 0.5, 0, 1), y_true=label_train.round())
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(train, val)
    loss_valid = F.binary_cross_entropy(input=output, target=label_val)
    acc_valid = accuracy_score(y_pred=np.where(output.detach().numpy() < 0.5, 0, 1), y_true=label_val)

    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}] | ACC = [train: {4:.4f}, val: {5:.4f}]'
          .format(epoch, time.time() - t, loss_train.item() ,loss_valid.item(), acc_train, acc_valid))

"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output = model(train, test)
loss_test = F.binary_cross_entropy(input=output, target=label_test)
acc_test = accuracy_score(y_pred=np.where(output.detach().numpy() < 0.5, 0, 1), y_true=label_test)
print('======================Testing======================')
print('Loss = [test: {0:.4f}] | ACC = [test: {1:.4f}]'.format(loss_test.item(), acc_test))


train_ranking = prepare_ranking_input(train, n_entity)
score = model(train, train_ranking).view(-1, n_entity)
rank = get_ranking(-score.detach().numpy(), train, 'tail')

print(np.mean(rank))
