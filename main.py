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

train = data.train
val = data.val
test = data.test

label_train = data.label_train
label_val = data.label_val
label_test = data.label_test
label_train = label_smoothing(label_train, args.label_smoothing)
label_val = label_smoothing(label_val, args.label_smoothing)
label_test = label_smoothing(label_test, args.label_smoothing)

triple_train = data.triple_train
triple_val = data.triple_val
triple_test = data.triple_test



print('Loaded {0} dataset with {1} entities and {2} relations'.format(args.dataset, n_entity, n_relation))
"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = RGAT(n_entity=n_entity, n_relation=n_relation, dim=args.hidden, dropout=args.dropout, 
             n_head=args.n_head, n_channel=args.n_channel, kernel_size=args.kernel)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss = torch.nn.BCELoss()

for epoch in range(1, args.epoch+1):
    t = time.time()

    model.train()
    optimizer.zero_grad()
    output = model(triple_train, train)
    loss_train = loss(input=output, target=label_train)
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(triple_train, val)
    loss_valid = loss(input=output, target=label_val)


    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}]'
         .format(epoch, time.time() - t, loss_train.item() ,loss_valid.item()))

    if epoch % args.evaluation == 0:
        t1 = time.time()
        score_train = model(triple_train, triple_train[:, :2])
        rank_train = get_ranking(-score_train.detach().numpy(), triple_train)
        score_val = model(triple_train, triple_val[:, :2])
        rank_val = get_ranking(-score_val.detach().numpy(), triple_val)
        print('===================Evaluation on Epoch {0:04d}==================='.format(epoch))
        print('MR = [train: {0:.4f}, val: {1:.4f}]'.format(np.mean(rank_train), np.mean(rank_val)))
        print('MRR = [train: {0:.4f}, val: {1:.4f}]'.format(np.mean(np.power(rank_train, -1)), np.mean(np.power(rank_val, -1))))
        print('TOP1 = [train: {0:.4f}, val: {1:.4f}]'.format(topNhit(rank_train, 1), topNhit(rank_val, 1)))
        print('TOP3 = [train: {0:.4f}, val: {1:.4f}]'.format(topNhit(rank_train, 3), topNhit(rank_val, 3)))
        print('TOP10 = [train: {0:.4f}, val: {1:.4f}]'.format(topNhit(rank_train, 10), topNhit(rank_val, 10)))
        print('============Evaluation completed using time: {0:.2f}s============'.format(time.time() - t1))
        


"""
===========================================================================
Testing
===========================================================================
"""
model.eval()
output = model(triple_train, test)
loss_test = F.binary_cross_entropy(input=output, target=label_test)

score_test = model(triple_train, triple_test[:, :2])
rank_test = get_ranking(-score_test.detach().numpy(), triple_test)
print('===========================Testing============================')
print('Loss = [test: {0:.4f}]'.format(loss_test.item()))
print('MR = [test: {0:.4f}]'.format(np.mean(rank_test)))
print('MRR = [test: {0:.4f}]'.format(np.mean( np.mean(np.power(rank_test, -1)))))
print('TOP1 = [test: {0:.4f}]'.format(topNhit(rank_test, 1)))
print('TOP3 = [test: {0:.4f}]'.format(topNhit(rank_test, 3)))
print('TOP10 = [test: {0:.4f}]'.format(topNhit(rank_test, 10)))
