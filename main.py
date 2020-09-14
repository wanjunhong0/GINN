import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import rankdata 

from GINN.parser import parse_args
from GINN.models import GINN
from GINN.load_data import Data
from GINN.utils import label_smoothing, topNhit

# Settings
args = parse_args()
# print configuation
for arg in vars(args):
    print('{0} = {1}'.format(arg, getattr(args, arg)))
torch.manual_seed(args.seed)

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
model = GINN(n_entity=n_entity, n_relation=n_relation, dim=args.hidden, dropout=args.dropout, 
             n_head=args.n_head, n_channel=args.n_channel, kernel_size=args.kernel)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epoch+1):
    t = time.time()

    model.train()
    optimizer.zero_grad()
    output = model(triple_train, train)
    loss_train = F.binary_cross_entropy(input=output, target=label_train)
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(triple_train, val)
    loss_valid = F.binary_cross_entropy(input=output, target=label_val)


    print('Epoch {0:04d} | time: {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}]'
         .format(epoch, time.time() - t, loss_train.item() ,loss_valid.item()))

    if epoch % args.evaluation == 0:
        t1 = time.time()

        score_val = model(triple_train, val)
        score_val = torch.mul(score_val, data.filter_val)
        rank_val = rankdata(-score_val.detach().numpy(), axis=1)[data.index_val, triple_val[:, 2]]

        print('===================Evaluation on Epoch {0:04d}==================='.format(epoch))
        print('MR = val: {0:.4f}]'.format(np.mean(rank_val)))
        print('MRR = val: {0:.4f}]'.format(np.mean(np.power(rank_val, -1))))
        print('TOP1 = val: {0:.4f}]'.format(topNhit(rank_val, 1)))
        print('TOP3 = val: {0:.4f}]'.format(topNhit(rank_val, 3)))
        print('TOP10 =  val: {0:.4f}]'.format( topNhit(rank_val, 10)))
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
score_test = torch.mul(score_test, data.filter_test)
rank_test = rankdata(-score_val.detach().numpy(), axis=1)[data.index_test, triple_test[:, 2]]
print('===========================Testing============================')
print('Loss = [test: {0:.4f}]'.format(loss_test.item()))
print('MR = [test: {0:.4f}]'.format(np.mean(rank_test)))
print('MRR = [test: {0:.4f}]'.format(np.mean( np.mean(np.power(rank_test, -1)))))
print('TOP1 = [test: {0:.4f}]'.format(topNhit(rank_test, 1)))
print('TOP3 = [test: {0:.4f}]'.format(topNhit(rank_test, 3)))
print('TOP10 = [test: {0:.4f}]'.format(topNhit(rank_test, 10)))
