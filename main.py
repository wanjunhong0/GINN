import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GINN.parser import parse_args
from GINN.models import GINN
from GINN.load_data import Data, Dataset, collate
from GINN.utils import label_smoothing, rank_filter, topNhit, EarlyStopping

# Settings
args = parse_args()
for arg in vars(args):
    print('{0} = {1}'.format(arg, getattr(args, arg)))
torch.manual_seed(args.seed)
# training on the first GPU if available otherwise on CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on device = {}'.format(device))
early_stop = EarlyStopping(patience=args.patience, mode='max')
"""
===========================================================================
Loading data
===========================================================================
"""
data = Data(path=args.data_path + args.dataset, reverse=args.reverse)
n_entity = data.n_entity
n_relation = data.n_relation
train = Dataset(data.triple_train, data.train)
val = Dataset(data.triple_val, data.val)
test = Dataset(data.triple_test, data.test)
train_loader = DataLoader(dataset=train, batch_size=args.batch_size, collate_fn=collate)
val_loader = DataLoader(dataset=val, batch_size=args.batch_size, collate_fn=collate)
test_loader = DataLoader(dataset=test, batch_size=args.batch_size, collate_fn=collate)

label_train = label_smoothing(data.label_train, args.label_smoothing)
label_val = label_smoothing(data.label_val, args.label_smoothing)
label_test = label_smoothing(data.label_test, args.label_smoothing)

print('Loaded {0} dataset with {1} entities and {2} relations'.format(args.dataset, n_entity, n_relation))

"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = GINN(n_entity=n_entity, n_relation=n_relation, dim=args.hidden, dropout=args.dropout, 
             n_head=args.n_head, n_channel=args.n_channel, kernel_size=args.kernel, 
             attention=args.attention, score_func=args.score_func, reshape_size=args.reshape_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epoch+1):
    t = time.time()

    model.train()
    loss_train = 0
    for triple_batch, h_r_batch, idx_batch in train_loader:
        optimizer.zero_grad()
        output = model(triple_batch.to(device), h_r_batch.to(device)).cpu()
        loss = F.binary_cross_entropy(input=output, target=label_train[idx_batch])
        loss.backward()
        optimizer.step()
        loss_train += loss.item() * len(idx_batch)
    loss_train = loss_train / len(train)

    # Validation
    model.eval()
    loss_val, output_val = 0, []
    for triple_batch, h_r_batch, idx_batch in val_loader:
        output = model(triple_batch.to(device), h_r_batch.to(device)).cpu()
        loss = F.binary_cross_entropy(input=output, target=label_val[idx_batch])
        loss_val += loss.item() * len(idx_batch)
        output_val.append(output)
    loss_val = loss_val / len(val)
    output_val = torch.cat(output_val)

    print('Epoch {0:04d} | time = {1:.2f}s | Loss = [train: {2:.4f}, val: {3:.4f}]'
          .format(epoch, time.time() - t, loss_train, loss_val))

    if epoch % args.evaluation == 0:
        t1 = time.time()

        rank_val = rank_filter(output_val, data.filter_val, data.label_val, data.index_val)

        print('====================Evaluation on Epoch {0:04d}==================='.format(epoch))
        print('MRR = {0:.4f} | MR = {1:.4f}'.format(rank_val.pow(-1).mean().item(), rank_val.mean().item()))
        print('TOPN = [1: {0:.4f}, 3: {1:.4f}, 10: {2:.4f}]'
              .format(topNhit(rank_val, 1), topNhit(rank_val, 3), topNhit(rank_val, 10)))
        print('=============Evaluation completed using time: {0:.2f}s============'.format(time.time() - t1))   

        # Early stop
        early_stop(rank_val.pow(-1).mean().item(), model)
        if early_stop.early_stop:
            print('Early stop triggered at epoch {0}!'.format(epoch - args.evaluation * args.patience))
            break

"""
===========================================================================
Testing
===========================================================================
"""
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()
loss_test, output_test = 0, []
for triple_batch, h_r_batch, idx_batch in test_loader:
    output = model(triple_batch.to(device), h_r_batch.to(device)).cpu()
    loss = F.binary_cross_entropy(input=output, target=label_val[idx_batch])
    loss_test += loss.item() * len(idx_batch)
    output_test.append(output)
loss_test = loss_train / len(test)
output_test = torch.cat(output_test)

rank_test = rank_filter(output, data.filter_test, data.label_test, data.index_test)
print('============================Testing============================')
print('Loss = {0:.4f}'.format(loss_test))
print('MRR = {0:.4f} | MR = {1:.4f}'.format(rank_test.pow(-1).mean().item(), rank_test.mean().item()))
print('TOPN = [1: {0:.4f}, 3: {1:.4f}, 10: {2:.4f}]'
        .format(topNhit(rank_test, 1), topNhit(rank_test, 3), topNhit(rank_test, 10)))
