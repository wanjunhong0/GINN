import time
import argparse
import colorama
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GINN.parser import parse_args
from GINN.models import GINN
from GINN.load_data import Data, Dataset
from GINN.utils import label_smoothing, rank_filter, topNhit, EarlyStopping


t_start = time.time()
# Settings
colorama.init(autoreset=True)
args = parse_args()
for arg in vars(args):
    print('{0} = {1}'.format(arg, getattr(args, arg)))
torch.manual_seed(args.seed)
device = torch.device(args.device)
torch.backends.cudnn.deterministic = True
early_stop = EarlyStopping(patience=args.patience, mode='max', path=args.model_path)
"""
===========================================================================
Loading data
===========================================================================
"""
data = Data(path=args.data_path + args.dataset, reverse=args.reverse)
train = Dataset(data.triple_train, data.neighbor_dict, data.triple_index, data.train)
train_loader = DataLoader(dataset=train, batch_size=args.batch_size, collate_fn=train.collate)
train_loader = [i for i in train_loader]
label_train = label_smoothing(data.label_train, args.label_smoothing)
label_val = label_smoothing(data.label_val, args.label_smoothing)
label_test = label_smoothing(data.label_test, args.label_smoothing)

print('Loaded dataset with {0} entities and {1} relations using {2:.1f}s'
      .format(data.n_entity, data.n_relation, time.time() - t_start))

"""
===========================================================================
Training
===========================================================================
"""
# Model and optimizer
model = GINN(n_entity=data.n_entity, n_relation=data.n_relation, dim=args.hidden, dropout=args.dropout, 
             n_head=args.head, n_channel=args.channel, kernel_size=args.kernel, 
             attention=args.attention, score_func=args.score_func, reshape_size=args.reshape_size)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(1, args.epoch+1):
    t = time.time()

    model.train()
    model.to(device)
    loss_train = 0
    for triple_hop1_batch, triple_hop2_batch, h_r_batch, idx_batch in train_loader:
        optimizer.zero_grad()
        output = model(triple_hop1_batch.to(device), triple_hop2_batch.to(device), h_r_batch.to(device)).cpu()
        loss = F.binary_cross_entropy(input=output, target=label_train[idx_batch])
        loss.backward()
        optimizer.step()
        loss_train += loss.item() * len(idx_batch)
    loss_train = loss_train / len(data.train)

    print('Epoch {0:04d} | Time = {1:.2f}s | Loss = {2:.4f}'.format(epoch, time.time() - t, loss_train))

    # Validation
    if epoch % args.evaluation == 0:
        t = time.time()

        model.eval()
        model.cpu()
        output = model(data.triple_train, data.triple_train, data.val)
        loss_val = F.binary_cross_entropy(input=output, target=label_val)
        rank_val = rank_filter(output, data.filter_val, data.label_val, data.index_val)

        print('===============Evaluation================')
        print('Epoch {0:04d} | Time = {1:.2f}s | Loss = {2:.4f}'.format(epoch, time.time() - t, loss_val))
        print(colorama.Fore.RED + 'MRR = {0:.4f} | MR = {1:.4f}'.format(rank_val.pow(-1).mean(), rank_val.mean()))
        print('TOPN = [1: {0:.4f}, 3: {1:.4f}, 10: {2:.4f}]'
              .format(topNhit(rank_val, 1), topNhit(rank_val, 3), topNhit(rank_val, 10)))
        print('=========================================')

        # Early stop
        early_stop(rank_val.pow(-1).mean(), model)
        if early_stop.early_stop:
            print('Early stop triggered at epoch {0}!'.format(epoch - args.evaluation * args.patience))
            break

"""
===========================================================================
Testing
===========================================================================
"""
t = time.time()
model.load_state_dict(torch.load(args.model_path))
output = model(data.triple_train, data.triple_train, data.test)
loss_test = F.binary_cross_entropy(input=output, target=label_test)

rank_test = rank_filter(output, data.filter_test, data.label_test, data.index_test)
print('==================Testing================')
print('Time = {0:.2f}s | Loss = {1:.4f}'.format(time.time() - t, loss_test))
print(colorama.Fore.RED + 'MRR = {0:.4f} | MR = {1:.4f}'.format(rank_test.pow(-1).mean(), rank_test.mean()))
print('TOPN = [1: {0:.4f}, 3: {1:.4f}, 10: {2:.4f}]'
      .format(topNhit(rank_test, 1), topNhit(rank_test, 3), topNhit(rank_test, 10)))
print('Finished using total {0:.1f}mins'.format((time.time() - t_start) / 60))
