# GINN
## Developing completed (Prototype)
- [x] GAT
- [x] ConvKB-Attention
- [x] Merge
- [x] Testing
- [x] Code Review


## Requirement
- Require `pytorch>=1.6.0` for CPU (torch.sparse.softmax)
- Require `pytorch>=1.7.0` for GPU (CUDA torch.sparse.softmax)

## History
### Version 3.1 2020-11-30
- fix bugs in val and test dataset and dataloader
- add kinship dataset
- add k-hop option for dataset
- val and test running on cpu

### Version 3.0 2020-11-20
- enable batch training
- enable batch to val and test also
- add customized collate function to dataloader for different length batch input

### Version 2.1 2020-11-16
- modify filter to filter all other triples
- remove for loop in load data and clean up
- add None to attention type
- change ranking data from scipy to torch
- add reverse function
  
### Version 2.0 2020-10-14
- add GAT for models.py
- add ConvE and separate DistMult for score_function.py
- add early stopping
- ConvAttenion code polished
- add GPU support

### Version 1.1 2020-09-14
- add label smoothing
- add filter for model results
- change name to GINN
- ranking part code polished

### Version 1.0 2020-08-31
- change loss to BCEloss as ConvE (1-N scoring)
- remove negative sampling
- add new ops_new dataset for extensive test triple

### Version 0.3 2020-08-17
- add WN18RR dataset
- change measurement from ACC to MRR
- clean layers.py and parameterized

### Version 0.2 2020-08-12
- create parser.py
- add negative sampling
- create utils.py

### Version 0.1 2020-08-05
- initial commit
- add README

  


