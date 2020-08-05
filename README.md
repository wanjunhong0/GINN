# Developing phrase
- [x] GAT
- [x] ConvKB-Attention
- [x] Merge
- [x] Testing
- [ ] Code Review


# TODO
### Test on server for more parameters
- R-GCN dim=500, now dim=8
- Install `pytorch==1.6.0` on server, `torch-1.6-CUDA10.2` == 700M, `cudatoolkit-10.2` == 700M
- Training takes days, now dim=8, head=4, need 60S/epoch(200+epoch may needed)

### Unit test
Ensure if simpler version of the model works
- R * E * E -> E * E
- ConvKB to generate attention
- more complex score function
- 0-1 classification or ranking problem

### GPU implementation
- estimate 10x faster
- limited memory for large parameters(8*16G=128G)

