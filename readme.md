# STAL V2 -> LAST

Experiments to improve the STAL encoder, a learnable spike train encoder for Spiking Neural Networks (SNNs).

Rebranding to Learnable Adaptive Spike Thresholds: **LAST**!

Ideas:
- Test LAST on benchmark datasets, smartly using spatiotemporal dimensions
- Experiment with Z1 Z2 in loss or without
- Test LAST on a known good spiking architecture for a dataset
- (Make LAST a package?)

WIP:
- Experimenting with convlutional layer for combining channels instead of flattening:
```
input: omega * c -> convolution -> omega * 1 -> (optional hidden layers for feature extraction) -> map to omega * psi -> learnable thresholds
```
Name: Convolutional Adaptive Spike Thresholds: __CAST__!

Needs improvement: 
- LAST (Stacked) needs to encode spikes more in spatial and temporal dimension! Maybe tweak the loss?
