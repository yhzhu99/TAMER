# TAMER: A Test-Time Adaptive MoE-Driven Framework for EHR Representation Learning

TAMER combines Mixture of Experts (MoE) with Test-Time Adaptation (TTA) to address patient population heterogeneity and distribution shifts in EHR data.

## Key Features

- Plug-in framework compatible with existing EHR feature extractors
- Mixture of Experts (MoE) module for handling diverse patient subgroups
- Test-Time Adaptation (TTA) for real-time adaptation to evolving health status distributions
- Consistent performance improvements across various EHR modeling backbones

## Datasets

TAMER has been evaluated on four real-world EHR datasets:

- MIMIC-III
- MIMIC-IV
- PhysioNet Challenge 2012
- eICU

## Tasks

The framework has been tested on two main tasks:
1. In-hospital mortality prediction
2. 30-day readmission prediction
