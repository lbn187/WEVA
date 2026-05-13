# WEVA: Warm-up EV Abstraction

Code for reproducing the experiments in *"Effective, Efficient, and General Information Abstraction for Imperfect-Information Extensive-Form Games"*.

## Overview

WEVA (Warm-up EV Abstraction) uses a small number of CFR warm-up iterations to extract per-hand expected value (EV) features, then applies k-means clustering to produce information abstractions. The method requires no domain knowledge, no pre-training, and works across different game types.

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `open_spiel`

## Run experiments

python run_experiments.py
