# Advanced Particle Filter Library

A NumPy/SciPy library for Bayesian filtering in state space models, implementing classical filters, particle flow filters, and differentiable particle filters.

Developed as part of the JP Morgan MLCOE TSRL Internship (2026).

## Requirements

**Core library** (CPU, all `.py` files and local notebooks):
- Python ≥ 3.9
- NumPy, SciPy, Matplotlib

**Differentiable PF notebooks** (GPU, Colab):
- TensorFlow ≥ 2.12, TensorFlow Probability

Install core dependencies:
```bash
pip install numpy scipy matplotlib
```
or one can install exact dependencies via environment.yaml:
```bash
conda env create -f environment.yml
conda activate pf_pf
```
## Project Structure

```
├── models/                         # State space model definitions
│   ├── base.py                     #   StateSpaceModel dataclass
│   ├── linear_gaussian.py          #   Linear-Gaussian SSM factory
│   ├── range_bearing.py            #   Range-bearing with Student-t noise
│   ├── acoustic.py                 #   Multi-target acoustic tracking (Li & Coates 2017)
│   └── dai22_example2.py           #   Dai & Daum (2022) Example 2 model
│
├── filters/                        # Filtering algorithms
│   ├── base.py                     #   FilterResult container
│   ├── kalman.py                   #   KF, EKF, UKF
│   ├── particle.py                 #   Bootstrap particle filter (SIR)
│   ├── edh.py                      #   EDH flow + EDH PF-PF (Li & Coates 2017)
│   ├── ledh.py                     #   LEDH flow + LEDH PF-PF
│   ├── kernel_pff.py               #   Kernel PFF with scalar/matrix kernels (Hu & van Leeuwen 2021)
│   ├── stochastic_pff.py           #   Stochastic PF flow + optimal homotopy BVP (Dai & Daum 2022)
│   └── stochastic_pfpf.py          #   SPFF-PF (stochastic flow as PF-PF proposal)
│
├── simulation/                     # Data generation
│   └── trajectory.py               #   Trajectory container, simulate, simulate_batch
│
├── utils/                          # Shared utilities
│   ├── resampling.py               #   Systematic, stratified, multinomial, residual resampling
│   └── metrics.py                  #   OMAT, RMSE, position RMSE
│
├── pfresults/                      # Saved experiment outputs (.npz, .csv)
├── dpf_pretrained/                 # Saved pretrained model params (.h5)
│
│── 01_test_lgssm.ipynb             # KF on linear-Gaussian SSM
│── 02_test_nonlinear_ssm.ipynb     # EKF/UKF/BPF comparison on range-bearing model
│── 03_test_Li2017_analysis.ipynb   # Replication of Li & Coates (2017) acoustic tracking
│── 03_test_PFF_comparison.ipynb    # EDH vs LEDH vs Kernel PFF comparison
│── 03_test_kernel_pff.ipynb        # Kernel PFF: scalar vs matrix kernel, collapse analysis
│── 04_test_dai2022_example1_seed_sensitivity.ipynb
│                                   # Dai & Daum (2022) Ex.1 seed sensitivity analysis
│
│── Differentiable_Particle_Filter.ipynb      # [Colab/GPU] Soft & OT resampling, parameter learning
│── Sinkhorn_Parameter_Exploration.ipynb      # [Colab/GPU] OT resampling ε and K sweep
│── Three_Model_DPF_Comparison.ipynb          # [Colab/GPU] Three-model DPF with normalizing flows
│
├── acoustic_experiment.py          # Script: Li & Coates (2017) replication
├── filter_comparison.py            # Script: EDH/LEDH/Kernel PFF comparison experiments
├── 04_test_dai2022_example1.py     # Script: Dai & Daum (2022) Example 1
├── 04_test_dai2022_example2.py     # Script: Dai & Daum (2022) Example 2
├── 04_test_ledh_vs_spfpf.py        # Script: LEDH PF-PF vs SPFF-PF on range-bearing
├── dai22_example2_experiment.py    # Script: Dai & Daum (2022) Example 2 full experiment
│
├── test_basic.py                   # Integration tests (8-filter equivalence on LGSSM)
├── test_filters.py                 # Unit tests (KF/EKF/UKF/PF, 22 pytest cases)
└── test_kernel_pff.py              # Kernel PFF specific tests
```

## Quick Start

```python
from advanced_particle_filter.models import make_lgssm
from advanced_particle_filter.simulation import simulate
from advanced_particle_filter.filters import (
    KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter,
    BootstrapParticleFilter, EDHParticleFilter, LEDHParticleFilter,
)

import numpy as np

# Define a constant-velocity tracking model
A = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
C = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
Q = 0.1 * np.eye(4)
R = 0.5 * np.eye(2)
m0 = np.zeros(4)
P0 = np.eye(4)

model = make_lgssm(A, C, Q, R, m0, P0)

# Simulate and filter
traj = simulate(model, T=100, seed=42)
result = KalmanFilter().filter(model, traj.observations)

print(f"Mean RMSE: {result.mean_rmse(traj.states):.4f}")
```

## Design

All filters operate on a shared `StateSpaceModel` dataclass that bundles dynamics, observation functions, Jacobians, noise covariances, and precomputed Cholesky factors. Filters receive this object directly and return a uniform `FilterResult` containing means, covariances, ESS, log-likelihood, and optional particle histories. This makes it possible to swap any filter on any model without changing downstream code.

The filter hierarchy mirrors the literature: `EDHFlow` → `EDHParticleFilter` (adds importance weights), `LEDHFlow` → `LEDHParticleFilter` (per-particle linearization), `StochasticPFFlow` → `StochasticPFParticleFilter` (SDE-based flow as PF-PF proposal). `KernelPFF` supports both scalar and matrix-valued kernels via a flag.

GPU-dependent differentiable PF work (TensorFlow/TFP) lives in self-contained Colab notebooks, keeping the core library free of TF dependencies.

## Running Tests

```bash
# Integration tests (prints results, no pytest required)
python test_basic.py

# Unit tests (requires pytest)
pytest test_filters.py -v
```

## Notebooks

Local notebooks (`01_*` through `04_*`) run on CPU with the core library. They contain visualization and analysis code for the results presented in the report.

Colab notebooks (`Differentiable_Particle_Filter.ipynb`, `Sinkhorn_Parameter_Exploration.ipynb`, `Three_Model_DPF_Comparison.ipynb`) require a GPU runtime and install TensorFlow Probability at the top of the notebook. They are self-contained and do not import from the core library.
