# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific codebase for estimating damping parameters from oscillatory time series using topological data analysis (persistent homology) and machine learning methods. Implements methods from the publication "Damping parameter estimation using topological signal processing."

Supports three damping models:
- **Viscous**: Exponential amplitude decay (`F_d = 2ζθ̇`)
- **Coulomb**: Linear amplitude decay (`F_d = μ_c·sign(θ̇)`)
- **Quadratic**: Velocity-squared decay (`F_d = μ_q·θ̇|θ̇|`)

## Project Structure

```
├── experimental/           # Real pendulum data analysis
│   ├── data/              # Raw data files (60°-120°)
│   ├── scripts/           # Analysis scripts (21 methods)
│   ├── figures/           # Result visualizations
│   └── reports/           # LaTeX reports & PDFs
│
├── numerical/             # Simulation-based analysis
│   ├── simulation/        # Pendulum simulation code
│   ├── methods/           # Estimation methods
│   │   ├── classical/     # Least squares, optimization
│   │   └── ml/            # SINDy, PINNs, Neural ODEs, RNN, etc.
│   ├── convergence/       # Convergence analysis
│   └── reports/           # Research papers
│
├── notebooks/             # Jupyter notebooks
├── matlab/                # MATLAB code
└── references/            # Reference papers
```

## Running the Code

```bash
# Experimental analysis (all 21 methods)
python experimental/scripts/all_methods_complete.py

# Numerical simulation
python numerical/simulation/nonlinear_pendulum_inverse.py

# Jupyter notebook
jupyter notebook notebooks/code_with_example.ipynb

# MATLAB simulation
matlab -r "run('matlab/Run_Compare_Damping.m')"
```

## Dependencies

Python: numpy, scipy, matplotlib, torch
Optional: pysindy (for SINDy-based estimation)

## Key Results

### Experimental (80° Pendulum)
- 21 methods (11 classical + 10 ML) all achieve < 0.1% error
- Estimated damping ratio: ζ = 0.00875
- Quality factor: Q ≈ 57

### Methods Implemented

**Classical (1-11):** OLS, polyfit, Normal Equations, QR, SVD, Gradient Descent, L-BFGS-B, Differential Evolution, curve_fit, least_squares, Weighted Regression

**Machine Learning (12-21):** SINDy, PINNs, Neural ODE, RNN/LSTM, Symbolic Regression, Weak SINDy, Bayesian Regression, Envelope Matching, Gaussian Process, Transformer

## Key Mathematical Concepts

- **Envelope decay**: `A(t) = A₀·exp(-λt)` → `ln(A) = ln(A₀) - λt`
- **Damping ratio**: `ζ = λ/ω_n`
- **Lifetime (L)**: D - B (death minus birth in persistence diagram)
- **Cutoff (C_α)**: Separates signal from noise
- **Optimal ratio**: L_n/L_0 ≈ 0.3299 for viscous estimation

## Output Locations

- Experimental figures: `experimental/figures/`
- Experimental reports: `experimental/reports/`
- Numerical figures: `numerical/figures/`
- Numerical reports: `numerical/reports/`
