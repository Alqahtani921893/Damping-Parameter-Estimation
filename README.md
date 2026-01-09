# Damping Parameter Estimation using Topological Signal Processing

Python code for estimating damping parameters from oscillatory time series using topological data analysis and machine learning methods.

## Project Structure

```
├── experimental/           # Experimental data analysis
│   ├── data/              # Raw pendulum data (60°-120°)
│   ├── scripts/           # Analysis scripts (21 methods)
│   ├── figures/           # Result visualizations
│   ├── reports/           # LaTeX reports & PDFs
│   └── RESULTS.md         # Summary of experimental results
│
├── numerical/             # Numerical/simulation analysis
│   ├── simulation/        # Pendulum simulation code
│   ├── methods/           # Estimation methods
│   │   ├── classical/     # Least squares, optimization
│   │   └── ml/            # SINDy, PINNs, Neural ODEs, RNN, etc.
│   ├── convergence/       # Convergence analysis
│   ├── figures/           # Simulation figures
│   └── reports/           # Research papers
│
├── notebooks/             # Jupyter notebooks
├── matlab/                # MATLAB code
└── references/            # Reference papers
```

## Key Results

### Experimental Analysis (80° Pendulum)
- **21 estimation methods** (11 classical + 10 ML)
- **All methods achieve < 0.1% error**
- Estimated damping ratio: **ζ = 0.00875**
- Quality factor: Q ≈ 57

### Methods Implemented

**Classical (1-11):** OLS, polyfit, Normal Equations, QR, SVD, Gradient Descent, L-BFGS-B, Differential Evolution, curve_fit, least_squares, Weighted Regression

**Machine Learning (12-21):** SINDy, PINNs, Neural ODE, RNN/LSTM, Symbolic Regression, Weak SINDy, Bayesian Regression, Envelope Matching, Gaussian Process, Transformer

## Quick Start

```bash
# Run experimental analysis (all 21 methods)
cd experimental/scripts
python all_methods_complete.py

# Run numerical simulation
cd numerical/simulation
python nonlinear_pendulum_inverse.py
```

## Dependencies

```
numpy, scipy, matplotlib, torch
```

## References

Based on: "Damping parameter estimation using topological signal processing" (Myers & Khasawneh, 2022)
