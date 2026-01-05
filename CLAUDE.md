# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific codebase for estimating damping parameters from oscillatory time series using topological data analysis (persistent homology). Implements methods from the publication "Damping parameter estimation using topological signal processing."

Supports three damping models:
- **Viscous**: Exponential amplitude decay (`F_d = 2ζθ̇`)
- **Coulomb**: Linear amplitude decay (`F_d = μ_c·sign(θ̇)`)
- **Quadratic**: Velocity-squared decay (`F_d = μ_q·θ̇|θ̇|`)

## Running the Code

```bash
# Interactive analysis (main example)
jupyter notebook Python_code/code_with_example.ipynb

# Standalone scripts
python Python_code/nonlinear_pendulum_inverse.py   # Full inverse estimation pipeline
python Python_code/optimized_estimation.py         # Optimization-based estimation
python Python_code/generate_plots.py               # Generate report figures

# MATLAB simulation
matlab -r "run('matlab/Run_Compare_Damping.m')"
```

## Dependencies

Python: numpy, scipy (signal, special, optimize, integrate), matplotlib, pandas
Optional: pysindy (for SINDy-based estimation)

## Architecture

### Core Topological Functions (`code_with_example.ipynb`, replicated in standalone scripts)

**`Persistence0D(sample_data, min_or_max, edges)`**
- Computes 0D persistence diagram using peak-valley pairing algorithm
- Returns: feature indices (birth/death) and persistence points (B, D pairs)

**`damping_constant(t, ts, damping, params, sigma, alpha, noise_comp, plotting)`**
- Main entry point orchestrating the full pipeline:
  1. `Persistence0D()` → Extract persistence features
  2. `cutoff_from_lifetimes()` → Compute significance threshold C_α
  3. `floor_from_lifetimes()` → Compute noise floor F
  4. `damping_param_estimation()` → Estimate damping via optimal ratio or curve fitting

**`damping_param_estimation(damping_type, L, B, D, T_B, T_D, floor, ...)`**
- Core estimation supporting three methods:
  - Optimal ratio method: Uses L_n/L_0 ≈ 0.3299 threshold
  - Curve fitting: BFGS minimization with `fit_two_curves()`
  - Single lifetime fallback

### Standalone Scripts

**`nonlinear_pendulum_inverse.py`**
- Full pendulum simulation with `solve_ivp()` (nonlinear EOM: θ̈ + damping + k_θ·θ - cos(θ) + excitation = 0)
- Applies topological estimation to simulated data
- Compares estimated vs true parameters

**`optimized_estimation.py`**
- Alternative approach: direct optimization matching envelope decay
- Uses Hilbert transform for envelope extraction
- `minimize_scalar()` with bounded search, `differential_evolution()` for joint estimation

### PySINDy Estimation (`Python_code/pysindy_estimation/`)

**`sindy_damping_estimation.py`**
- SINDy (Sparse Identification of Nonlinear Dynamics) approach
- Discovers governing equation from time series using sparse regression (STLSQ)
- Builds custom library: [1, θ, θ̇, cos(θ), sin(θ), θ̇|θ̇|, sign(θ̇), θ², θ·θ̇]
- Extracts damping parameters from identified coefficients
- Works with or without pysindy package (manual STLSQ fallback)

**`compare_methods.py`**
- Compares all three estimation approaches on same data:
  1. Topological (persistence homology)
  2. SINDy (sparse identification)
  3. Optimization (envelope matching)
- Includes robustness analysis across noise levels

```bash
# Run SINDy estimation
python Python_code/pysindy_estimation/sindy_damping_estimation.py

# Compare all methods
python Python_code/pysindy_estimation/compare_methods.py
```

### MATLAB Code

**`EOM_Base_Pendulum.m`**: ODE function for horizontal pendulum with multiple damping models
**`Run_Compare_Damping.m`**: Comparison simulations for all damping types

## Key Mathematical Concepts

- **Lifetime (L)**: D - B (death minus birth in persistence diagram)
- **Cutoff (C_α)**: `C_α = 2^(3/2)·σ·erfinv(2(1-√α)^(1/n) - 1)` - separates signal from noise
- **Floor (F)**: Noise compensation floor for improved accuracy
- **Optimal ratio**: L_n/L_0 ≈ 0.3299 determines optimal lifetime pair for viscous estimation
- **Viscous ζ formula**: `ζ = √(1/(1 + (2nπ/ln(L_0/L_n))²))`

## Output Locations

- Figures saved to `figures/` directory
- Reports in `reports/` (LaTeX + PDF)
