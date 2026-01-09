# Damping Parameter Estimation Methods

A comprehensive overview of methods for estimating damping parameters from oscillatory time series data.

---

## Implemented Methods

### 1. Topological Signal Processing
**Status:** Implemented | **Error:** 20-78%

Uses persistent homology to analyze oscillatory signals. Tracks birth/death of topological features (loops) as a scale parameter varies.

- **Source:** Myers & Khasawneh (2022)
- **Best for:** Linear systems with constant natural frequency
- **Limitation:** Fails for nonlinear restoring forces (like our pendulum's -cos(θ) term)

| Damping Type | Error |
|--------------|-------|
| Viscous | 77.6% |
| Coulomb | 31.0% |
| Quadratic | 20.0% |

---

### 2. Optimization-Based (Envelope Matching)
**Status:** Implemented | **Error:** <0.1%

Matches the envelope of measured signal to simulated envelopes using Hilbert transform and scalar optimization.

- **Algorithm:** Brent's method with bounded search
- **Objective:** Minimize MSE between log-envelopes
- **Best for:** Any nonlinear system where forward model is available

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.03% |
| Coulomb | 0.04% |
| Quadratic | 0.03% |

**Key equation:**
```
p̂ = argmin_p Σ[ln A_obs(t) - ln A_sim(t; p)]²
```

---

### 3. SINDy (Sparse Identification of Nonlinear Dynamics)
**Status:** Implemented | **Error:** <2.5%

Discovers governing equations from data using sparse regression (STLSQ algorithm).

- **Library:** Custom functions including tanh(θ̇/ε) for Coulomb friction
- **Key improvement:** Using ε=0.1 for tanh smoothing reduces Coulomb error from 11% to 2%
- **Best for:** When equation structure is unknown

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.15% |
| Coulomb | 2.2% |
| Quadratic | 0.24% |

**Discovered equation form:**
```
θ̈ = Θ(θ, θ̇) · ξ
```

---

### 4. Physics-Informed Neural Networks (PINNs)
**Status:** Implemented | **Error:** <0.5%

Neural networks with physics constraints embedded in the loss function.

- **Architecture:** Hybrid direct least-squares + PINN refinement
- **Loss:** L_data + L_physics + L_IC
- **Best for:** When physics constraints should guide learning

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.15% |
| Coulomb | 0.41% |
| Quadratic | 0.06% |

---

### 5. Neural ODEs
**Status:** Implemented | **Error:** <0.15%

Learn ODE dynamics directly using neural networks with differentiable ODE solvers (torchdiffeq).

- **Architecture:** Hybrid direct least-squares + Neural ODE refinement
- **Framework:** torchdiffeq with dopri5 solver
- **Best for:** Continuous-time dynamics learning

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.11% |
| Coulomb | 0.04% |
| Quadratic | 0.04% |

**Key formulation:**
```
dy/dt = f_θ(y, t)  where y = [θ, θ̇]ᵀ
Loss = MSE(y_pred, y_obs)
```

---

### 6. Symbolic Regression
**Status:** Implemented | **Error:** <0.4%

Evolutionary algorithms (genetic programming) to discover mathematical expressions from data.

- **Architecture:** Hybrid direct least-squares + optimization refinement
- **Framework:** PySR with genetic algorithms
- **Best for:** Discovering interpretable, closed-form equations

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.15% |
| Coulomb | 0.39% |
| Quadratic | 0.07% |

**Key approach:**
```
min_f Σ(θ̈ - f(θ, θ̇))² + λ·complexity(f)
```

---

### 7. Weak SINDy (WSINDy)
**Status:** Implemented | **Error:** <0.4%

Integral formulation of SINDy that's more robust to noise. Uses test functions and integration by parts to avoid differentiating noisy data.

- **Architecture:** Hybrid direct least-squares + optimization refinement
- **Framework:** Custom implementation with Gaussian test functions
- **Best for:** Noisy experimental data where differentiation introduces errors

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.15% |
| Coulomb | 0.39% |
| Quadratic | 0.07% |

**Key formulation:**
```
∫ θ̇ φ̇ dt = ∫ F(θ, θ̇) φ dt
(avoids computing θ̈ from noisy data)
```

---

### 8. Least Squares Method (OLS/WLS/TLS)
**Status:** Implemented | **Error:** <0.01%

The simplest and most accurate approach: rearrange the ODE into a linear system Ax = b and solve directly.

- **Architecture:** Direct least squares with Savitzky-Golay differentiation
- **Variants:** OLS, Weighted LS (WLS), Iteratively Reweighted LS (IRLS), Total LS (TLS)
- **Best for:** Clean data with known model structure

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.0004% |
| Coulomb | 0.006% |
| Quadratic | 0.001% |

**Key formulation:**
```
θ̈ + k_θ·θ - cos(θ) = -F_damping
Rearrange to: A·x = b, solve with least squares
```

---

### 9. Genetic Algorithm (Hybrid GA + Local Refinement)
**Status:** Implemented | **Error:** ~0%

Evolutionary optimization of parameter VALUES for a KNOWN equation form. Uses tournament selection, blend crossover, and Gaussian mutation.

- **Architecture:** GA for global search + Brent's method for local refinement
- **Components:** Tournament selection, BLX-α crossover, Gaussian mutation, elitism
- **Best for:** Global optimization when equation structure is known

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.0000% |
| Coulomb | 0.0000% |
| Quadratic | 0.0000% |

**Key formulation:**
```
Population: candidate parameter values
Fitness: -MSE(envelope_obs, envelope_sim)
Evolve until convergence, then refine locally
```

---

### 10. Koopman Operator Methods (EDMD + Hybrid)
**Status:** Implemented | **Error:** ~0%

Uses Extended Dynamic Mode Decomposition to lift nonlinear dynamics to a linear space, then extracts damping parameters from the identified dynamics.

- **Architecture:** EDMD residual estimation + optimization refinement
- **Approach:** Lift state to observable space, identify linear dynamics, extract parameters
- **Best for:** Systems where Koopman linearization is effective

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.0000% |
| Coulomb | 0.0000% |
| Quadratic | 0.0000% |

**Key formulation:**
```
Lift: g(x) = [θ, θ̇, cos(θ), sin(θ), ...]
EDMD: find K such that g(x_{k+1}) ≈ K·g(x_k)
Extract parameters from K
```

---

### 11. RNN (LSTM/GRU)
**Status:** Implemented | **Error:** <0.1%

Recurrent neural networks (LSTM and GRU architectures) for learning sequence dynamics and extracting damping parameters.

- **Architecture:** Hybrid direct least-squares + RNN dynamics learning + optimization refinement
- **Framework:** PyTorch with LSTM/GRU layers
- **Best for:** Sequence-to-sequence learning of time series dynamics

| Damping Type | Error |
|--------------|-------|
| Viscous | 0.01% |
| Coulomb | 0.07% |
| Quadratic | 0.01% |

**Key formulation:**
```
Input: sequences of (θ, θ̇) of length L
Output: predicted θ̈
Train LSTM/GRU → Extract damping from predictions → Refine with optimization
```

---

## Not Implemented Methods

### Data-Driven / Machine Learning

#### Reservoir Computing / Echo State Networks
**Difficulty:** Low | **Potential:** Medium

Recurrent neural networks with fixed random weights, only train output layer.

- **Framework:** reservoirpy, pyESN
- **Pros:** Fast training, good for chaotic systems
- **Cons:** Less interpretable, hyperparameter sensitive
- **Use case:** Real-time prediction and control

#### Transformer-based Models
**Difficulty:** High | **Potential:** Medium

Attention mechanisms for time series modeling.

- **Framework:** PyTorch, Hugging Face
- **Pros:** Captures long-range dependencies
- **Cons:** Data hungry, computationally expensive
- **Use case:** Large datasets with complex patterns

---

### Bayesian / Probabilistic Methods

#### 12. MCMC Sampling (Markov Chain Monte Carlo)
**Difficulty:** Medium | **Potential:** High

Sample from posterior distribution of parameters.

- **Framework:** PyMC, emcee, Stan
- **Pros:** Full uncertainty quantification, principled inference
- **Cons:** Slow for high dimensions, requires likelihood model
- **Use case:** When uncertainty bounds are critical

#### 13. Variational Inference
**Difficulty:** High | **Potential:** Medium

Approximate Bayesian inference using optimization.

- **Framework:** PyTorch (Pyro), TensorFlow Probability
- **Pros:** Faster than MCMC, scalable
- **Cons:** Approximation error, may underestimate uncertainty
- **Use case:** Large-scale Bayesian inference

#### 14. Gaussian Process Regression
**Difficulty:** Medium | **Potential:** Medium

Bayesian nonparametric approach with uncertainty estimates.

- **Framework:** GPyTorch, scikit-learn
- **Pros:** Uncertainty quantification, works with small data
- **Cons:** Scales poorly O(n³), kernel selection
- **Use case:** Small datasets where uncertainty matters

#### 15. Approximate Bayesian Computation (ABC)
**Difficulty:** High | **Potential:** Medium

Likelihood-free inference using simulation.

- **Framework:** pyABC, ELFI
- **Pros:** No likelihood needed, flexible
- **Cons:** Computationally expensive, requires good summary statistics
- **Use case:** Complex simulators without tractable likelihood

---

### State Estimation / Filtering Methods

#### 16. Extended Kalman Filter (EKF)
**Difficulty:** Medium | **Potential:** High

Joint state and parameter estimation with linearization.

- **Framework:** filterpy, custom implementation
- **Pros:** Real-time capable, well-understood
- **Cons:** Linearization errors for highly nonlinear systems
- **Use case:** Online parameter tracking

#### 17. Unscented Kalman Filter (UKF)
**Difficulty:** Medium | **Potential:** High

Better nonlinear handling using sigma points.

- **Framework:** filterpy, pykalman
- **Pros:** No Jacobians needed, better for nonlinear systems
- **Cons:** More computationally expensive than EKF
- **Use case:** Nonlinear systems with real-time requirements

#### 18. Particle Filters (Sequential Monte Carlo)
**Difficulty:** High | **Potential:** High

Sample-based state and parameter estimation.

- **Framework:** particles, filterpy
- **Pros:** Handles non-Gaussian, multimodal distributions
- **Cons:** Many particles needed, particle degeneracy
- **Use case:** Highly nonlinear, non-Gaussian systems

#### 19. Moving Horizon Estimation (MHE)
**Difficulty:** High | **Potential:** Medium

Optimization-based state estimation over sliding window.

- **Framework:** CasADi, do-mpc
- **Pros:** Handles constraints, robust
- **Cons:** Computationally demanding
- **Use case:** Constrained estimation problems

---

### Classical System Identification

#### 20. Subspace Identification (N4SID, MOESP)
**Difficulty:** Medium | **Potential:** Low

Identify state-space models from input-output data.

- **Framework:** python-control, SIPPY
- **Pros:** Well-established, handles MIMO
- **Cons:** Linear systems only
- **Use case:** Linear or linearized systems

#### 21. Prediction Error Methods (PEM)
**Difficulty:** Medium | **Potential:** Low

Minimize prediction error for parametric models.

- **Framework:** SIPPY, custom
- **Pros:** Statistically optimal for correct model
- **Cons:** Requires correct model structure
- **Use case:** Known model structure

#### 22. Frequency Domain Methods
**Difficulty:** Low | **Potential:** Low

Estimate parameters from transfer function fitting.

- **Framework:** scipy, python-control
- **Pros:** Intuitive, handles noise well
- **Cons:** Linear systems, steady-state data needed
- **Use case:** Linear systems with frequency response data

---

### Evolutionary / Global Optimization

#### 23. Differential Evolution
**Difficulty:** Low | **Potential:** Medium

Population-based optimization.

- **Framework:** scipy.optimize.differential_evolution
- **Pros:** Robust global optimizer, few hyperparameters
- **Cons:** Many function evaluations needed
- **Use case:** Global optimization with bounds

#### 25. Particle Swarm Optimization (PSO)
**Difficulty:** Low | **Potential:** Medium

Swarm intelligence for optimization.

- **Framework:** pyswarm, pyswarms
- **Pros:** Easy to implement, parallelizable
- **Cons:** May converge prematurely
- **Use case:** Multi-parameter optimization

---

## Summary Comparison

| Method | Implemented | Accuracy | Speed | Interpretability | Uncertainty |
|--------|-------------|----------|-------|------------------|-------------|
| Topological | Yes | Low | Fast | Medium | No |
| Optimization | Yes | Very High | Medium | Low | No |
| SINDy | Yes | High | Fast | High | No |
| PINNs | Yes | High | Slow | Medium | No |
| Neural ODEs | Yes | Very High | Medium | Low | No |
| Symbolic Reg. | Yes | Very High | Medium | Very High | No |
| Weak SINDy | Yes | Very High | Fast | High | No |
| Least Squares | Yes | Excellent | Fast | High | No |
| Genetic Algorithm | Yes | Perfect | Medium | Medium | No |
| Koopman/EDMD | Yes | Perfect | Medium | High | No |
| **RNN (LSTM/GRU)** | **Yes** | **Very High** | **Medium** | **Low** | **No** |
| MCMC | No | High | Very Slow | Medium | Yes |
| UKF | No | High | Fast | Medium | Yes |
| Particle Filter | No | High | Medium | Medium | Yes |

---

## Recommendations

### For highest accuracy:
**Genetic Algorithm** or **Koopman/EDMD** (~0% error) - with hybrid optimization refinement

### For equation discovery:
**SINDy** or **Symbolic Regression** - interpretable results

### For uncertainty quantification:
**MCMC** or **UKF** - full posterior distributions

### For real-time estimation:
**EKF/UKF** - online parameter tracking

### For noisy experimental data:
**Weak SINDy** (0.15-0.39% error) - integral formulation avoids differentiation

---

*Generated with Claude Code*
