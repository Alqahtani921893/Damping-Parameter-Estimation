# Experimental Analysis Results

## 80-Degree Horizontal Pendulum Experiment

### Measured Quantities (directly from data)

| Parameter | Value | Unit |
|-----------|-------|------|
| Period | 0.288 | s |
| Frequency | 3.47 | Hz |
| Angular frequency (ω) | 21.82 | rad/s |
| Decay rate (λ) | 0.191 | 1/s |
| Initial amplitude | 32.2 | degrees |
| Decay R² | 0.94 | - |

### Physical System

| Parameter | Value | Unit |
|-----------|-------|------|
| Mass (m) | 50 | g |
| Length (L) | 100 | mm |
| Moment of inertia (I = mL²) | 5.0×10⁻⁴ | kg·m² |

### Static Stiffness Measurements

| Spring ID | Dimensions (L-b-c mm) | Stiffness (kt) |
|-----------|----------------------|----------------|
| 140-12-1.3 | 140×12×1.3 | 0.0326 Nm/rad |
| 160-12-1 | 160×12×1 | 0.0164 Nm/rad |
| 120-12-1 | 120×12×1 | 0.0242 Nm/rad |
| 120-10-1 | 120×10×1 | 0.0177 Nm/rad |

### Effective Stiffness (from experimental frequency)

```
kt_eff = I × ω² = 5.0×10⁻⁴ × (21.82)² = 0.238 Nm/rad
```

**Note:** The effective stiffness is ~9.5× higher than static measurements, suggesting:
- Different moment of inertia than I = mL²
- Additional system stiffness (pivot mechanism, pre-tension)
- Different spring configuration

### Damping Parameters

#### Simple Model (Viscous Only)
```
θ̈ + 2ζωₙθ̇ + ωₙ²θ = 0
```

| Parameter | Value | Formula |
|-----------|-------|---------|
| Damping ratio (ζ) | **0.0088** | λ/ω |
| Damping coefficient (c) | 1.91×10⁻⁴ Nm·s/rad | 2Iλ |
| Quality factor (Q) | 57 | ω/(2λ) |

#### Damping Type: **VISCOUS**
- Confirmed by exponential decay (R² = 0.94)
- Low damping ratio indicates underdamped system

### Simulation Parameters

For matching the experimental behavior, use:

```python
# Option 1: Direct parameters
omega_n = 21.82  # rad/s
zeta = 0.0088    # damping ratio

# Option 2: Physical parameters
kt = 0.238       # Nm/rad (effective stiffness)
c = 1.91e-4      # Nm·s/rad (damping coefficient)
I = 5.0e-4       # kg·m² (moment of inertia)

# Envelope decay
A(t) = A0 * exp(-0.191 * t)  # amplitude decay
```

### Key Findings

1. **Damping is primarily viscous** - exponential decay with R² = 0.94
2. **Very low damping ratio** (ζ ≈ 0.009) - highly underdamped system
3. **Effective stiffness mismatch** - dynamic measurement gives higher kt than static
4. **Quality factor Q ≈ 57** - system oscillates ~57 times before amplitude decays to 1/e

### Files Generated

- `damping_from_experiment.py` - Main analysis script
- `analysis_with_measured_stiffness.py` - Comparison with static stiffness
- `practical_summary.py` - Summary of reliable measurements
- `figures/experimental/damping_estimation_results.png` - Results visualization
