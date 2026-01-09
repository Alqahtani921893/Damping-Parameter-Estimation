#!/usr/bin/env python3
"""
Damping Parameter Estimation from Experimental Data
===================================================
1. Calculate effective stiffness from measured oscillation frequency
2. Estimate damping parameters using the derived stiffness
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
import os
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Codes/Dec16/Damping-parameter-estimation-using-topological-signal-processing/figures/experimental'
DATA_FILE = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Experiment/80P.txt'

# Physical parameters
m = 0.05   # kg (50g)
L = 0.1    # m (100mm)
g = 9.81   # m/s²
I = m * L**2  # kg·m² (nominal, may differ from effective)


def load_data(filepath):
    times, angles = [], []
    with open(filepath, 'r') as f:
        for line in f.readlines()[2:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    times.append(float(parts[1].strip()))
                    angles.append(float(parts[2].strip()))
                except:
                    continue
    times = np.array(times) - times[0]
    angles_rad = np.radians(np.array(angles))
    return times, np.array(angles), angles_rad


def resample(t, y, dt=0.002):
    t_new = np.arange(t[0], t[-1], dt)
    return t_new, np.interp(t_new, t, y)


print("=" * 70)
print("DAMPING PARAMETER ESTIMATION FROM 80° EXPERIMENT")
print("=" * 70)

# =============================================================================
# STEP 1: Load and analyze experimental data
# =============================================================================
print("\n" + "-" * 70)
print("STEP 1: Analyze Experimental Data")
print("-" * 70)

t_raw, angles_deg, theta_raw = load_data(DATA_FILE)
t, theta = resample(t_raw, theta_raw)
dt = t[1] - t[0]

# Remove equilibrium offset
offset = np.mean(theta[int(len(theta)*0.8):])
theta = theta - offset

# Find peaks
peaks, _ = find_peaks(theta, distance=50, prominence=0.01)
peak_times = t[peaks]
peak_amps = np.abs(theta[peaks])

# Period and frequency
periods = np.diff(peak_times)
T = np.median(periods)
omega = 2 * np.pi / T
f = 1 / T

print(f"Measured oscillation:")
print(f"  Period T = {T:.4f} s")
print(f"  Frequency f = {f:.2f} Hz")
print(f"  Angular frequency ω = {omega:.2f} rad/s")

# Decay analysis
t_peaks = peak_times - peak_times[0]
log_amps = np.log(peak_amps)
decay_coeffs = np.polyfit(t_peaks, log_amps, 1)
decay_rate = -decay_coeffs[0]
A0 = np.exp(decay_coeffs[1])

# R² for decay
pred = A0 * np.exp(-decay_rate * t_peaks)
ss_res = np.sum((peak_amps - pred)**2)
ss_tot = np.sum((peak_amps - np.mean(peak_amps))**2)
R2 = 1 - ss_res / ss_tot

print(f"\nAmplitude decay:")
print(f"  Decay rate λ = {decay_rate:.4f} (1/s)")
print(f"  Initial amplitude A₀ = {np.degrees(A0):.1f}°")
print(f"  Fit R² = {R2:.4f}")

# =============================================================================
# STEP 2: Calculate effective stiffness from frequency
# =============================================================================
print("\n" + "-" * 70)
print("STEP 2: Calculate Effective Stiffness")
print("-" * 70)

# For a torsional oscillator: ω² = kt / I
# Effective stiffness: kt_eff = I × ω²
kt_eff = I * omega**2

# Nondimensional stiffness: κ = kt / (m*g*L)
kappa_eff = kt_eff / (m * g * L)

print(f"Using nominal I = mL² = {I:.4e} kg·m²")
print(f"\nEffective stiffness (from ω² = kt/I):")
print(f"  kt_eff = I × ω² = {kt_eff:.4f} Nm/rad")
print(f"  κ_eff = kt/(mgL) = {kappa_eff:.2f} (nondimensional)")

# Compare with static measurements
print(f"\nNote: Static measurements gave kt = 0.016 - 0.033 Nm/rad")
print(f"      The effective stiffness is {kt_eff/0.025:.1f}× higher")
print(f"      This could be due to different I or additional system stiffness")

# =============================================================================
# STEP 3: Estimate damping parameters
# =============================================================================
print("\n" + "-" * 70)
print("STEP 3: Estimate Damping Parameters")
print("-" * 70)

# Method 1: Direct from decay rate (viscous only)
# For viscous damping: A(t) = A0 * exp(-ζωt), so λ = ζω
zeta_direct = decay_rate / omega

print(f"\nMethod 1: Direct from decay (viscous-only model)")
print(f"  λ = ζ × ω → ζ = λ/ω = {zeta_direct:.6f}")

# Method 2: Damping coefficient
# For θ̈ + (c/I)θ̇ + (kt/I)θ = 0, decay rate = c/(2I)
c_eff = 2 * I * decay_rate

print(f"\nMethod 2: Damping coefficient")
print(f"  c = 2Iλ = {c_eff:.4e} Nm·s/rad")

# Critical damping
c_critical = 2 * np.sqrt(kt_eff * I)
print(f"  c_critical = 2√(kt·I) = {c_critical:.4e} Nm·s/rad")
print(f"  ζ = c/c_cr = {c_eff/c_critical:.6f}")

# =============================================================================
# STEP 4: Validate with simulation
# =============================================================================
print("\n" + "-" * 70)
print("STEP 4: Validate with Simulation")
print("-" * 70)


def pendulum_ode_simple(t, y, omega_n, zeta):
    """Simple linear damped oscillator: θ̈ + 2ζω_n θ̇ + ω_n² θ = 0"""
    th, th_dot = y
    th_ddot = -2*zeta*omega_n*th_dot - omega_n**2 * th
    return [th_dot, th_ddot]


def pendulum_ode_full(t, y, kappa, zeta, mu_c, mu_q, eps=0.1):
    """Full nonlinear model: θ̈ + 2ζθ̇ + κθ - cos(θ) + friction = 0"""
    th, th_dot = y
    F_damp = 2*zeta*th_dot + mu_c*np.tanh(th_dot/eps) + mu_q*th_dot*np.abs(th_dot)
    th_ddot = -F_damp - kappa*th + np.cos(th)
    return [th_dot, th_ddot]


# Initial conditions
theta0 = theta[0]
theta_dot = savgol_filter(np.gradient(theta, dt), 51, 3)
theta_dot0 = theta_dot[0]

# Simulate with simple model
t_span = (t[0], t[-1])
t_eval = np.arange(t_span[0], t_span[1], dt)

sol_simple = solve_ivp(
    lambda t, y: pendulum_ode_simple(t, y, omega, zeta_direct),
    t_span, [theta0, theta_dot0], t_eval=t_eval, method='RK45'
)
theta_simple = np.interp(t, sol_simple.t, sol_simple.y[0])
corr_simple = np.corrcoef(theta, theta_simple)[0, 1]

print(f"Simple model (linear, viscous only):")
print(f"  ω_n = {omega:.2f} rad/s, ζ = {zeta_direct:.6f}")
print(f"  Correlation with experiment: {corr_simple:.4f}")

# Optimize full model
print("\nOptimizing full nonlinear model...")


def objective(params):
    zeta, mu_c, mu_q = params
    if zeta < 0 or mu_c < 0 or mu_q < 0:
        return 1e10
    try:
        sol = solve_ivp(
            lambda t, y: pendulum_ode_full(t, y, kappa_eff, zeta, mu_c, mu_q),
            t_span, [theta0, theta_dot0], t_eval=t_eval, method='RK45'
        )
        th_sim = np.interp(t, sol.t, sol.y[0])
        return np.mean((th_sim - theta)**2)
    except:
        return 1e10


# Start near the simple estimate
bounds = [(0, 0.1), (0, 0.5), (0, 0.1)]
result = differential_evolution(objective, bounds, seed=42, maxiter=200,
                                 workers=1, disp=False, polish=True,
                                 x0=[zeta_direct, 0.01, 0.01])

zeta_opt, mu_c_opt, mu_q_opt = result.x
mse_opt = result.fun

# Simulate with optimized parameters
sol_opt = solve_ivp(
    lambda t, y: pendulum_ode_full(t, y, kappa_eff, zeta_opt, mu_c_opt, mu_q_opt),
    t_span, [theta0, theta_dot0], t_eval=t_eval, method='RK45'
)
theta_opt = np.interp(t, sol_opt.t, sol_opt.y[0])
corr_opt = np.corrcoef(theta, theta_opt)[0, 1]

print(f"\nOptimized full model:")
print(f"  ζ = {zeta_opt:.6f}")
print(f"  μ_c = {mu_c_opt:.6f}")
print(f"  μ_q = {mu_q_opt:.6f}")
print(f"  MSE = {mse_opt:.6e}")
print(f"  Correlation: {corr_opt:.4f}")

# =============================================================================
# STEP 5: Final Results
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"""
EFFECTIVE STIFFNESS (from experimental frequency):
  kt_eff = {kt_eff:.4f} Nm/rad
  κ_eff = {kappa_eff:.2f} (nondimensional)

DAMPING PARAMETERS:
  ┌─────────────────────────────────────────────────────────┐
  │  Simple model (viscous only):                           │
  │    ζ = {zeta_direct:.6f}                                      │
  │    c = {c_eff:.4e} Nm·s/rad                           │
  │    Correlation: {corr_simple:.4f}                               │
  ├─────────────────────────────────────────────────────────┤
  │  Full model (optimized):                                │
  │    ζ = {zeta_opt:.6f} (viscous)                              │
  │    μ_c = {mu_c_opt:.6f} (Coulomb)                            │
  │    μ_q = {mu_q_opt:.6f} (quadratic)                          │
  │    Correlation: {corr_opt:.4f}                               │
  └─────────────────────────────────────────────────────────┘

RECOMMENDED VALUES FOR SIMULATION:
  Use: ω_n = {omega:.2f} rad/s, ζ = {zeta_direct:.4f}

  Or equivalently:
    kt = {kt_eff:.4f} Nm/rad
    c = {c_eff:.4e} Nm·s/rad
    I = {I:.4e} kg·m²

DAMPING TYPE: Primarily VISCOUS (exponential decay, R² = {R2:.2f})
""")

# =============================================================================
# Create Figure
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Experimental data with envelope
ax = axes[0, 0]
ax.plot(t, np.degrees(theta), 'b-', linewidth=0.5, alpha=0.8, label='Experimental')
t_env = np.linspace(0, t[-1], 100)
ax.plot(t_env, np.degrees(A0 * np.exp(-decay_rate * t_env)), 'r--', lw=2, label='Envelope')
ax.plot(t_env, -np.degrees(A0 * np.exp(-decay_rate * t_env)), 'r--', lw=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title(f'80° Experiment: T = {T:.3f}s, λ = {decay_rate:.3f}/s')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Decay analysis
ax = axes[0, 1]
ax.scatter(t_peaks, np.degrees(peak_amps), c='red', s=50, label='Peaks')
ax.plot(t_peaks, np.degrees(A0 * np.exp(-decay_rate * t_peaks)), 'b-', lw=2,
        label=f'Exp fit (R² = {R2:.3f})')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (degrees)')
ax.set_title('Amplitude Decay Analysis')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Experimental vs Simulated (both models)
ax = axes[1, 0]
ax.plot(t, np.degrees(theta), 'b-', linewidth=1, alpha=0.7, label='Experimental')
ax.plot(t, np.degrees(theta_simple), 'g--', linewidth=1.5, alpha=0.8,
        label=f'Simple (corr={corr_simple:.3f})')
ax.plot(t, np.degrees(theta_opt), 'r:', linewidth=1.5,
        label=f'Full model (corr={corr_opt:.3f})')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('Comparison: Experimental vs Simulated')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Summary
ax = axes[1, 1]
ax.axis('off')
summary = f"""
════════════════════════════════════════════════════════════
          DAMPING ESTIMATION RESULTS (80° Experiment)
════════════════════════════════════════════════════════════

FROM EXPERIMENTAL DATA:
  Period:            T = {T:.4f} s
  Frequency:         ω = {omega:.2f} rad/s
  Decay rate:        λ = {decay_rate:.4f} (1/s)
  Decay R²:          {R2:.4f}

EFFECTIVE STIFFNESS:
  kt_eff = I×ω² = {kt_eff:.4f} Nm/rad
  κ_eff = {kappa_eff:.2f}

DAMPING PARAMETERS:
  ┌─────────────────────────────────────┐
  │ VISCOUS (simple model):             │
  │   ζ = {zeta_direct:.6f}                   │
  │   c = {c_eff:.2e} Nm·s/rad        │
  ├─────────────────────────────────────┤
  │ FULL MODEL (optimized):             │
  │   ζ   = {zeta_opt:.6f}                   │
  │   μ_c = {mu_c_opt:.6f}                   │
  │   μ_q = {mu_q_opt:.6f}                   │
  └─────────────────────────────────────┘

DAMPING TYPE: VISCOUS (exponential decay)
════════════════════════════════════════════════════════════
"""
ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Damping Parameter Estimation from 80° Experiment',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'damping_estimation_results.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nFigure saved: {os.path.join(FIGURES_DIR, 'damping_estimation_results.png')}")
print("=" * 70)
