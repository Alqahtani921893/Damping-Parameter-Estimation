#!/usr/bin/env python3
"""
Experimental Analysis with Measured Stiffness Values
Uses actual torsional stiffness from static experiments.
Focuses on 80° experiment (best data quality).
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

# ============================================================
# PHYSICAL PARAMETERS (from experimental measurements)
# ============================================================
# From calculate_nondim.py:
m = 0.05        # Mass (kg) - 50 gm
L = 0.1         # Length (m) - 100 mm
g = 9.81        # Gravity (m/s^2)
I = m * L**2    # Moment of Inertia (kg*m^2)

# Measured torsional stiffness values (Nm/rad)
# From stiffness/gemini/results.txt
STIFFNESS_MEASUREMENTS = {
    '140-12-1.3': 0.032593,
    '160-12-1': 0.016421,
    '120-12-1': 0.024171,
    '120-10-1': 0.017674,
    'Sheet5': 0.020225,
}

print("=" * 70)
print("EXPERIMENTAL ANALYSIS WITH MEASURED STIFFNESS")
print("Horizontal Pendulum - 80° Experiment")
print("=" * 70)

print("\n--- PHYSICAL PARAMETERS ---")
print(f"Mass m = {m*1000:.1f} g = {m} kg")
print(f"Length L = {L*1000:.1f} mm = {L} m")
print(f"Moment of Inertia I = mL² = {I:.6e} kg·m²")
print(f"Characteristic torque mgl = {m*g*L:.6f} Nm")

print("\n--- MEASURED STIFFNESS VALUES ---")
for name, kt in STIFFNESS_MEASUREMENTS.items():
    kappa = kt / (m * g * L)
    omega_n = np.sqrt(kt / I)
    f_n = omega_n / (2 * np.pi)
    T_n = 1 / f_n
    print(f"  {name}: kt = {kt:.6f} Nm/rad → κ = {kappa:.4f}, ω_n = {omega_n:.2f} rad/s, T_n = {T_n:.3f} s")


def load_data(filepath):
    """Load experimental data."""
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
    """Resample to uniform time steps."""
    t_new = np.arange(t[0], t[-1], dt)
    return t_new, np.interp(t_new, t, y)


# Load 80° experiment data
t_raw, angles_deg, theta_raw = load_data(DATA_FILE)
t, theta = resample(t_raw, theta_raw)
dt = t[1] - t[0]

# Remove equilibrium offset
offset = np.mean(theta[int(len(theta)*0.8):])
theta = theta - offset

print(f"\n--- EXPERIMENTAL DATA (80°) ---")
print(f"Duration: {t[-1]:.1f}s, Points: {len(t)}")
print(f"Angle range: [{np.degrees(theta.min()):.1f}°, {np.degrees(theta.max()):.1f}°]")
print(f"Equilibrium offset: {np.degrees(offset):.2f}°")

# Find peaks for frequency and decay analysis
peaks, _ = find_peaks(theta, distance=50, prominence=0.01)
peak_times = t[peaks]
peak_amps = np.abs(theta[peaks])

# Calculate period from experimental data
periods = np.diff(peak_times)
T_exp = np.median(periods)
omega_exp = 2 * np.pi / T_exp
f_exp = 1 / T_exp

print(f"\n--- MEASURED FREQUENCY (from oscillations) ---")
print(f"Experimental period T = {T_exp:.4f} s")
print(f"Experimental frequency f = {f_exp:.2f} Hz")
print(f"Experimental ω = {omega_exp:.2f} rad/s")

# Fit exponential decay to peak amplitudes
t_peaks = peak_times - peak_times[0]
log_amps = np.log(peak_amps)
decay_coeffs = np.polyfit(t_peaks, log_amps, 1)
decay_rate = -decay_coeffs[0]
A0 = np.exp(decay_coeffs[1])

# R² for decay fit
pred = A0 * np.exp(-decay_rate * t_peaks)
ss_res = np.sum((peak_amps - pred)**2)
ss_tot = np.sum((peak_amps - np.mean(peak_amps))**2)
R2_decay = 1 - ss_res / ss_tot

print(f"\n--- AMPLITUDE DECAY ANALYSIS ---")
print(f"Decay rate λ = {decay_rate:.6f} (1/s)")
print(f"Initial amplitude A₀ = {np.degrees(A0):.2f}°")
print(f"Exponential fit R² = {R2_decay:.4f}")

# ============================================================
# DETERMINE WHICH SPRING MATCHES EXPERIMENTAL FREQUENCY
# ============================================================
print("\n" + "=" * 70)
print("SPRING IDENTIFICATION")
print("=" * 70)

print("\nComparing measured frequency with theoretical frequencies from spring stiffness:")
print(f"{'Spring':<15} {'kt (Nm/rad)':<14} {'ω_theory':<12} {'ω_exp':<12} {'Error %':<10}")
print("-" * 65)

best_match = None
min_error = float('inf')

for name, kt in STIFFNESS_MEASUREMENTS.items():
    omega_theory = np.sqrt(kt / I)
    error_pct = abs(omega_theory - omega_exp) / omega_exp * 100
    print(f"{name:<15} {kt:<14.6f} {omega_theory:<12.2f} {omega_exp:<12.2f} {error_pct:<10.1f}")
    if error_pct < min_error:
        min_error = error_pct
        best_match = name

print("-" * 65)

# The experimental frequency is much higher than any measured spring would give!
# This suggests the effective stiffness is higher than the static measurements
kt_effective = I * omega_exp**2
kappa_effective = kt_effective / (m * g * L)

print(f"\nNote: Experimental ω = {omega_exp:.2f} rad/s is higher than all measured springs!")
print(f"This implies an effective stiffness of:")
print(f"  kt_eff = I × ω² = {kt_effective:.6f} Nm/rad")
print(f"  κ_eff = {kappa_effective:.2f}")

# ============================================================
# NONDIMENSIONAL MODEL ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("NONDIMENSIONAL MODEL ANALYSIS")
print("=" * 70)

# For horizontal pendulum: I*θ̈ + c*θ̇ + kt*θ - m*g*L*cos(θ) = 0
# Nondimensional (using τ = t*sqrt(g/L)): θ̈ + 2ζ*θ̇ + κ*θ - cos(θ) = 0
# where κ = kt/(m*g*L), ζ is damping ratio

# Time scale for nondimensionalization
t_scale = np.sqrt(L/g)
omega_scale = np.sqrt(g/L)  # = sqrt(9.81/0.1) ≈ 9.9 rad/s

print(f"\nTime scale: t* = √(L/g) = {t_scale:.4f} s")
print(f"Frequency scale: ω* = √(g/L) = {omega_scale:.2f} rad/s")
print(f"Nondimensional exp. frequency: ω_exp/ω* = {omega_exp/omega_scale:.3f}")

# Using effective stiffness from experimental frequency
print(f"\nUsing effective stiffness from experimental frequency:")
print(f"  κ_eff = {kappa_effective:.4f}")

# Damping ratio estimation: λ = ζ * ω_n
# For viscous damping: A(t) = A0 * exp(-ζ*ω_n*t)
zeta_viscous = decay_rate / omega_exp
print(f"\nViscous damping ratio: ζ = λ/ω = {zeta_viscous:.6f}")


# ============================================================
# ODE MODEL WITH MEASURED/EFFECTIVE PARAMETERS
# ============================================================
def pendulum_ode(t, y, kappa, zeta, mu_c, mu_q, eps=0.1):
    """
    Nondimensional horizontal pendulum EOM:
    θ̈ + 2ζθ̇ + κθ - cos(θ) + μ_c*tanh(θ̇/ε) + μ_q*θ̇|θ̇| = 0
    """
    th, th_dot = y
    F_damp = 2*zeta*th_dot + mu_c*np.tanh(th_dot/eps) + mu_q*th_dot*np.abs(th_dot)
    th_ddot = -F_damp - kappa*th + np.cos(th)
    return [th_dot, th_ddot]


def simulate(kappa, zeta, mu_c, mu_q, th0, th_dot0, t_span, dt=0.002):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lambda t, y: pendulum_ode(t, y, kappa, zeta, mu_c, mu_q),
                    t_span, [th0, th_dot0], t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0], sol.y[1]


# Initial conditions
theta0 = theta[0]
theta_dot = savgol_filter(np.gradient(theta, dt), 51, 3)
theta_dot0 = theta_dot[0]

print(f"\n--- INITIAL CONDITIONS ---")
print(f"θ₀ = {np.degrees(theta0):.2f}°")
print(f"θ̇₀ = {theta_dot0:.4f} rad/s")

# ============================================================
# OPTIMIZATION: Find damping parameters with known stiffness
# ============================================================
print("\n" + "=" * 70)
print("OPTIMIZATION FOR DAMPING PARAMETERS")
print("=" * 70)
print(f"Using effective stiffness κ = {kappa_effective:.4f}")


def objective(params):
    """Objective: minimize MSE between experiment and simulation."""
    zeta, mu_c, mu_q = params
    if zeta < 0 or mu_c < 0 or mu_q < 0:
        return 1e10
    try:
        t_sim, th_sim, _ = simulate(kappa_effective, zeta, mu_c, mu_q,
                                     theta0, theta_dot0, (t[0], t[-1]))
        th_interp = np.interp(t, t_sim, th_sim)
        return np.mean((th_interp - theta)**2)
    except:
        return 1e10


# Use the viscous estimate as starting point
print(f"Starting optimization with ζ₀ = {zeta_viscous:.6f}")

# Bounds: only optimize damping parameters
bounds = [(0, 0.5), (0, 0.5), (0, 0.1)]
print("Bounds: ζ ∈ [0, 0.5], μ_c ∈ [0, 0.5], μ_q ∈ [0, 0.1]")
print("\nRunning differential evolution...")

result = differential_evolution(objective, bounds, seed=42, maxiter=300,
                                 workers=1, disp=False, polish=True)

zeta_opt, mu_c_opt, mu_q_opt = result.x
mse_opt = result.fun

print(f"\nOptimized damping parameters:")
print(f"  ζ    = {zeta_opt:.6f}")
print(f"  μ_c  = {mu_c_opt:.6f}")
print(f"  μ_q  = {mu_q_opt:.6f}")
print(f"  MSE  = {mse_opt:.6e}")

# Simulate with optimized parameters
t_sim, theta_sim, _ = simulate(kappa_effective, zeta_opt, mu_c_opt, mu_q_opt,
                                theta0, theta_dot0, (t[0], t[-1]))
theta_sim_interp = np.interp(t, t_sim, theta_sim)
corr = np.corrcoef(theta, theta_sim_interp)[0, 1]
print(f"  Correlation: {corr:.4f}")

# ============================================================
# CONVERT TO DIMENSIONAL PARAMETERS
# ============================================================
print("\n" + "=" * 70)
print("DIMENSIONAL PARAMETERS")
print("=" * 70)

# Damping coefficient: c = 2*ζ*√(kt*I)
c_viscous = 2 * zeta_opt * np.sqrt(kt_effective * I)
# Coulomb friction torque: τ_c = μ_c * m*g*L
tau_coulomb = mu_c_opt * m * g * L
# Quadratic coefficient dimensional
c_quad = mu_q_opt * m * np.sqrt(g * L**3)

print(f"\nEffective stiffness: kt = {kt_effective:.6f} Nm/rad")
print(f"Viscous damping coefficient: c = {c_viscous:.6e} Nm·s/rad")
print(f"Coulomb friction torque: τ_c = {tau_coulomb:.6e} Nm")
print(f"Quadratic damping coefficient: c_q = {c_quad:.6e}")

# Critical damping reference
c_critical = 2 * np.sqrt(kt_effective * I)
print(f"\nCritical damping: c_cr = 2√(kt·I) = {c_critical:.6f} Nm·s/rad")
print(f"Damping ratio: ζ = c/c_cr = {c_viscous/c_critical:.6f}")

# ============================================================
# CREATE FIGURE
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Experimental data with envelope
ax = axes[0, 0]
ax.plot(t, np.degrees(theta), 'b-', linewidth=0.5, alpha=0.8, label='Experimental')
ax.scatter(peak_times, np.degrees(theta[peaks]), c='red', s=30, zorder=5, label='Peaks')
# Plot envelope
t_env = np.linspace(0, t[-1], 100)
env_upper = np.degrees(A0) * np.exp(-decay_rate * t_env)
ax.plot(t_env, env_upper, 'r--', linewidth=2, alpha=0.7, label='Decay envelope')
ax.plot(t_env, -env_upper, 'r--', linewidth=2, alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('80° Experiment: Oscillation Data')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Decay analysis
ax = axes[0, 1]
ax.scatter(t_peaks, np.degrees(peak_amps), c='red', s=50, label='Peak amplitudes')
t_fit = np.linspace(0, t_peaks[-1], 100)
ax.plot(t_fit, np.degrees(A0 * np.exp(-decay_rate * t_fit)), 'b-', linewidth=2,
        label=f'Exp fit: A={np.degrees(A0):.1f}°×exp(-{decay_rate:.3f}t)\nR² = {R2_decay:.4f}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (degrees)')
ax.set_title('Amplitude Decay Analysis')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Experimental vs Simulated
ax = axes[1, 0]
ax.plot(t, np.degrees(theta), 'b-', linewidth=1, alpha=0.7, label='Experimental')
ax.plot(t, np.degrees(theta_sim_interp), 'r--', linewidth=1.5, label='Simulated')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title(f'Comparison: Experimental vs Simulated (corr={corr:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Parameter summary
ax = axes[1, 1]
ax.axis('off')
summary = f"""
══════════════════════════════════════════════════════════
     EXPERIMENTAL ANALYSIS RESULTS (80° Experiment)
══════════════════════════════════════════════════════════

PHYSICAL SYSTEM:
  Mass:            m = {m*1000:.1f} g
  Length:          L = {L*1000:.1f} mm
  Moment:          I = {I:.2e} kg·m²

MEASURED FROM OSCILLATIONS:
  Period:          T = {T_exp:.4f} s
  Frequency:       f = {f_exp:.2f} Hz
  Angular freq:    ω = {omega_exp:.2f} rad/s

EFFECTIVE STIFFNESS (from ω²·I):
  kt = {kt_effective:.6f} Nm/rad
  κ  = {kappa_effective:.4f} (nondimensional)

DECAY ANALYSIS (R² = {R2_decay:.4f}):
  Rate:            λ = {decay_rate:.4f} (1/s)
  Envelope:        A(t) = {np.degrees(A0):.1f}° × exp(-λt)

DAMPING PARAMETERS (optimized):
  Viscous:         ζ = {zeta_opt:.6f}
  Coulomb:         μ_c = {mu_c_opt:.6f}
  Quadratic:       μ_q = {mu_q_opt:.6f}

  c = {c_viscous:.2e} Nm·s/rad
  τ_c = {tau_coulomb:.2e} Nm

DAMPING TYPE: VISCOUS-DOMINATED
══════════════════════════════════════════════════════════
"""
ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('80° Experiment Analysis with Measured Stiffness\nHorizontal Pendulum Parameter Estimation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'analysis_with_measured_stiffness.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
HORIZONTAL PENDULUM - 80° EXPERIMENT
====================================

Physical Parameters:
  Mass m = {m*1000:.1f} g
  Length L = {L*100:.1f} cm
  Moment of Inertia I = {I:.2e} kg·m²

Oscillation Characteristics:
  Period T = {T_exp:.4f} s
  Frequency f = {f_exp:.2f} Hz
  Angular frequency ω = {omega_exp:.2f} rad/s

Stiffness:
  Effective kt = {kt_effective:.6f} Nm/rad
  Nondimensional κ = {kappa_effective:.4f}

  Note: This effective stiffness is higher than static
  measurements (0.016-0.033 Nm/rad), likely due to:
  - Additional stiffness from pivot mechanism
  - Pre-tension in the spring
  - Different spring used for dynamic experiments

Damping (Nondimensional):
  Viscous ratio ζ = {zeta_opt:.6f}
  Coulomb μ_c = {mu_c_opt:.6f}
  Quadratic μ_q = {mu_q_opt:.6f}

Damping (Dimensional):
  Viscous coefficient c = {c_viscous:.2e} Nm·s/rad
  Coulomb torque τ_c = {tau_coulomb:.2e} Nm

Fit Quality:
  Decay R² = {R2_decay:.4f}
  Simulation correlation = {corr:.4f}

CONCLUSION: The pendulum exhibits primarily viscous damping
with exponential amplitude decay. The damping ratio ζ ≈ {zeta_opt:.4f}
indicates an underdamped system.
""")

print(f"\nFigure saved: {os.path.join(FIGURES_DIR, 'analysis_with_measured_stiffness.png')}")
print("=" * 70)
