#!/usr/bin/env python3
"""
Focused Analysis of 80° Experiment
Best quality data with highest R² = 0.97 for decay fit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import os
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Codes/Dec16/Damping-parameter-estimation-using-topological-signal-processing/figures/experimental'
DATA_FILE = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Experiment/80P.txt'

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

print("=" * 60)
print("80° EXPERIMENT ANALYSIS")
print("=" * 60)

# Load data
t_raw, angles_deg, theta_raw = load_data(DATA_FILE)
t, theta = resample(t_raw, theta_raw)
dt = t[1] - t[0]

# Remove equilibrium offset (use last 20% of data)
offset = np.mean(theta[int(len(theta)*0.8):])
theta = theta - offset

print(f"\nData: {len(t)} points, {t[-1]:.1f}s duration")
print(f"Angle range: [{np.degrees(theta.min()):.1f}°, {np.degrees(theta.max()):.1f}°]")
print(f"Equilibrium offset removed: {np.degrees(offset):.2f}°")

# Find peaks
peaks, _ = find_peaks(theta, distance=50, prominence=0.01)
valleys, _ = find_peaks(-theta, distance=50, prominence=0.01)

peak_times = t[peaks]
peak_amps = np.abs(theta[peaks])
valley_times = t[valleys]
valley_amps = np.abs(theta[valleys])

print(f"Detected: {len(peaks)} peaks, {len(valleys)} valleys")

# Estimate period from peak-to-peak
periods = np.diff(peak_times)
period = np.median(periods)
omega = 2 * np.pi / period
freq = 1 / period

print(f"\n--- FREQUENCY ---")
print(f"Period: {period:.4f} s")
print(f"Frequency: {freq:.4f} Hz")
print(f"Angular frequency ω: {omega:.4f} rad/s")

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
R2 = 1 - ss_res / ss_tot

print(f"\n--- DECAY ANALYSIS ---")
print(f"Decay rate λ: {decay_rate:.6f} (1/s)")
print(f"Initial amplitude A₀: {np.degrees(A0):.2f}°")
print(f"Exponential fit R²: {R2:.4f}")
print(f"Envelope: A(t) = {np.degrees(A0):.1f}° × exp(-{decay_rate:.4f}t)")

# Damping ratio from decay
zeta = decay_rate / omega
print(f"\n--- DAMPING RATIO ---")
print(f"ζ = λ/ω = {decay_rate:.4f}/{omega:.4f} = {zeta:.6f}")

# Stiffness (assuming horizontal pendulum model)
k_theta = omega**2 + 1
print(f"\n--- STIFFNESS (model: k_θ = ω² + 1) ---")
print(f"k_θ = {k_theta:.4f}")

# ODE model
def pendulum_ode(t, y, k_th, zeta, mu_c, mu_q, eps=0.1):
    th, th_dot = y
    F_damp = 2 * zeta * th_dot + mu_c * np.tanh(th_dot/eps) + mu_q * th_dot * np.abs(th_dot)
    th_ddot = -F_damp - k_th * th + np.cos(th)
    return [th_dot, th_ddot]

def simulate(k_th, zeta, mu_c, mu_q, th0, th_dot0, t_span, dt=0.002):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lambda t, y: pendulum_ode(t, y, k_th, zeta, mu_c, mu_q),
                    t_span, [th0, th_dot0], t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0], sol.y[1]

# Initial conditions
theta0 = theta[0]
theta_dot = np.gradient(theta, dt)
theta_dot = savgol_filter(theta_dot, 51, 3)
theta_dot0 = theta_dot[0]

print(f"\n--- INITIAL CONDITIONS ---")
print(f"θ₀ = {np.degrees(theta0):.2f}°")
print(f"θ̇₀ = {np.degrees(theta_dot0):.2f}°/s")

# Optimization to find best parameters
print(f"\n--- OPTIMIZATION ---")

def objective(params):
    k_th, zeta, mu_c, mu_q = params
    if k_th < 1 or zeta < 0 or mu_c < 0 or mu_q < 0:
        return 1e10
    try:
        t_sim, th_sim, _ = simulate(k_th, zeta, mu_c, mu_q, theta0, theta_dot0, (t[0], t[-1]))
        th_interp = np.interp(t, t_sim, th_sim)
        return np.mean((th_interp - theta)**2)
    except:
        return 1e10

# Use differential evolution for global optimization
bounds = [(k_theta*0.5, k_theta*1.5), (0, 0.5), (0, 0.5), (0, 0.1)]
print("Running global optimization (differential evolution)...")

result = differential_evolution(objective, bounds, seed=42, maxiter=200,
                                 workers=1, disp=False, polish=True)

k_opt, zeta_opt, mu_c_opt, mu_q_opt = result.x
mse_opt = result.fun

print(f"\nOptimized parameters:")
print(f"  k_θ  = {k_opt:.4f}")
print(f"  ζ    = {zeta_opt:.6f}")
print(f"  μ_c  = {mu_c_opt:.6f}")
print(f"  μ_q  = {mu_q_opt:.6f}")
print(f"  MSE  = {mse_opt:.6e}")

# Simulate with optimized parameters
t_sim, theta_sim, _ = simulate(k_opt, zeta_opt, mu_c_opt, mu_q_opt, theta0, theta_dot0, (t[0], t[-1]))
theta_sim_interp = np.interp(t, t_sim, theta_sim)

# Calculate correlation
corr = np.corrcoef(theta, theta_sim_interp)[0, 1]
print(f"  Correlation: {corr:.4f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Oscillation with peaks
ax = axes[0, 0]
ax.plot(t, np.degrees(theta), 'b-', linewidth=0.5, alpha=0.8, label='Experimental')
ax.scatter(peak_times, np.degrees(theta[peaks]), c='red', s=30, zorder=5, label='Peaks')
ax.scatter(valley_times, np.degrees(theta[valleys]), c='green', s=30, zorder=5, label='Valleys')
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
        label=f'Exp fit: A={np.degrees(A0):.1f}°×exp(-{decay_rate:.3f}t)\nR² = {R2:.4f}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (degrees)')
ax.set_title('Amplitude Decay (Viscous Damping)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Experimental vs Simulated
ax = axes[1, 0]
ax.plot(t, np.degrees(theta), 'b-', linewidth=1, alpha=0.7, label='Experimental')
ax.plot(t, np.degrees(theta_sim_interp), 'r--', linewidth=1.5, label='Simulated (optimized)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title(f'Comparison: Experimental vs Simulated (corr={corr:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Parameter summary
ax = axes[1, 1]
ax.axis('off')
summary = f"""
══════════════════════════════════════════
    80° EXPERIMENT - FINAL RESULTS
══════════════════════════════════════════

MEASURED QUANTITIES:
  Period:              T = {period:.4f} s
  Frequency:           f = {freq:.4f} Hz
  Angular frequency:   ω = {omega:.4f} rad/s

DECAY ANALYSIS:
  Decay rate:          λ = {decay_rate:.6f} (1/s)
  Initial amplitude:   A₀ = {np.degrees(A0):.2f}°
  Fit quality:         R² = {R2:.4f}

  Envelope: A(t) = A₀ × exp(-λt)

ESTIMATED PARAMETERS:
  Stiffness:           k_θ = {k_opt:.4f}
  Viscous damping:     ζ   = {zeta_opt:.6f}
  Coulomb friction:    μ_c = {mu_c_opt:.6f}
  Quadratic damping:   μ_q = {mu_q_opt:.6f}

  Simple estimate:     ζ = λ/ω = {zeta:.6f}

DAMPING TYPE: VISCOUS (exponential decay)
══════════════════════════════════════════
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('80° Experiment Analysis\nHorizontal Pendulum Parameter Estimation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'analysis_80deg_final.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n" + "=" * 60)
print("FINAL RESULTS (80° Experiment)")
print("=" * 60)
print(f"""
OSCILLATION:
  Period T = {period:.4f} s
  Frequency f = {freq:.4f} Hz
  Angular frequency ω = {omega:.4f} rad/s

DAMPING (exponential decay, R² = {R2:.4f}):
  Decay rate λ = {decay_rate:.6f} (1/s)
  Damping ratio ζ = {zeta_opt:.6f}

STIFFNESS:
  k_θ = {k_opt:.4f}

ADDITIONAL DAMPING:
  Coulomb μ_c = {mu_c_opt:.6f}
  Quadratic μ_q = {mu_q_opt:.6f}
""")
print(f"Figure saved: {os.path.join(FIGURES_DIR, 'analysis_80deg_final.png')}")
print("=" * 60)
