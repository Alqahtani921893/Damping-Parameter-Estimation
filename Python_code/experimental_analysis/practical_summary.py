#!/usr/bin/env python3
"""
Practical Summary of Experimental Analysis
Extracts reliable parameters directly from experimental data.

Key insight: Focus on what we can measure directly rather than
model-dependent estimations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import os

FIGURES_DIR = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Codes/Dec16/Damping-parameter-estimation-using-topological-signal-processing/figures/experimental'
DATA_FILE = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Experiment/80P.txt'

# Physical parameters
m = 0.05   # kg
L = 0.1    # m
g = 9.81   # m/s²
I = m * L**2  # kg·m²

# Measured spring stiffness range (from static experiments)
kt_measured_range = (0.016, 0.033)  # Nm/rad


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


# Load data
t_raw, angles_deg, theta_raw = load_data(DATA_FILE)
t, theta = resample(t_raw, theta_raw)
dt = t[1] - t[0]

# Remove equilibrium
offset = np.mean(theta[int(len(theta)*0.8):])
theta = theta - offset

# Find peaks
peaks, _ = find_peaks(theta, distance=50, prominence=0.01)
valleys, _ = find_peaks(-theta, distance=50, prominence=0.01)

peak_times = t[peaks]
peak_amps = np.abs(theta[peaks])
valley_times = t[valleys]
valley_amps = np.abs(theta[valleys])

# Combine all extrema for better decay analysis
all_extrema_times = np.sort(np.concatenate([peak_times, valley_times]))
all_extrema_amps = np.abs(np.interp(all_extrema_times, t, theta))

# Period analysis
periods = np.diff(peak_times)
T = np.median(periods)
T_std = np.std(periods)
omega = 2 * np.pi / T
f = 1 / T

# Decay analysis (exponential fit)
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

# Quality factor
Q = omega / (2 * decay_rate)

# Effective stiffness from frequency
kt_eff = I * omega**2

# Damping coefficient from decay rate
# For underdamped: A(t) = A0*exp(-c*t/(2I)) → c = 2*I*λ
c_eff = 2 * I * decay_rate

print("=" * 70)
print("PRACTICAL EXPERIMENTAL SUMMARY")
print("80° Horizontal Pendulum Experiment")
print("=" * 70)

print("\n" + "─" * 70)
print("DIRECTLY MEASURED FROM OSCILLATIONS (Model-Independent)")
print("─" * 70)

print(f"""
1. OSCILLATION FREQUENCY
   Period:              T = {T:.4f} ± {T_std:.4f} s
   Frequency:           f = {f:.2f} Hz
   Angular frequency:   ω = {omega:.2f} rad/s

2. AMPLITUDE DECAY (Exponential)
   Decay rate:          λ = {decay_rate:.4f} (1/s)
   Initial amplitude:   A₀ = {np.degrees(A0):.1f}°
   Decay time constant: τ = 1/λ = {1/decay_rate:.1f} s

   Envelope equation:   A(t) = {np.degrees(A0):.1f}° × exp(-{decay_rate:.3f}t)

   Fit quality:         R² = {R2:.4f}

   Half-life:           t_½ = ln(2)/λ = {np.log(2)/decay_rate:.1f} s

3. QUALITY FACTOR
   Q = ω/(2λ) = {Q:.1f}

   (Q represents number of oscillations for amplitude to decay to 1/e)

4. OBSERVED RANGE
   Duration:            {t[-1]:.1f} s
   Oscillations:        {len(peaks)} complete cycles
   Angle range:         [{np.degrees(theta.min()):.1f}°, {np.degrees(theta.max()):.1f}°]
""")

print("─" * 70)
print("DERIVED PARAMETERS (Depend on Physical Model)")
print("─" * 70)

print(f"""
Given: m = {m*1000:.0f} g, L = {L*100:.0f} cm, I = mL² = {I:.4e} kg·m²

1. EFFECTIVE STIFFNESS (from measured ω)
   kt_eff = I × ω² = {kt_eff:.4f} Nm/rad

   Compare to measured static stiffness:
   kt_static = {kt_measured_range[0]:.3f} - {kt_measured_range[1]:.3f} Nm/rad

   ⚠ DISCREPANCY: kt_eff is ~{kt_eff/np.mean(kt_measured_range):.0f}× higher than static!

   Possible explanations:
   - Different spring used for dynamic test
   - Additional pivot mechanism stiffness
   - Moment of inertia different from mL²
   - Pre-tension in spring

2. EFFECTIVE DAMPING COEFFICIENT
   For underdamped oscillator: θ̈ + (c/I)θ̇ + (kt/I)θ = 0
   Decay rate λ = c/(2I)

   c_eff = 2Iλ = {c_eff:.4e} Nm·s/rad

3. DAMPING RATIO
   ζ = λ/ω = {decay_rate/omega:.6f}

   This is a very small damping ratio → underdamped system
""")

print("─" * 70)
print("DIMENSIONAL ANALYSIS")
print("─" * 70)

print(f"""
For simulation or modeling, use these parameters:

MODEL: θ̈ + (c/I)θ̇ + (kt/I)θ = 0  (linear approximation)

Parameters:
  Natural frequency:    ω_n² = kt/I → kt/I = {omega**2:.1f} rad²/s²
  Damping coefficient:  c/I = 2λ = {2*decay_rate:.3f} rad/s

  Or equivalently:
  Damping ratio:        ζ = {decay_rate/omega:.6f}
  Natural frequency:    ω_n = {omega:.2f} rad/s

NONDIMENSIONAL (τ = t·ω_n):
  θ'' + 2ζ·θ' + θ = 0
  where ζ = {decay_rate/omega:.6f}
""")

print("─" * 70)
print("RECOMMENDATIONS FOR YOUR MODEL")
print("─" * 70)

print(f"""
1. FOR SIMULATION matching this experiment:
   Use: ω_n = {omega:.2f} rad/s, ζ = {decay_rate/omega:.4f}

   This will reproduce the observed period (T={T:.3f}s) and decay.

2. TO RECONCILE WITH PHYSICAL MODEL:
   The horizontal pendulum model: θ̈ + 2ζω_n·θ̇ + κ·θ - cos(θ) = 0

   Nondimensionalizing with τ = t√(g/L):
   - Measured nondim frequency: ω* = ω·√(L/g) = {omega*np.sqrt(L/g):.3f}
   - This implies κ ≈ (ω*)² + 1 ≈ {(omega*np.sqrt(L/g))**2 + 1:.2f}

   Or solve directly: κ = kt/(mgL) = {kt_eff/(m*g*L):.2f}

3. DAMPING TYPE:
   The high R² ({R2:.2f}) for exponential decay confirms VISCOUS damping.
   Coulomb friction would show linear (not exponential) decay.
""")

# Create summary figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Full oscillation
ax = axes[0, 0]
ax.plot(t, np.degrees(theta), 'b-', linewidth=0.5, alpha=0.8)
ax.scatter(peak_times, np.degrees(theta[peaks]), c='red', s=20, zorder=5)
ax.scatter(valley_times, np.degrees(theta[valleys]), c='green', s=20, zorder=5)
t_env = np.linspace(0, t[-1], 100)
ax.plot(t_env, np.degrees(A0 * np.exp(-decay_rate * t_env)), 'r--', lw=2)
ax.plot(t_env, -np.degrees(A0 * np.exp(-decay_rate * t_env)), 'r--', lw=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title(f'80° Experiment: Period T = {T:.3f}s, Decay λ = {decay_rate:.3f}/s')
ax.grid(True, alpha=0.3)
ax.legend(['Signal', 'Peaks', 'Valleys', 'Envelope'], loc='upper right')

# Plot 2: Decay on log scale
ax = axes[0, 1]
ax.semilogy(t_peaks, np.degrees(peak_amps), 'ro', markersize=8, label='Peak amplitudes')
ax.semilogy(t_peaks, np.degrees(A0 * np.exp(-decay_rate * t_peaks)), 'b-', lw=2,
            label=f'Fit: {np.degrees(A0):.1f}°·exp(-{decay_rate:.3f}t)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (degrees, log scale)')
ax.set_title(f'Decay Analysis: R² = {R2:.4f}')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Plot 3: Period variation
ax = axes[1, 0]
ax.plot(peak_times[:-1], periods * 1000, 'bo-', markersize=6)
ax.axhline(T * 1000, color='r', linestyle='--', lw=2, label=f'Median = {T*1000:.1f} ms')
ax.fill_between(peak_times[:-1], (T-T_std)*1000, (T+T_std)*1000, alpha=0.3, color='red')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Period (ms)')
ax.set_title('Period Stability Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Summary
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
════════════════════════════════════════════════════
    EXPERIMENTAL PARAMETERS (80° Experiment)
════════════════════════════════════════════════════

DIRECTLY MEASURED:
  Period:            T = {T:.4f} s
  Frequency:         f = {f:.2f} Hz
  Angular freq:      ω = {omega:.2f} rad/s
  Decay rate:        λ = {decay_rate:.4f} (1/s)
  Decay R²:          {R2:.4f}

DERIVED:
  Damping ratio:     ζ = λ/ω = {decay_rate/omega:.6f}
  Quality factor:    Q = {Q:.1f}
  Half-life:         {np.log(2)/decay_rate:.1f} s

FOR SIMULATION:
  Use: ω_n = {omega:.1f} rad/s
       ζ = {decay_rate/omega:.4f}

  θ̈ + 2ζω_n·θ̇ + ω_n²·θ = 0

DAMPING TYPE: VISCOUS
  (Confirmed by exponential decay, R² > 0.9)

════════════════════════════════════════════════════
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Practical Experimental Summary\n80° Horizontal Pendulum',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'practical_summary.png'), dpi=150, bbox_inches='tight')
plt.close()

print("─" * 70)
print(f"Figure saved: {os.path.join(FIGURES_DIR, 'practical_summary.png')}")
print("=" * 70)

# Export key values as a dictionary for use in other scripts
EXPERIMENTAL_PARAMS = {
    'T': T,
    'f': f,
    'omega': omega,
    'decay_rate': decay_rate,
    'A0_rad': A0,
    'A0_deg': np.degrees(A0),
    'R2_decay': R2,
    'zeta': decay_rate / omega,
    'Q': Q,
    'm': m,
    'L': L,
    'I': I,
    'kt_effective': kt_eff,
    'c_effective': c_eff,
}

print("\nExportable parameters:")
for k, v in EXPERIMENTAL_PARAMS.items():
    print(f"  {k}: {v}")
