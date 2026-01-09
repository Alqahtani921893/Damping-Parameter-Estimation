#!/usr/bin/env python3
"""
Final Summary of Experimental Analysis
Provides practical parameter estimates for the horizontal pendulum system.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

FIGURES_DIR = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Codes/Dec16/Damping-parameter-estimation-using-topological-signal-processing/figures/experimental'

# Direct measurements from experiments (robust values)
# These are directly observable from the data
experiments = {
    '60°': {'period': 0.21, 'freq_hz': 4.77, 'omega': 30.0, 'decay_rate': 0.23, 'decay_R2': 0.79},
    '70°': {'period': 0.25, 'freq_hz': 4.02, 'omega': 25.3, 'decay_rate': 0.20, 'decay_R2': 0.96},
    '80°': {'period': 0.29, 'freq_hz': 3.43, 'omega': 21.5, 'decay_rate': 0.18, 'decay_R2': 0.97},
    '90°': {'period': 0.32, 'freq_hz': 3.10, 'omega': 19.5, 'decay_rate': 0.15, 'decay_R2': 0.95},
    '100°': {'period': 0.36, 'freq_hz': 2.78, 'omega': 17.4, 'decay_rate': 0.12, 'decay_R2': 0.97},
    '110°': {'period': 0.36, 'freq_hz': 2.77, 'omega': 17.4, 'decay_rate': 0.12, 'decay_R2': 0.96},
    '120°': {'period': 0.40, 'freq_hz': 2.47, 'omega': 15.5, 'decay_rate': 0.10, 'decay_R2': 0.97},
}

print("=" * 70)
print("EXPERIMENTAL DATA ANALYSIS - FINAL SUMMARY")
print("Horizontal Pendulum Parameter Estimation")
print("=" * 70)

print("\n" + "=" * 70)
print("DIRECTLY MEASURED QUANTITIES")
print("=" * 70)

print(f"\n{'Exp':<8} {'Period(s)':<12} {'Freq(Hz)':<10} {'ω(rad/s)':<12} {'Decay Rate':<12} {'R²':<8}")
print("-" * 70)

periods = []
omegas = []
decay_rates = []

for name, data in experiments.items():
    print(f"{name:<8} {data['period']:<12.3f} {data['freq_hz']:<10.2f} {data['omega']:<12.2f} "
          f"{data['decay_rate']:<12.4f} {data['decay_R2']:<8.2f}")
    periods.append(data['period'])
    omegas.append(data['omega'])
    decay_rates.append(data['decay_rate'])

print("-" * 70)
print(f"{'Mean':<8} {np.mean(periods):<12.3f} {1/np.mean(periods):<10.2f} {np.mean(omegas):<12.2f} "
      f"{np.mean(decay_rates):<12.4f}")
print(f"{'Std':<8} {np.std(periods):<12.3f} {'':<10} {np.std(omegas):<12.2f} "
      f"{np.std(decay_rates):<12.4f}")

print("\n" + "=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)

print("""
1. OSCILLATION FREQUENCY:
   - Period increases with initial angle (0.21s at 60° → 0.40s at 120°)
   - This indicates NONLINEAR restoring force (amplitude-dependent frequency)
   - At small amplitudes: ω₀ ≈ 15-30 rad/s (f ≈ 2.5-4.8 Hz)

2. DAMPING CHARACTERISTICS:
   - Decay pattern is consistently EXPONENTIAL (viscous-type)
   - All experiments show R² > 0.79 for exponential fit
   - Decay rate λ ≈ 0.10 - 0.23 (1/s)
   - This corresponds to: A(t) = A₀ × exp(-λt)

3. DAMPING RATIO ESTIMATION:
   For viscous damping: λ = ζ × ω₀
   Using average values: ζ ≈ λ/ω = 0.16/20.9 ≈ 0.0077

   However, this assumes pure viscous damping. The high variability
   suggests there may be a combination of damping mechanisms.
""")

# Estimate parameters
omega_avg = np.mean(omegas)
decay_avg = np.mean(decay_rates)

# For model: θ̈ + 2ζω₀θ̇ + ω₀²θ = restoring terms
# Decay envelope: A(t) = A₀ exp(-ζω₀t)
# So: λ = ζω₀ => ζ = λ/ω₀

zeta_est = decay_avg / omega_avg

# For stiffness, we use: ω₀² = k_eff/I (effective stiffness over inertia)
# Without knowing I, we can only estimate k_eff/I = ω₀²

print("=" * 70)
print("ESTIMATED PARAMETERS")
print("=" * 70)

print(f"""
NATURAL FREQUENCY (small amplitude limit):
  ω₀ ≈ {omega_avg:.2f} ± {np.std(omegas):.2f} rad/s
  f₀ ≈ {omega_avg/(2*np.pi):.2f} Hz
  T₀ ≈ {np.mean(periods):.3f} s

DAMPING (assuming viscous-dominated):
  Decay rate λ ≈ {decay_avg:.4f} ± {np.std(decay_rates):.4f} (1/s)
  Damping ratio ζ ≈ {zeta_est:.4f} (= λ/ω₀)

  Note: Low damping ratio indicates underdamped system (ζ << 1)

STIFFNESS PARAMETER:
  If using model: θ̈ + 2ζω₀θ̇ + k_θ×θ - cos(θ) = 0
  Then: k_θ = ω₀² + 1 ≈ {omega_avg**2 + 1:.1f}

  However, the high value suggests the physical model may differ.
  The stiffness parameter depends on your specific system setup.
""")

print("=" * 70)
print("RECOMMENDATIONS FOR YOUR MODEL")
print("=" * 70)

print("""
Based on the experimental data, your pendulum shows:

1. DOMINANT DAMPING TYPE: Viscous (exponential decay)
   - The decay is well-described by A(t) = A₀ × exp(-λt)
   - Coulomb friction appears minimal (linear decay would show poor fit)

2. PRACTICAL PARAMETER VALUES for simulation:

   For a LINEAR model (small angles):
     θ̈ + 2ζω_n × θ̇ + ω_n² × θ = 0
     where: ω_n ≈ 20 rad/s, ζ ≈ 0.008

   For a NONLINEAR model (horizontal pendulum):
     θ̈ + 2ζω_n × θ̇ + k_θ × θ - cos(θ) = 0
     The k_θ value depends on your physical setup.

3. TO BETTER ESTIMATE k_θ:
   - Measure the equilibrium angle under gravity
   - Determine the spring constant or torsional stiffness
   - Or: use the measured frequency with known moment of inertia

4. AMPLITUDE-DEPENDENT EFFECTS:
   - Period increases ~90% from 60° to 120° initial angle
   - This strong nonlinearity affects parameter estimation
   - Consider using data from smaller amplitude oscillations
     for more accurate k_θ estimation
""")

# Create summary figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

names = list(experiments.keys())
x = np.arange(len(names))

# Plot 1: Period vs Initial Angle
ax = axes[0, 0]
periods_plot = [experiments[n]['period'] for n in names]
ax.bar(x, periods_plot, color='steelblue', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_xlabel('Initial Angle')
ax.set_ylabel('Period (s)')
ax.set_title('Oscillation Period vs Initial Angle')
ax.grid(True, alpha=0.3, axis='y')

# Add trend line
angles_numeric = [60, 70, 80, 90, 100, 110, 120]
z = np.polyfit(angles_numeric, periods_plot, 2)
p = np.poly1d(z)
ax.plot(x, p(angles_numeric), 'r--', linewidth=2, label='Quadratic fit')
ax.legend()

# Plot 2: Frequency vs Initial Angle
ax = axes[0, 1]
omegas_plot = [experiments[n]['omega'] for n in names]
ax.bar(x, omegas_plot, color='darkorange', alpha=0.8)
ax.axhline(np.mean(omegas_plot), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(omegas_plot):.1f} rad/s')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_xlabel('Initial Angle')
ax.set_ylabel('Angular Frequency ω (rad/s)')
ax.set_title('Natural Frequency vs Initial Angle')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Decay Rate
ax = axes[1, 0]
decay_plot = [experiments[n]['decay_rate'] for n in names]
ax.bar(x, decay_plot, color='forestgreen', alpha=0.8)
ax.axhline(np.mean(decay_plot), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(decay_plot):.3f}')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_xlabel('Initial Angle')
ax.set_ylabel('Decay Rate λ (1/s)')
ax.set_title('Exponential Decay Rate')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Summary text
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
FINAL PARAMETER ESTIMATES

Oscillation Frequency:
  ω₀ = {omega_avg:.2f} ± {np.std(omegas):.2f} rad/s
  f₀ = {omega_avg/(2*np.pi):.2f} Hz
  T₀ = {np.mean(periods):.3f} s

Damping (Viscous):
  Decay rate λ = {decay_avg:.4f} (1/s)
  Damping ratio ζ = {zeta_est:.4f}

  Envelope: A(t) = A₀ × exp(-{decay_avg:.3f}t)

Damping Type: VISCOUS (exponential)
  All experiments show R² > 0.79
  for exponential decay fit

System Characteristics:
  • Underdamped (ζ << 1)
  • Nonlinear restoring force
  • Period increases with amplitude
"""
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Experimental Analysis Summary\nHorizontal Pendulum Parameter Estimation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'final_summary.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSummary figure saved: {os.path.join(FIGURES_DIR, 'final_summary.png')}")
print("\n" + "=" * 70)
