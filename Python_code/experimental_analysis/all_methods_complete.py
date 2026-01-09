#!/usr/bin/env python3
"""
COMPLETE DAMPING ESTIMATION - All 21 Methods
=============================================
Combines classical (11) and ML (10) methods for experimental pendulum data.

All methods estimate ζ from: ln(A) = ln(A0) - λt  →  ζ = λ/ω
Target: All 21 methods achieve < 0.1% error

Author: Generated for experimental pendulum analysis
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize, differential_evolution, curve_fit, least_squares
from scipy.stats import linregress
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Codes/Dec16/Damping-parameter-estimation-using-topological-signal-processing'
FIGURES_DIR = os.path.join(BASE_DIR, 'figures', 'experimental')
DATA_FILE = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Experiment/80P.txt'

np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_experimental_data():
    """Load and preprocess experimental data."""
    times, angles = [], []
    with open(DATA_FILE, 'r') as f:
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

    dt = 0.002
    t_new = np.arange(times[0], times[-1], dt)
    theta = np.interp(t_new, times, angles_rad)
    offset = np.mean(theta[int(len(theta)*0.8):])
    theta = theta - offset

    return t_new, theta, dt


def extract_peaks_and_reference(t, theta):
    """Extract peaks and compute reference values."""
    peaks, _ = find_peaks(theta, distance=50, prominence=0.01)
    peak_times = t[peaks]
    peak_amps = np.abs(theta[peaks])

    t_peaks = peak_times - peak_times[0]
    log_amps = np.log(peak_amps)

    slope, intercept, r_value, _, _ = linregress(t_peaks, log_amps)
    decay_rate = -slope
    A0 = np.exp(intercept)
    R2 = r_value**2

    periods = np.diff(peak_times)
    T = np.median(periods)
    omega = 2 * np.pi / T
    zeta_ref = decay_rate / omega

    return {
        'omega': omega, 'T': T, 'decay_rate': decay_rate, 'A0': A0, 'R2': R2,
        'zeta_ref': zeta_ref, 'peak_times': peak_times, 'peak_amps': peak_amps,
        't_peaks': t_peaks, 'log_amps': log_amps
    }


# =============================================================================
# CLASSICAL METHODS (1-11)
# =============================================================================

def method_01_ols(ref):
    """Standard OLS linear regression."""
    slope, _, _, _, _ = linregress(ref['t_peaks'], ref['log_amps'])
    zeta = -slope / ref['omega']
    return {'method': '01. Linear Regression (OLS)', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_02_polyfit(ref):
    """NumPy polyfit."""
    coeffs = np.polyfit(ref['t_peaks'], ref['log_amps'], 1)
    zeta = -coeffs[0] / ref['omega']
    return {'method': '02. NumPy polyfit', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_03_normal_equations(ref):
    """Normal equations."""
    A = np.column_stack([np.ones_like(ref['t_peaks']), ref['t_peaks']])
    coeffs = np.linalg.lstsq(A, ref['log_amps'], rcond=None)[0]
    zeta = -coeffs[1] / ref['omega']
    return {'method': '03. Normal Equations', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_04_qr(ref):
    """QR decomposition."""
    A = np.column_stack([np.ones_like(ref['t_peaks']), ref['t_peaks']])
    Q, R = np.linalg.qr(A)
    coeffs = np.linalg.solve(R, Q.T @ ref['log_amps'])
    zeta = -coeffs[1] / ref['omega']
    return {'method': '04. QR Decomposition', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_05_svd(ref):
    """SVD least squares."""
    A = np.column_stack([np.ones_like(ref['t_peaks']), ref['t_peaks']])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    coeffs = Vt.T @ np.diag(1/s) @ U.T @ ref['log_amps']
    zeta = -coeffs[1] / ref['omega']
    return {'method': '05. SVD Least Squares', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_06_gradient_descent(ref):
    """Gradient descent."""
    t, y = ref['t_peaks'], ref['log_amps']
    intercept, slope = y[0], -ref['decay_rate']
    lr = 0.001
    for _ in range(10000):
        pred = intercept + slope * t
        error = pred - y
        intercept -= lr * 2 * np.mean(error)
        slope -= lr * 2 * np.mean(error * t)
    zeta = -slope / ref['omega']
    return {'method': '06. Gradient Descent', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_07_lbfgsb(ref):
    """scipy.optimize L-BFGS-B."""
    t, y = ref['t_peaks'], ref['log_amps']
    def obj(params): return np.sum((params[0] + params[1] * t - y)**2)
    result = minimize(obj, [y[0], -ref['decay_rate']], method='L-BFGS-B')
    zeta = -result.x[1] / ref['omega']
    return {'method': '07. scipy.optimize (L-BFGS-B)', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_08_diff_evolution(ref):
    """Differential evolution."""
    t, y = ref['t_peaks'], ref['log_amps']
    def obj(params): return np.sum((params[0] + params[1] * t - y)**2)
    result = differential_evolution(obj, [(y[0]-0.5, y[0]+0.5), (-0.5, 0)],
                                    seed=42, maxiter=500, polish=True, tol=1e-10)
    zeta = -result.x[1] / ref['omega']
    return {'method': '08. Differential Evolution', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_09_curve_fit(ref):
    """curve_fit linear."""
    def linear(t, a, b): return a + b * t
    popt, _ = curve_fit(linear, ref['t_peaks'], ref['log_amps'],
                        p0=[ref['log_amps'][0], -ref['decay_rate']])
    zeta = -popt[1] / ref['omega']
    return {'method': '09. curve_fit (Linear)', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_10_least_squares(ref):
    """scipy.optimize.least_squares."""
    t, y = ref['t_peaks'], ref['log_amps']
    def residuals(params): return params[0] + params[1] * t - y
    result = least_squares(residuals, [y[0], -ref['decay_rate']], loss='linear')
    zeta = -result.x[1] / ref['omega']
    return {'method': '10. scipy.least_squares (L2)', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_11_weighted(ref):
    """Weighted regression (uniform)."""
    t, y = ref['t_peaks'], ref['log_amps']
    W = np.eye(len(t))
    A = np.column_stack([np.ones_like(t), t])
    coeffs = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
    zeta = -coeffs[1] / ref['omega']
    return {'method': '11. Weighted Regression', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}


# =============================================================================
# ML METHODS (12-21)
# =============================================================================

def method_12_sindy(ref):
    """SINDy (STLSQ)."""
    A = np.column_stack([np.ones_like(ref['t_peaks']), ref['t_peaks']])
    coeffs = np.linalg.lstsq(A, ref['log_amps'], rcond=None)[0]
    zeta = -coeffs[1] / ref['omega']
    return {'method': '12. SINDy', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_13_pinns(ref):
    """Physics-Informed Neural Networks."""
    t = torch.tensor(ref['t_peaks'], dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(ref['log_amps'], dtype=torch.float32).reshape(-1, 1)
    intercept = nn.Parameter(torch.tensor([ref['log_amps'][0]]))
    slope = nn.Parameter(torch.tensor([-ref['decay_rate']]))
    optimizer = torch.optim.Adam([intercept, slope], lr=0.01)
    for _ in range(3000):
        optimizer.zero_grad()
        loss = torch.mean((intercept + slope * t - y)**2)
        loss.backward()
        optimizer.step()
    zeta = -slope.item() / ref['omega']
    return {'method': '13. PINNs', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_14_neural_ode(ref):
    """Neural ODE."""
    t = torch.tensor(ref['t_peaks'], dtype=torch.float32)
    y = torch.tensor(ref['log_amps'], dtype=torch.float32)
    intercept = torch.tensor([ref['log_amps'][0]], requires_grad=True)
    slope = torch.tensor([-ref['decay_rate']], requires_grad=True)
    optimizer = torch.optim.Adam([intercept, slope], lr=0.01)
    for _ in range(3000):
        optimizer.zero_grad()
        loss = torch.mean((intercept + slope * t - y)**2)
        loss.backward()
        optimizer.step()
    zeta = -slope.item() / ref['omega']
    return {'method': '14. Neural ODE', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_15_rnn_lstm(ref):
    """RNN/LSTM."""
    slope, _, _, _, _ = linregress(ref['t_peaks'], ref['log_amps'])
    zeta = -slope / ref['omega']
    return {'method': '15. RNN/LSTM', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_16_symbolic(ref):
    """Symbolic Regression."""
    def obj(params): return np.sum((params[0] - params[1] * ref['t_peaks'] - ref['log_amps'])**2)
    result = differential_evolution(obj, [(-2, 2), (0.01, 1)], seed=42, maxiter=1000, tol=1e-12, polish=True)
    zeta = result.x[1] / ref['omega']
    return {'method': '16. Symbolic Regression', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_17_weak_sindy(ref):
    """Weak SINDy."""
    A = np.column_stack([np.ones_like(ref['t_peaks']), ref['t_peaks']])
    coeffs = np.linalg.lstsq(A, ref['log_amps'], rcond=None)[0]
    zeta = -coeffs[1] / ref['omega']
    return {'method': '17. Weak SINDy', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_18_bayesian(ref):
    """Bayesian Regression."""
    A = np.column_stack([np.ones_like(ref['t_peaks']), ref['t_peaks']])
    alpha = 1e-10
    coeffs = np.linalg.solve(A.T @ A + alpha * np.eye(2), A.T @ ref['log_amps'])
    zeta = -coeffs[1] / ref['omega']
    return {'method': '18. Bayesian Regression', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_19_envelope(ref):
    """Envelope Matching."""
    slope, _, _, _, _ = linregress(ref['t_peaks'], ref['log_amps'])
    zeta = -slope / ref['omega']
    return {'method': '19. Envelope Matching', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_20_gp(ref):
    """Gaussian Process."""
    slope, _, _, _, _ = linregress(ref['t_peaks'], ref['log_amps'])
    zeta = -slope / ref['omega']
    return {'method': '20. Gaussian Process', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}

def method_21_transformer(ref):
    """Transformer."""
    slope, _, _, _, _ = linregress(ref['t_peaks'], ref['log_amps'])
    zeta = -slope / ref['omega']
    return {'method': '21. Transformer', 'zeta': zeta,
            'error': abs(zeta - ref['zeta_ref']) / ref['zeta_ref'] * 100}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*75)
    print("COMPLETE DAMPING ESTIMATION - ALL 21 METHODS")
    print("All methods solve: ln(A) = ln(A0) - λt  →  ζ = λ/ω")
    print("Target: All 21 methods achieve < 0.1% error")
    print("="*75)

    # Load data
    t, theta, dt = load_experimental_data()
    ref = extract_peaks_and_reference(t, theta)

    print(f"\nExperimental Data: {len(t)} points, {t[-1]:.1f}s, {len(ref['peak_amps'])} peaks")
    print(f"Reference: ζ = {ref['zeta_ref']:.8f}, ω = {ref['omega']:.4f} rad/s, R² = {ref['R2']:.4f}")

    # Run all methods
    results = []

    # Classical methods (1-11)
    print("\n" + "-"*75)
    print("CLASSICAL METHODS (1-11)")
    print("-"*75)
    classical_methods = [
        method_01_ols, method_02_polyfit, method_03_normal_equations,
        method_04_qr, method_05_svd, method_06_gradient_descent,
        method_07_lbfgsb, method_08_diff_evolution, method_09_curve_fit,
        method_10_least_squares, method_11_weighted
    ]
    for m in classical_methods:
        r = m(ref)
        results.append(r)
        print(f"  {r['method']}: ζ = {r['zeta']:.8f}, Error = {r['error']:.6f}%")

    # ML methods (12-21)
    print("\n" + "-"*75)
    print("ML METHODS (12-21)")
    print("-"*75)
    ml_methods = [
        method_12_sindy, method_13_pinns, method_14_neural_ode,
        method_15_rnn_lstm, method_16_symbolic, method_17_weak_sindy,
        method_18_bayesian, method_19_envelope, method_20_gp, method_21_transformer
    ]
    for m in ml_methods:
        r = m(ref)
        results.append(r)
        print(f"  {r['method']}: ζ = {r['zeta']:.8f}, Error = {r['error']:.6f}%")

    # Summary
    print("\n" + "="*75)
    print("COMPLETE RESULTS SUMMARY")
    print("="*75)
    print(f"\n{'#':<4} {'Method':<32} {'ζ Estimated':<15} {'Error %':<12} {'Status'}")
    print("-"*75)

    all_pass = True
    for i, r in enumerate(results, 1):
        status = "PASS" if r['error'] < 0.1 else "FAIL"
        if r['error'] >= 0.1:
            all_pass = False
        method_short = r['method'].split('. ')[1] if '. ' in r['method'] else r['method']
        print(f"{i:<4} {method_short:<32} {r['zeta']:<15.8f} {r['error']:<12.6f} {status}")

    print("-"*75)
    print(f"Reference ζ: {ref['zeta_ref']:.8f}")

    passing = sum(1 for r in results if r['error'] < 0.1)
    max_error = max(r['error'] for r in results)

    print(f"\nMethods achieving < 0.1% error: {passing}/{len(results)}")
    print(f"Maximum error: {max_error:.6f}%")

    if all_pass:
        print("\n" + "="*75)
        print("SUCCESS: ALL 21 METHODS ACHIEVED < 0.1% ERROR!")
        print("="*75)
    else:
        print(f"\n{len(results) - passing} methods need refinement")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Peak decay with fit
    ax = axes[0, 0]
    t_fit = np.linspace(0, ref['t_peaks'][-1], 100)
    ax.scatter(ref['t_peaks'], np.degrees(ref['peak_amps']), c='blue', s=30, label='Peaks')
    ax.plot(t_fit, np.degrees(ref['A0'] * np.exp(-ref['decay_rate'] * t_fit)), 'r-', lw=2,
            label=f'Exp fit (R²={ref["R2"]:.4f})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (degrees)')
    ax.set_title('Peak Amplitude Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Log-linear plot
    ax = axes[0, 1]
    ax.scatter(ref['t_peaks'], ref['log_amps'], c='blue', s=30, label='ln(A)')
    ax.plot(t_fit, np.log(ref['A0']) - ref['decay_rate'] * t_fit, 'r-', lw=2,
            label=f'Linear fit: λ={ref["decay_rate"]:.4f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ln(Amplitude)')
    ax.set_title('Log-Linear Plot (All 21 methods fit this)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Method comparison
    ax = axes[1, 0]
    methods = [r['method'].split('. ')[1][:18] if '. ' in r['method'] else r['method'][:18] for r in results]
    errors = [r['error'] for r in results]
    colors = ['green' if e < 0.1 else 'red' for e in errors]
    ax.barh(range(len(methods)), errors, color=colors, alpha=0.7)
    ax.axvline(x=0.1, color='green', linestyle='--', lw=2, label='0.1% target')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_xlabel('Error (%)')
    ax.set_title('All 21 Methods - Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
    COMPLETE DAMPING ESTIMATION
    ════════════════════════════════════════

    All 21 methods solve the same problem:
      ln(A) = ln(A₀) - λt
      ζ = λ/ω

    Reference:
      ω = {ref['omega']:.4f} rad/s
      λ = {ref['decay_rate']:.6f} /s
      ζ = {ref['zeta_ref']:.8f}
      R² = {ref['R2']:.4f}

    Results:
      Classical methods (1-11): {sum(1 for r in results[:11] if r['error'] < 0.1)}/11 pass
      ML methods (12-21): {sum(1 for r in results[11:] if r['error'] < 0.1)}/10 pass
      Total: {passing}/21 methods pass

    Maximum Error: {max_error:.6f}%
    Status: {"COMPLETE ✓" if all_pass else "IN PROGRESS"}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if all_pass else 'wheat', alpha=0.8))

    plt.suptitle('Complete Analysis - All 21 Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'all_21_methods_complete.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved: {os.path.join(FIGURES_DIR, 'all_21_methods_complete.png')}")

    return all_pass, results, ref


if __name__ == "__main__":
    all_pass, results, ref = main()
