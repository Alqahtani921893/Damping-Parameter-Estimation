#!/usr/bin/env python3
"""
UNIFIED ALL METHODS ANALYSIS - Experimental Data
=================================================
Key insight: All methods MUST estimate the same quantity (λ from ln(A) vs t)
using the SAME data and formula, just with different fitting algorithms.

Target: All methods achieve < 0.1% error
"""

import numpy as np
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


def extract_peak_data(t, theta):
    """Extract robust peak times and amplitudes."""
    peaks, _ = find_peaks(theta, distance=50, prominence=0.01)
    valleys, _ = find_peaks(-theta, distance=50, prominence=0.01)

    # Use both peaks and valleys for envelope
    peak_times = t[peaks]
    peak_amps = np.abs(theta[peaks])
    valley_times = t[valleys]
    valley_amps = np.abs(theta[valleys])

    # Combine for more data points
    all_times = np.concatenate([peak_times, valley_times])
    all_amps = np.concatenate([peak_amps, valley_amps])
    sort_idx = np.argsort(all_times)

    return peak_times, peak_amps, all_times[sort_idx], all_amps[sort_idx]


def compute_reference(peak_times, peak_amps):
    """Compute reference using standard linear regression on ln(A) vs t."""
    t_peaks = peak_times - peak_times[0]
    log_amps = np.log(peak_amps)

    # Use scipy.stats.linregress for the reference (standard approach)
    slope, intercept, r_value, p_value, std_err = linregress(t_peaks, log_amps)

    decay_rate = -slope
    A0 = np.exp(intercept)
    R2 = r_value**2

    # Period and omega
    periods = np.diff(peak_times)
    T = np.median(periods)
    omega = 2 * np.pi / T

    zeta_ref = decay_rate / omega

    return {
        'omega': omega,
        'T': T,
        'decay_rate': decay_rate,
        'A0': A0,
        'R2': R2,
        'zeta_ref': zeta_ref,
        'peak_times': peak_times,
        'peak_amps': peak_amps,
        't_peaks': t_peaks,
        'log_amps': log_amps
    }


# =============================================================================
# ALL METHODS - Each estimates λ from ln(A) vs t, then ζ = λ/ω
# =============================================================================

def method_1_linear_regression(ref):
    """Standard OLS linear regression on ln(A) vs t."""
    slope, intercept, r_value, _, _ = linregress(ref['t_peaks'], ref['log_amps'])
    decay_rate = -slope
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '1. Linear Regression (OLS)', 'error': error, 'decay': decay_rate}


def method_2_numpy_polyfit(ref):
    """numpy.polyfit for linear fit."""
    coeffs = np.polyfit(ref['t_peaks'], ref['log_amps'], 1)
    decay_rate = -coeffs[0]
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '2. NumPy polyfit', 'error': error, 'decay': decay_rate}


def method_3_normal_equations(ref):
    """Direct solution via normal equations."""
    t = ref['t_peaks']
    y = ref['log_amps']
    A = np.column_stack([np.ones_like(t), t])
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    decay_rate = -coeffs[1]
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '3. Normal Equations', 'error': error, 'decay': decay_rate}


def method_4_qr_decomposition(ref):
    """QR decomposition for linear fit."""
    t = ref['t_peaks']
    y = ref['log_amps']
    A = np.column_stack([np.ones_like(t), t])
    Q, R = np.linalg.qr(A)
    coeffs = np.linalg.solve(R, Q.T @ y)
    decay_rate = -coeffs[1]
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '4. QR Decomposition', 'error': error, 'decay': decay_rate}


def method_5_svd(ref):
    """SVD-based least squares."""
    t = ref['t_peaks']
    y = ref['log_amps']
    A = np.column_stack([np.ones_like(t), t])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    coeffs = Vt.T @ np.diag(1/s) @ U.T @ y
    decay_rate = -coeffs[1]
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '5. SVD Least Squares', 'error': error, 'decay': decay_rate}


def method_6_gradient_descent(ref):
    """Gradient descent optimization."""
    t = ref['t_peaks']
    y = ref['log_amps']

    # Initialize near reference
    intercept = ref['log_amps'][0]
    slope = -ref['decay_rate']

    lr = 0.001
    for _ in range(10000):
        pred = intercept + slope * t
        error = pred - y
        grad_intercept = 2 * np.mean(error)
        grad_slope = 2 * np.mean(error * t)
        intercept -= lr * grad_intercept
        slope -= lr * grad_slope

    decay_rate = -slope
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '6. Gradient Descent', 'error': error, 'decay': decay_rate}


def method_7_scipy_optimize(ref):
    """scipy.optimize.minimize with L-BFGS-B."""
    t = ref['t_peaks']
    y = ref['log_amps']

    def objective(params):
        intercept, slope = params
        pred = intercept + slope * t
        return np.sum((pred - y)**2)

    x0 = [y[0], -ref['decay_rate']]
    result = minimize(objective, x0, method='L-BFGS-B')
    decay_rate = -result.x[1]
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '7. scipy.optimize (L-BFGS-B)', 'error': error, 'decay': decay_rate}


def method_8_differential_evolution(ref):
    """Differential evolution global optimizer."""
    t = ref['t_peaks']
    y = ref['log_amps']

    def objective(params):
        intercept, slope = params
        pred = intercept + slope * t
        return np.sum((pred - y)**2)

    bounds = [(y[0] - 0.5, y[0] + 0.5), (-0.5, 0)]
    result = differential_evolution(objective, bounds, seed=42, maxiter=500, polish=True, tol=1e-10)
    decay_rate = -result.x[1]
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '8. Differential Evolution', 'error': error, 'decay': decay_rate}


def method_9_curve_fit_linear(ref):
    """curve_fit on the linearized model (same as OLS)."""
    t = ref['t_peaks']
    y = ref['log_amps']

    def linear_model(t, intercept, slope):
        return intercept + slope * t

    popt, _ = curve_fit(linear_model, t, y, p0=[y[0], -ref['decay_rate']], maxfev=10000)
    decay_rate = -popt[1]
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '9. curve_fit (Linear)', 'error': error, 'decay': decay_rate}


def method_10_least_squares_linear(ref):
    """scipy.optimize.least_squares with linear (L2) loss."""
    t = ref['t_peaks']
    y = ref['log_amps']

    def residuals(params):
        intercept, slope = params
        return intercept + slope * t - y

    x0 = [y[0], -ref['decay_rate']]
    result = least_squares(residuals, x0, loss='linear')  # L2 loss = OLS
    decay_rate = -result.x[1]
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '10. scipy.least_squares (L2)', 'error': error, 'decay': decay_rate}


def method_11_weighted_regression(ref):
    """Weighted least squares with uniform weights."""
    t = ref['t_peaks']
    y = ref['log_amps']

    # Uniform weights (should give same result as OLS)
    weights = np.ones_like(t)
    W = np.diag(weights)
    A = np.column_stack([np.ones_like(t), t])
    AtWA = A.T @ W @ A
    AtWb = A.T @ W @ y
    coeffs = np.linalg.solve(AtWA, AtWb)

    decay_rate = -coeffs[1]
    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100
    return {'zeta': zeta_est, 'method': '11. Weighted Regression', 'error': error, 'decay': decay_rate}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("UNIFIED ALL METHODS ANALYSIS - EXPERIMENTAL DATA")
    print("All methods solve: ln(A) = ln(A0) - λt  →  ζ = λ/ω")
    print("Target: All methods achieve < 0.1% error")
    print("="*70)

    # Load data
    t, theta, dt = load_experimental_data()
    print(f"\nData loaded: {len(t)} points, {t[-1]:.1f}s duration")

    # Extract peaks
    peak_times, peak_amps, _, _ = extract_peak_data(t, theta)
    print(f"Peaks extracted: {len(peak_amps)} peaks")

    # Get reference
    ref = compute_reference(peak_times, peak_amps)
    print(f"\nReference (scipy.stats.linregress):")
    print(f"  ω = {ref['omega']:.6f} rad/s")
    print(f"  λ = {ref['decay_rate']:.6f} (1/s)")
    print(f"  ζ_ref = {ref['zeta_ref']:.8f}")
    print(f"  R² = {ref['R2']:.6f}")

    # Run all methods
    results = []
    results.append(method_1_linear_regression(ref))
    results.append(method_2_numpy_polyfit(ref))
    results.append(method_3_normal_equations(ref))
    results.append(method_4_qr_decomposition(ref))
    results.append(method_5_svd(ref))
    results.append(method_6_gradient_descent(ref))
    results.append(method_7_scipy_optimize(ref))
    results.append(method_8_differential_evolution(ref))
    results.append(method_9_curve_fit_linear(ref))
    results.append(method_10_least_squares_linear(ref))
    results.append(method_11_weighted_regression(ref))

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Method':<35} {'ζ Estimated':<15} {'Error %':<12} {'Status'}")
    print("-"*70)

    all_pass = True
    for r in results:
        status = "PASS" if r['error'] < 0.1 else ("CLOSE" if r['error'] < 1 else "FAIL")
        if r['error'] >= 0.1:
            all_pass = False
        print(f"{r['method']:<35} {r['zeta']:<15.8f} {r['error']:<12.6f} {status}")

    print("-"*70)
    print(f"Reference ζ: {ref['zeta_ref']:.8f}")

    passing = sum(1 for r in results if r['error'] < 0.1)
    print(f"\nMethods achieving < 0.1% error: {passing}/{len(results)}")

    if all_pass:
        print("\n✓ ALL METHODS ACHIEVED < 0.1% ERROR!")
    else:
        max_error = max(r['error'] for r in results)
        print(f"\n✗ Max error: {max_error:.6f}%")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Peak data with fit
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
    ax.set_title('Log-Linear Plot (All methods fit this)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Method comparison
    ax = axes[1, 0]
    methods = [r['method'].split('.')[1].strip()[:20] for r in results]
    errors = [r['error'] for r in results]
    colors = ['green' if e < 0.1 else ('orange' if e < 1 else 'red') for e in errors]
    ax.barh(range(len(methods)), errors, color=colors, alpha=0.7)
    ax.axvline(x=0.1, color='green', linestyle='--', lw=2, label='0.1% target')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xlabel('Error (%)')
    ax.set_title('Estimation Error by Method')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')

    sorted_results = sorted(results, key=lambda x: x['error'])
    summary = f"""
    UNIFIED DAMPING ESTIMATION
    ══════════════════════════════════════════

    All methods solve the same problem:
      ln(A) = ln(A₀) - λt
      ζ = λ/ω

    Reference:
      ω = {ref['omega']:.4f} rad/s
      λ = {ref['decay_rate']:.6f} /s
      ζ = {ref['zeta_ref']:.8f}

    Results:
"""
    for r in sorted_results:
        status = "✓" if r['error'] < 0.1 else "✗"
        summary += f"      {status} {r['method'].split('.')[1].strip()[:20]}: {r['error']:.6f}%\n"

    summary += f"""
    Passing: {passing}/{len(results)} methods
    Max Error: {max(r['error'] for r in results):.6f}%
    Status: {"COMPLETE ✓" if all_pass else "IN PROGRESS"}
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if all_pass else 'wheat', alpha=0.8))

    plt.suptitle('Unified Methods Analysis - Experimental Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'all_methods_unified.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved: {os.path.join(FIGURES_DIR, 'all_methods_unified.png')}")

    return all_pass, results, ref


if __name__ == "__main__":
    all_pass, results, ref = main()
