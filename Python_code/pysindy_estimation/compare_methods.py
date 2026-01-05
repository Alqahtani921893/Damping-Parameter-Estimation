"""
Comparison of Damping Parameter Estimation Methods

Compares three approaches:
1. Topological Signal Processing (Persistence Homology)
2. SINDy (Sparse Identification of Nonlinear Dynamics)
3. Optimization-based (Envelope matching)

All methods are applied to the same simulated pendulum data.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import argrelmax, argrelmin, hilbert, find_peaks, savgol_filter
from scipy.special import erfinv
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import operator
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import os
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# PENDULUM SIMULATION
# =============================================================================

def nonlinear_pendulum_ode(t, y, k_th, zeta, mu_c, mu_q):
    """Nonlinear pendulum ODE."""
    theta, theta_dot = y
    epsilon = 1e-6
    sign_smooth = np.tanh(theta_dot / epsilon)
    F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)
    A = F_damping + k_th * theta - np.cos(theta)
    return [theta_dot, -A]


def simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt=0.002):
    """Simulate pendulum."""
    y0 = [np.radians(theta0_deg), 0]
    t_eval = np.arange(0, t_final, dt)
    sol = solve_ivp(
        lambda t, y: nonlinear_pendulum_ode(t, y, k_th, zeta, mu_c, mu_q),
        (0, t_final), y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10
    )
    return sol.t, sol.y[0], sol.y[1]


# =============================================================================
# METHOD 1: TOPOLOGICAL (PERSISTENCE HOMOLOGY)
# =============================================================================

def Persistence0D(sample_data, min_or_max=0, edges=False):
    """Compute 0D persistence diagram from time series."""
    if min_or_max == 'localMax':
        min_or_max = 1
    else:
        min_or_max = 0

    from itertools import groupby
    sample_data = list(sample_data)
    sample_data = [k for k, g in groupby(sample_data) if k != 0]
    sample_data = np.array(sample_data)

    NegEnd = -100 * np.max(np.abs(sample_data))

    if edges == False:
        sample_data = np.insert(sample_data, 0, NegEnd, axis=0)
        sample_data = np.insert(sample_data, len(sample_data), NegEnd, axis=0)
        sample_data = np.insert(sample_data, len(sample_data), NegEnd / 2, axis=0)
        maxloc = np.array(argrelmax(sample_data, mode='clip'))
        minloc = np.array(argrelmin(sample_data, mode='clip'))
    else:
        maxloc = np.array(argrelmax(sample_data, mode='wrap'))
        minloc = np.array(argrelmin(sample_data, mode='wrap'))

    if len(maxloc[0]) == 0 or len(minloc[0]) == 0:
        return np.zeros((0, 1)), np.zeros((0, 1)), np.zeros((0, 2))

    max_vals = sample_data[maxloc]
    min_vals = sample_data[minloc]
    minmax_mat = np.concatenate((min_vals, max_vals, minloc, maxloc), axis=0)

    i = 1
    L = len(maxloc[0])
    persistenceDgm = np.zeros((L, 2))
    feature_ind_1 = np.zeros((L, 1))
    feature_ind_2 = np.zeros((L, 1))

    while (minmax_mat).shape[1] > 0.5:
        if maxloc[0][0] < minloc[0][0]:
            y = np.vstack((minmax_mat[1], minmax_mat[0])).T
        else:
            y = np.vstack((minmax_mat[0], minmax_mat[1])).T

        y = y.reshape(2 * len(minmax_mat[0]), )
        pairwiseDiff = abs(np.diff(y))
        differences = pairwiseDiff.reshape(len(pairwiseDiff), )
        smallestDiff_ind = min(enumerate(differences), key=operator.itemgetter(1))[0]

        if maxloc[0][0] < minloc[0][0]:
            ind1 = (int((smallestDiff_ind + 1) / 2))
            ind2 = (int((smallestDiff_ind) / 2))
        else:
            ind1 = (int((smallestDiff_ind) / 2))
            ind2 = (int((smallestDiff_ind + 1) / 2))

        peak_val = minmax_mat[1][ind1]
        peak_ind = minmax_mat[3][ind1]
        minmax_mat[1][ind1] = np.nan
        minmax_mat[3][ind1] = np.nan

        valley_val = minmax_mat[0][ind2]
        valley_ind = minmax_mat[2][ind2]
        minmax_mat[0][ind2] = np.nan
        minmax_mat[2][ind2] = np.nan

        if valley_val > NegEnd:
            feature_ind_1[i - 1] = (1 - min_or_max) * valley_ind + min_or_max * peak_ind
            feature_ind_2[i - 1] = (min_or_max) * valley_ind + (1 - min_or_max) * peak_ind
            persDgmPnt = [valley_val, peak_val]
            persistenceDgm[i - 1] = persDgmPnt

        for j in range(0, 4):
            temp = np.append([0], minmax_mat[j][~pd.isnull(minmax_mat[j])])
            minmax_mat[j] = temp
        minmax_mat = np.delete(minmax_mat, 0, axis=1)
        i = i + 1

    if edges == False:
        feature_ind_1 = feature_ind_1[:-1]
        feature_ind_2 = feature_ind_2[:-1]
        persistenceDgm = persistenceDgm[:-1]

    return feature_ind_1, feature_ind_2, persistenceDgm


def estimate_topological(t, theta, alpha=0.01):
    """Estimate viscous damping using topological method."""
    theta_centered = theta - np.mean(theta)

    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(theta_centered, 'localMin', edges=False)

    if len(persistenceDgm) == 0:
        return {'zeta_est': np.nan, 'method': 'topological'}

    B = np.flip(persistenceDgm.T[0], axis=0)
    D = np.flip(persistenceDgm.T[1], axis=0)
    L = D - B

    I_B = np.array(feature_ind_1.astype(int)).T[0]
    I_B[I_B >= len(theta_centered)] = len(theta_centered) - 1
    T_B = np.flip(t[I_B], axis=0)

    # Cutoff
    mu_L = np.median(L)
    cutoff = 1.923 * mu_L * erfinv(2 * (1 - np.sqrt(alpha)) ** (1 / len(theta)) - 1)
    if cutoff > 0.3211 * max(L):
        cutoff = 0.3211 * max(L)

    I_sig = np.argwhere(L > cutoff).flatten()

    if len(I_sig) < 2:
        return {'zeta_est': np.nan, 'method': 'topological'}

    L_sig = L[I_sig]
    t_sig_B = T_B[I_sig]

    I_sort = np.argsort(t_sig_B)
    L_sig = L_sig[I_sort]
    t_sig_B = t_sig_B[I_sort]

    # Floor
    t_stop = np.max(t_sig_B)
    N_stop = len(t[t < t_stop])
    n_floor = int(0.25 * N_stop / len(L_sig)) + 1
    floor = 1.923 * mu_L * erfinv(2 * (1 - np.sqrt(0.5)) ** (1 / n_floor) - 1)

    # Optimal ratio estimation
    I_opt = np.argmin(np.abs((L_sig - floor) / ((L_sig[0] - floor)) - 0.3299))

    if I_opt > 0 and L_sig[0] > floor and L_sig[I_opt] > floor:
        delta = np.log((L_sig[0] - floor) / (L_sig[I_opt] - floor))
        zeta_est = np.sqrt(1 / (1 + ((2 * np.pi * I_opt) / delta) ** 2))
    else:
        zeta_est = np.nan

    return {
        'zeta_est': zeta_est,
        'method': 'topological',
        'cutoff': cutoff,
        'floor': floor,
        'L_sig': L_sig
    }


# =============================================================================
# METHOD 2: SINDy (SPARSE IDENTIFICATION)
# =============================================================================

def stlsq(Theta, dXdt, threshold=0.1, max_iter=10):
    """Sequential Thresholded Least Squares."""
    n_features = Theta.shape[1]
    norms = np.linalg.norm(Theta, axis=0)
    norms[norms == 0] = 1
    Theta_norm = Theta / norms

    xi = np.linalg.lstsq(Theta_norm, dXdt, rcond=None)[0]

    for _ in range(max_iter):
        small_idx = np.abs(xi) < threshold
        xi[small_idx] = 0
        big_idx = ~small_idx
        if np.sum(big_idx) == 0:
            break
        xi[big_idx] = np.linalg.lstsq(Theta_norm[:, big_idx], dXdt, rcond=None)[0]

    return xi / norms


def estimate_sindy(t, theta, theta_dot, threshold=0.03):
    """Estimate damping using SINDy."""
    dt = t[1] - t[0]
    window = min(51, len(theta_dot) // 10)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)

    theta_ddot = savgol_filter(theta_dot, window_length=window, polyorder=3, deriv=1, delta=dt)

    # Build library
    epsilon = 1e-6
    sign_smooth = np.tanh(theta_dot / epsilon)

    Theta = np.column_stack([
        np.ones_like(theta),
        theta,
        theta_dot,
        np.cos(theta),
        np.sin(theta),
        theta_dot * np.abs(theta_dot),
        sign_smooth,
        theta ** 2,
        theta * theta_dot
    ])

    feature_names = ['1', 'θ', 'θ̇', 'cos(θ)', 'sin(θ)', 'θ̇|θ̇|', 'sign(θ̇)', 'θ²', 'θ·θ̇']

    # Trim edges
    trim = 50
    Theta = Theta[trim:-trim]
    theta_ddot = theta_ddot[trim:-trim]

    # STLSQ
    xi = stlsq(Theta, theta_ddot, threshold=threshold)

    # Extract parameters
    # θ̈ = c_θ·θ + c_θ̇·θ̇ + c_cos·cos(θ) + ...
    # Comparing to: θ̈ = -k_θ·θ - 2ζ·θ̇ + cos(θ) - ...

    zeta_est = -xi[2] / 2  # θ̇ coefficient / 2
    k_th_est = -xi[1]       # θ coefficient
    mu_q_est = -xi[5]       # θ̇|θ̇| coefficient
    mu_c_est = -xi[6]       # sign(θ̇) coefficient

    return {
        'zeta_est': zeta_est,
        'k_th_est': k_th_est,
        'mu_q_est': mu_q_est,
        'mu_c_est': mu_c_est,
        'method': 'sindy',
        'coefficients': xi,
        'feature_names': feature_names
    }


# =============================================================================
# METHOD 3: OPTIMIZATION-BASED (ENVELOPE MATCHING)
# =============================================================================

def extract_envelope(t, theta):
    """Extract amplitude envelope."""
    analytic = hilbert(theta)
    env = np.abs(analytic)
    return env


def estimate_optimization(t, theta, k_th, theta0_deg, t_final, dt=0.002):
    """Estimate damping using optimization (envelope matching)."""
    env_obs = extract_envelope(t, theta)

    def objective(zeta):
        if zeta <= 0:
            return 1e10
        try:
            t_sim, theta_sim, _ = simulate_pendulum(k_th, zeta, 0, 0, theta0_deg, t_final, dt)
            env_sim = extract_envelope(t_sim, theta_sim)
            env_sim_interp = np.interp(t, t_sim, env_sim)

            valid = env_obs > 0.01 * np.max(env_obs)
            if np.sum(valid) < 10:
                return 1e10

            log_obs = np.log(env_obs[valid] + 1e-10)
            log_sim = np.log(env_sim_interp[valid] + 1e-10)

            return np.mean((log_obs - log_sim) ** 2)
        except:
            return 1e10

    result = minimize_scalar(objective, bounds=(0.001, 0.3), method='bounded')

    return {
        'zeta_est': result.x,
        'method': 'optimization',
        'objective': result.fun
    }


# =============================================================================
# COMPARISON PIPELINE
# =============================================================================

def compare_methods(true_zeta=0.05, k_th=20, theta0_deg=30, t_final=60, dt=0.002,
                    noise_std=0.001, plotting=True):
    """
    Compare all three estimation methods on the same data.
    """
    print("\n" + "=" * 70)
    print("COMPARISON OF DAMPING PARAMETER ESTIMATION METHODS")
    print("=" * 70)

    # Simulate
    np.random.seed(42)
    t, theta, theta_dot = simulate_pendulum(k_th, true_zeta, 0, 0, theta0_deg, t_final, dt)

    # Add noise
    theta_noisy = theta + np.random.normal(0, noise_std, len(theta))
    theta_dot_noisy = theta_dot + np.random.normal(0, noise_std, len(theta_dot))

    print(f"\nSimulation Parameters:")
    print(f"  True ζ = {true_zeta}")
    print(f"  k_θ = {k_th}")
    print(f"  θ₀ = {theta0_deg}°")
    print(f"  Noise σ = {noise_std}")

    # Method 1: Topological
    print("\n[1] Topological (Persistence Homology)...")
    res_topo = estimate_topological(t, theta_noisy)
    print(f"    Estimated ζ = {res_topo['zeta_est']:.5f}")

    # Method 2: SINDy
    print("\n[2] SINDy (Sparse Identification)...")
    res_sindy = estimate_sindy(t, theta_noisy, theta_dot_noisy)
    print(f"    Estimated ζ = {res_sindy['zeta_est']:.5f}")

    # Method 3: Optimization
    print("\n[3] Optimization (Envelope Matching)...")
    res_opt = estimate_optimization(t, theta_noisy, k_th, theta0_deg, t_final, dt)
    print(f"    Estimated ζ = {res_opt['zeta_est']:.5f}")

    # Summary
    results = {
        'true': true_zeta,
        'topological': res_topo['zeta_est'],
        'sindy': res_sindy['zeta_est'],
        'optimization': res_opt['zeta_est']
    }

    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)
    print(f"{'Method':<20} {'Estimated ζ':<15} {'Error %':<10}")
    print("-" * 50)

    for method in ['topological', 'sindy', 'optimization']:
        est = results[method]
        if not np.isnan(est):
            err = abs(est - true_zeta) / true_zeta * 100
            status = "✓" if err < 5 else ("~" if err < 15 else "✗")
            print(f"{method:<20} {est:<15.5f} {err:<6.1f}%  {status}")
        else:
            print(f"{method:<20} {'N/A':<15} {'N/A':<10}")

    # Plotting
    if plotting:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Time series
        ax = axes[0, 0]
        ax.plot(t, np.degrees(theta_noisy), 'b-', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('θ [deg]')
        ax.set_title(f'Time Response (True ζ = {true_zeta})')
        ax.grid(True, alpha=0.3)

        # Phase portrait
        ax = axes[0, 1]
        ax.plot(np.degrees(theta_noisy), np.degrees(theta_dot_noisy), 'b-', linewidth=0.3)
        ax.set_xlabel('θ [deg]')
        ax.set_ylabel('θ̇ [deg/s]')
        ax.set_title('Phase Portrait')
        ax.grid(True, alpha=0.3)

        # Method comparison bar chart
        ax = axes[1, 0]
        methods = ['True', 'Topological', 'SINDy', 'Optimization']
        values = [true_zeta, res_topo['zeta_est'], res_sindy['zeta_est'], res_opt['zeta_est']]
        colors = ['black', 'blue', 'green', 'orange']
        bars = ax.bar(methods, values, color=colors, alpha=0.7)
        ax.set_ylabel('ζ')
        ax.set_title('Estimation Comparison')
        ax.axhline(y=true_zeta, color='red', linestyle='--', linewidth=2, label='True value')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add error annotations
        for i, (bar, val) in enumerate(zip(bars[1:], values[1:]), 1):
            if not np.isnan(val):
                err = abs(val - true_zeta) / true_zeta * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{err:.1f}%', ha='center', va='bottom', fontsize=10)

        # Error comparison
        ax = axes[1, 1]
        method_names = ['Topological', 'SINDy', 'Optimization']
        errors = []
        for method in ['topological', 'sindy', 'optimization']:
            est = results[method]
            if not np.isnan(est):
                errors.append(abs(est - true_zeta) / true_zeta * 100)
            else:
                errors.append(0)

        colors = ['green' if e < 5 else ('orange' if e < 15 else 'red') for e in errors]
        ax.bar(method_names, errors, color=colors, alpha=0.7)
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% threshold')
        ax.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='15% threshold')
        ax.set_ylabel('Error %')
        ax.set_title('Estimation Error Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Comparison of Damping Parameter Estimation Methods',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'method_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: method_comparison.png")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DAMPING PARAMETER ESTIMATION: METHOD COMPARISON")
    print("=" * 70)

    # Test with different noise levels
    noise_levels = [0.0, 0.001, 0.005, 0.01]

    all_results = []

    for noise in noise_levels:
        print(f"\n\n{'#' * 70}")
        print(f"NOISE LEVEL: σ = {noise}")
        print('#' * 70)

        results = compare_methods(
            true_zeta=0.05,
            k_th=20,
            theta0_deg=30,
            t_final=60,
            dt=0.002,
            noise_std=noise,
            plotting=(noise == 0.001)  # Only plot for one noise level
        )
        results['noise'] = noise
        all_results.append(results)

    # Final summary across noise levels
    print("\n\n" + "=" * 70)
    print("ROBUSTNESS ANALYSIS: ERROR vs NOISE LEVEL")
    print("=" * 70)
    print(f"\n{'Noise σ':<12} {'Topological':<15} {'SINDy':<15} {'Optimization':<15}")
    print("-" * 60)

    for res in all_results:
        noise = res['noise']
        true = res['true']
        topo_err = abs(res['topological'] - true) / true * 100 if not np.isnan(res['topological']) else np.nan
        sindy_err = abs(res['sindy'] - true) / true * 100 if not np.isnan(res['sindy']) else np.nan
        opt_err = abs(res['optimization'] - true) / true * 100 if not np.isnan(res['optimization']) else np.nan

        print(f"{noise:<12.4f} {topo_err:<15.2f} {sindy_err:<15.2f} {opt_err:<15.2f}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
