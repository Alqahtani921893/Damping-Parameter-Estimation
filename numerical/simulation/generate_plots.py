"""
Generate plots for nonlinear pendulum inverse parameter estimation report.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import argrelmax, argrelmin
from scipy.special import erfinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import operator
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (10, 6)

# Output directory
import os
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# PERSISTENCE FUNCTIONS
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


def fit_two_curves(x, y, func1, func2, initial_guess):
    """Fit data to two function models simultaneously."""
    def func(v):
        f1 = (y - np.array(func1(x, v[0], v[1], v[2], v[3]))) ** 2
        f2 = (y - np.array(func2(x, v[0], v[1], v[2], v[3]))) ** 2
        f = np.stack((f1, f2))
        f = np.min(f, axis=0)
        return np.sum(f) + np.max(f)

    v0 = initial_guess
    res = minimize(func, v0, method='BFGS', tol=10e-25)
    return res.x


def cutoff_from_lifetimes(L, len_ts, alpha, sigma):
    if len(L) == 0:
        return 0
    if sigma == False:
        mu_L = np.median(L)
        cutoff = 1.923 * mu_L * erfinv(2 * (1 - np.sqrt(alpha)) ** (1 / len_ts) - 1)
    else:
        cutoff = 2 ** (3 / 2) * sigma * erfinv(2 * (1 - np.sqrt(alpha)) ** (1 / len_ts) - 1)
    if cutoff > 0.3211 * max(L):
        cutoff = 0.3211 * max(L)
    return cutoff


def floor_from_lifetimes(L, t, L_sig, t_sig, cutoff, sigma):
    if len(L) > 0:
        alpha = 0.5
        t_stop = np.max(t_sig)
        N_stop = len(t[t < t_stop])
        n_floor = int(0.25 * N_stop / len(L_sig)) + 1
        if sigma == False:
            mu_L = np.median(L)
            floor = 1.923 * mu_L * erfinv(2 * (1 - np.sqrt(alpha)) ** (1 / n_floor) - 1)
        else:
            floor = 2 ** (3 / 2) * sigma * erfinv(2 * (1 - np.sqrt(alpha)) ** (1 / n_floor) - 1)
    if len(L) == 0:
        floor = 0
    if len(L) > 0 and cutoff == 0.3211 * max(L):
        floor = 0
    return floor


def damping_param_estimation_viscous(L, B, D, T_B, T_D, floor, L_all, T_all):
    """Estimate viscous damping parameters."""
    I_all = np.argsort(T_all)
    T_all = T_all[I_all]
    L_all = L_all[I_all]

    results = {}

    if len(L) > 1:
        I_opt = np.argmin(np.abs((L - floor) / ((L[0] - floor)) - 0.3299))
        delta = np.log((L[0] - floor) / (L[I_opt] - floor))
        zeta_opt = np.sqrt(1 / (1 + ((2 * np.pi * (0 - I_opt)) / delta) ** 2))

        def func1(data, a, b, c, d):
            return (a) * np.exp(-c * data) + b

        def func2(data, a, b, c, d):
            return b + 0 * data

        t_opt = np.max(T_all[L_all > 0.3299 * np.max(L_all)])
        a_guess = np.max(L_all)
        b_guess = 0.01 * np.max(L_all)
        c_guess = 2 * np.log(1.0 / 0.3299) / t_opt
        initial_guess = [a_guess, b_guess, c_guess, 0.0]
        parameters = fit_two_curves(T_all, L_all, func1, func2, initial_guess)
        a, b, c, d = parameters

        zeta_fit = (T_B[I_opt] - T_B[0]) * c / (I_opt * 2 * np.pi) if I_opt != 0 else np.nan

        results['zeta_opt'] = zeta_opt
        results['zeta_fit'] = zeta_fit
        results['fit_params'] = (a, b, c, d)
        results['func1'] = func1
        results['func2'] = func2

    if len(L) > 0:
        zeta_one = np.sqrt(1 / (1 + (np.pi / np.log((D[0] - 0.5 * floor) / (-B[0] - 0.5 * floor))) ** 2))
        results['zeta_one'] = zeta_one

    return results


# =============================================================================
# PENDULUM SIMULATION
# =============================================================================

def nonlinear_pendulum_ode(t, y, k_th, Om, qh, qv, zeta, mu_c, mu_q):
    """Nonlinear pendulum ODE."""
    theta = y[0]
    theta_dot = y[1]

    epsilon = 1e-6
    sign_smooth = np.tanh(theta_dot / epsilon)

    F_viscous = 2 * zeta * theta_dot
    F_coulomb = mu_c * sign_smooth
    F_quadratic = mu_q * theta_dot * np.abs(theta_dot)
    F_damping = F_viscous + F_coulomb + F_quadratic

    A = F_damping + k_th * theta - np.cos(theta) + Om**2 * np.sin(Om * t) * (qh * np.sin(theta) - qv * np.cos(theta))

    dydt = np.zeros(2)
    dydt[0] = theta_dot
    dydt[1] = -A

    return dydt


def simulate_pendulum(damping_type='viscous', zeta=0.05, mu_c=0.03, mu_q=0.05,
                      k_th=20, qh=0, qv=0, Om=5,
                      theta0=120, theta_dot0=0,
                      tf_cycles=100, dt=0.01, noise_std=0.0):
    """Simulate nonlinear pendulum."""
    if damping_type == 'viscous':
        zeta_use, mu_c_use, mu_q_use = zeta, 0, 0
    elif damping_type == 'coulomb':
        zeta_use, mu_c_use, mu_q_use = 0, mu_c, 0
    elif damping_type == 'quadratic':
        zeta_use, mu_c_use, mu_q_use = 0, 0, mu_q
    elif damping_type == 'combined':
        zeta_use, mu_c_use, mu_q_use = zeta, mu_c, mu_q
    else:
        raise ValueError(f"Unknown damping type: {damping_type}")

    T = 2 * np.pi / Om
    tf = tf_cycles * T
    t_span = (0, tf)
    t_eval = np.arange(0, tf, dt)

    y0 = [np.radians(theta0), np.radians(theta_dot0)]

    sol = solve_ivp(
        lambda t, y: nonlinear_pendulum_ode(t, y, k_th, Om, qh, qv, zeta_use, mu_c_use, mu_q_use),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-9
    )

    t = sol.t
    theta = sol.y[0]
    theta_dot = sol.y[1]

    if noise_std > 0:
        theta = theta + np.random.normal(0, noise_std, len(theta))

    params_used = {
        'damping_type': damping_type,
        'zeta': zeta_use,
        'mu_c': mu_c_use,
        'mu_q': mu_q_use,
        'k_th': k_th
    }

    return t, theta, theta_dot, params_used


# =============================================================================
# GENERATE PLOTS
# =============================================================================

print("Generating plots for nonlinear pendulum inverse estimation...")

# Simulation parameters
DAMPING_CONFIGS = {
    'viscous': {'zeta': 0.05, 'mu_c': 0, 'mu_q': 0, 'color': 'blue'},
    'coulomb': {'zeta': 0, 'mu_c': 0.03, 'mu_q': 0, 'color': 'red'},
    'quadratic': {'zeta': 0, 'mu_c': 0, 'mu_q': 0.05, 'color': 'green'},
}

# =============================================================================
# PLOT 1: Time Response Comparison
# =============================================================================
print("  [1/4] Time response comparison...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for idx, (damping_type, config) in enumerate(DAMPING_CONFIGS.items()):
    t, theta, theta_dot, params = simulate_pendulum(
        damping_type=damping_type,
        zeta=config['zeta'],
        mu_c=config['mu_c'],
        mu_q=config['mu_q'],
        theta0=120,
        tf_cycles=80,
        dt=0.01,
        noise_std=0.001
    )

    axes[idx].plot(t, np.degrees(theta), color=config['color'], linewidth=0.8)
    axes[idx].set_ylabel(r'$\theta$ [deg]')

    if damping_type == 'viscous':
        title = f'Viscous Damping ($\\zeta$ = {config["zeta"]})'
    elif damping_type == 'coulomb':
        title = f'Coulomb Damping ($\\mu_c$ = {config["mu_c"]})'
    else:
        title = f'Quadratic Damping ($\\mu_q$ = {config["mu_q"]})'

    axes[idx].set_title(title)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(0, 100)

axes[2].set_xlabel('Time [s]')
plt.suptitle('Nonlinear Pendulum Free Response with Different Damping Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_time_response.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# PLOT 2: Phase Portraits
# =============================================================================
print("  [2/4] Phase portraits...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for idx, (damping_type, config) in enumerate(DAMPING_CONFIGS.items()):
    t, theta, theta_dot, params = simulate_pendulum(
        damping_type=damping_type,
        zeta=config['zeta'],
        mu_c=config['mu_c'],
        mu_q=config['mu_q'],
        theta0=120,
        tf_cycles=80,
        dt=0.01,
        noise_std=0.0
    )

    axes[idx].plot(np.degrees(theta), np.degrees(theta_dot), color=config['color'], linewidth=0.5)
    axes[idx].set_xlabel(r'$\theta$ [deg]')
    axes[idx].set_ylabel(r'$\dot{\theta}$ [deg/s]')
    axes[idx].set_title(damping_type.capitalize())
    axes[idx].grid(True, alpha=0.3)
    # Use consistent axis limits instead of equal aspect ratio
    axes[idx].set_xlim(-130, 130)
    axes[idx].set_ylim(-500, 500)

plt.suptitle('Phase Portraits - Different Damping Types', fontsize=14, fontweight='bold')
plt.subplots_adjust(wspace=0.3)  # Reduce horizontal spacing
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_phase_portraits.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# PLOT 3: Persistence Lifetimes and Estimation (Viscous)
# =============================================================================
print("  [3/4] Persistence analysis (viscous)...")

# Simulate viscous damping
t, theta, theta_dot, params = simulate_pendulum(
    damping_type='viscous',
    zeta=0.05,
    theta0=120,
    tf_cycles=80,
    dt=0.01,
    noise_std=0.001
)

theta_centered = theta - np.mean(theta)

# Compute persistence
feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(theta_centered, 'localMin', edges=False)

if len(persistenceDgm) > 0:
    B = np.flip(persistenceDgm.T[0], axis=0)
    D = np.flip(persistenceDgm.T[1], axis=0)
    L = D - B
    I_B = np.array(feature_ind_1.astype(int)).T[0]
    I_D = np.array(feature_ind_2.astype(int)).T[0]
    I_D[I_D >= len(theta_centered)] = len(theta_centered) - 1
    T_B = np.flip(t[I_B], axis=0)
    T_D = np.flip(t[I_D], axis=0)

    # Compute cutoff and floor
    cutoff = cutoff_from_lifetimes(L, len(theta_centered), 0.01, False)

    I_insig = np.argwhere(L <= cutoff).flatten()
    I_sig = np.argwhere(L > cutoff).flatten()

    L_sig = L[I_sig]
    B_sig = B[I_sig]
    D_sig = D[I_sig]
    t_sig_B = T_B[I_sig]
    t_sig_D = T_D[I_sig]
    L_noise = L[I_insig]
    t_noise = T_B[I_insig]

    I_sort = np.argsort(t_sig_B)
    L_sig = L_sig[I_sort]
    B_sig = B_sig[I_sort]
    D_sig = D_sig[I_sort]
    t_sig_B = t_sig_B[I_sort]
    t_sig_D = t_sig_D[I_sort]

    floor = floor_from_lifetimes(L, t, L_sig, t_sig_B, cutoff, False)

    # Estimate parameters
    est_results = damping_param_estimation_viscous(L_sig, B_sig, D_sig, t_sig_B, t_sig_D, floor, L, T_B)

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    # Time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, np.degrees(theta), 'b-', linewidth=0.8, label='Signal')
    ax1.set_ylabel(r'$\theta$ [deg]')
    ax1.set_title(f'Viscous Damping: True $\\zeta$ = 0.05')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(t))
    ax1.legend()

    # Lifetimes
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(T_B, L, 'k.', alpha=0.3, markersize=3, label='All lifetimes')
    ax2.plot(t_noise, L_noise, 'r.', alpha=0.6, markersize=4, label='Noise ($L_N$)')
    ax2.plot(t_sig_B, L_sig, 'bd', markersize=6, label='Signal ($L_F$)')
    ax2.axhline(y=cutoff, color='k', linestyle='--', label=f'Cutoff = {cutoff:.4f}')
    ax2.axhline(y=floor, color='gray', linestyle='-', label=f'Floor = {floor:.4f}')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Lifetime $L$')
    ax2.set_title('Persistence Lifetimes')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(t))

    # Lifetime decay fitting
    ax3 = fig.add_subplot(gs[1, 1])

    I_all = np.argsort(T_B)
    T_all_sorted = T_B[I_all]
    L_all_sorted = L[I_all]

    ax3.plot(T_all_sorted, L_all_sorted, 'k.', alpha=0.5, markersize=4, label='Lifetimes')

    if 'fit_params' in est_results:
        a, b, c, d = est_results['fit_params']
        t_fit = np.linspace(min(T_all_sorted), max(T_all_sorted), 200)
        L_fit = a * np.exp(-c * t_fit) + b
        ax3.plot(t_fit, L_fit, 'b--', linewidth=2, label=f'Fit: $L = {a:.3f}e^{{-{c:.4f}t}} + {b:.3f}$')

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Lifetime $L$')
    ax3.set_title(f'Estimated $\\zeta_{{opt}}$ = {est_results.get("zeta_opt", np.nan):.4f}')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Topological Damping Parameter Estimation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_persistence_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
# PLOT 4: Summary Bar Chart
# =============================================================================
print("  [4/4] Summary comparison...")

# Run estimation for all damping types
results_summary = {}

for damping_type, config in DAMPING_CONFIGS.items():
    t, theta, theta_dot, params = simulate_pendulum(
        damping_type=damping_type,
        zeta=config['zeta'],
        mu_c=config['mu_c'],
        mu_q=config['mu_q'],
        theta0=120,
        tf_cycles=80,
        dt=0.01,
        noise_std=0.001
    )

    theta_centered = theta - np.mean(theta)
    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(theta_centered, 'localMin', edges=False)

    if len(persistenceDgm) > 0:
        B = np.flip(persistenceDgm.T[0], axis=0)
        D = np.flip(persistenceDgm.T[1], axis=0)
        L = D - B
        I_B = np.array(feature_ind_1.astype(int)).T[0]
        I_D = np.array(feature_ind_2.astype(int)).T[0]
        I_D[I_D >= len(theta_centered)] = len(theta_centered) - 1
        T_B = np.flip(t[I_B], axis=0)

        cutoff = cutoff_from_lifetimes(L, len(theta_centered), 0.01, False)
        I_sig = np.argwhere(L > cutoff).flatten()

        if len(I_sig) > 1:
            L_sig = L[I_sig]
            t_sig_B = T_B[I_sig]
            I_sort = np.argsort(t_sig_B)
            L_sig = L_sig[I_sort]
            t_sig_B = t_sig_B[I_sort]

            floor = floor_from_lifetimes(L, t, L_sig, t_sig_B, cutoff, False)

            # Simple estimation using optimal ratio
            I_opt = np.argmin(np.abs((L_sig - floor) / ((L_sig[0] - floor)) - 0.3299))
            if I_opt > 0:
                delta = np.log((L_sig[0] - floor) / (L_sig[I_opt] - floor))
                zeta_est = np.sqrt(1 / (1 + ((2 * np.pi * I_opt) / delta) ** 2))
            else:
                zeta_est = np.nan

            results_summary[damping_type] = {
                'true': config['zeta'] if damping_type == 'viscous' else (config['mu_c'] if damping_type == 'coulomb' else config['mu_q']),
                'estimated': zeta_est,
                'color': config['color']
            }

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(results_summary))
width = 0.35

true_vals = [results_summary[k]['true'] for k in results_summary.keys()]
est_vals = [results_summary[k]['estimated'] for k in results_summary.keys()]
colors = [results_summary[k]['color'] for k in results_summary.keys()]

bars1 = ax.bar(x - width/2, true_vals, width, label='True Value', color='gray', alpha=0.7)
bars2 = ax.bar(x + width/2, est_vals, width, label='Estimated Value', color=colors, alpha=0.7)

ax.set_xlabel('Damping Type')
ax.set_ylabel('Parameter Value')
ax.set_title('True vs Estimated Damping Parameters')
ax.set_xticks(x)
ax.set_xticklabels(['Viscous ($\\zeta$)', 'Coulomb ($\\mu_c$)', 'Quadratic ($\\mu_q$)'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars1, true_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10)

for bar, val in zip(bars2, est_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_parameter_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPlots saved to: {OUTPUT_DIR}")
print("  - fig_time_response.png")
print("  - fig_phase_portraits.png")
print("  - fig_persistence_analysis.png")
print("  - fig_parameter_comparison.png")
print("\nDone!")
