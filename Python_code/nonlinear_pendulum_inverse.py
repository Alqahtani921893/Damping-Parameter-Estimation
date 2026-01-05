"""
Nonlinear Pendulum Simulation and Inverse Parameter Estimation
Using Topological Signal Processing

Equation of Motion:
θ̈ + 2ζθ̇ + μ_c·sign(θ̇) + μ_q·θ̇|θ̇| + k_θ·θ - cos(θ) + Ω²sin(Ωt)(q_h·sin(θ) - q_v·cos(θ)) = 0
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


# =============================================================================
# TOPOLOGICAL SIGNAL PROCESSING FUNCTIONS (from code_with_example.ipynb)
# =============================================================================

def Persistence0D(sample_data, min_or_max=0, edges=False):
    """Compute 0D persistence diagram from time series."""
    if min_or_max == 'localMax':
        min_or_max = 1
    else:
        min_or_max = 0

    from itertools import groupby
    sample_data = [k for k, g in groupby(sample_data) if k != 0]
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


def damping_param_estimation(damping_type, L, B, D, T_B, T_D, floor, L_all, T_all, params, plotting=False):
    """Estimate damping parameters from persistence lifetimes."""
    I_all = np.argsort(T_all)
    T_all = T_all[I_all]
    L_all = L_all[I_all]

    if params == False:
        mass, N, spring = 1, 1, 1
    else:
        mass, N, spring = params[0], params[1], params[2]

    # Viscous damping
    if damping_type == 'viscous':
        if len(L) > 1:
            I_opt = np.argmin(np.abs((L - floor) / ((L[0] - floor)) - 0.3299))
            delta = np.log((L[0] - floor) / (L[I_opt] - floor))
            zeta_opt = np.sqrt(1 / (1 + ((2 * np.pi * (0 - I_opt)) / delta) ** 2))

            if params == False:
                mu_opt = np.nan
            else:
                mu_opt = zeta_opt * 2 * np.sqrt(mass * spring)

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

            if params == False:
                zeta_fit = (T_B[I_opt] - T_B[0]) * c / (I_opt * 2 * np.pi)
                mu_fit = np.nan
            else:
                zeta_fit = c / np.sqrt(spring / mass)
                mu_fit = 2 * zeta_fit * np.sqrt(spring * mass)

            if plotting:
                plt.figure(figsize=(8, 3))
                plt.xlabel('$t$', size=15)
                plt.ylabel('$L$', size=15)
                plt.plot(T_all, L_all, 'k.', label='L')
                t_plotting = np.linspace(min(T_all), max(T_all), 500)
                plt.plot(t_plotting, func1(t_plotting, a, b, c, d), 'b--', label='$f_S(t) = ae^{-ct} + b$')
                plt.plot(T_all, func2(T_all, a, b, c, d), 'r-.', label='$f_N(t) = b$')
                plt.legend(loc='upper right')
                plt.xlim(0, )
                plt.show()

        if len(L) > 0:
            zeta_one = np.sqrt(1 / (1 + (np.pi / np.log((D[0] - 0.5 * floor) / (-B[0] - 0.5 * floor))) ** 2))
            if params == False:
                mu_one = np.nan
            else:
                mu_one = zeta_one * 2 * np.sqrt(mass * spring)
            if len(L) == 1:
                mu_opt, zeta_opt, mu_fit, zeta_fit = np.nan, np.nan, np.nan, np.nan

    # Coulomb damping
    elif damping_type == 'coulomb':
        if len(L) > 1:
            I_opt = np.argmin(np.abs((L - floor) / ((L[0] - floor)) - 0.3299))

            def func1(data, a, b, c, d):
                return -a * data + b + d

            def func2(data, a, b, c, d):
                return b + 0 * data

            b_guess = 0.01 * np.max(L_all)
            m_guess = np.max(L_all) / np.max(T_all[L_all > 0.3299 * np.max(L_all)])
            d_guess = np.max(L_all)
            initial_guess = [m_guess, b_guess, 0.0, d_guess]
            parameters = fit_two_curves(T_all, L_all, func1, func2, initial_guess)
            a, b, c, d = parameters
            zeta_fit = 0.5 * a
            zeta_opt = (L[0] - L[I_opt]) / (2 * (T_B[I_opt] - T_B[0]))

            if params == False:
                mu_fit = np.nan
                mu_opt = np.nan
            else:
                omega_n = np.sqrt(spring / mass)
                mu_opt = spring * (L[0] - L[I_opt]) / (8 * N * (I_opt - 0))
                zeta_opt = 2 * mu_opt * N * omega_n / (np.pi * spring)
                mu_fit = zeta_fit * np.pi * spring / (2 * N * np.sqrt(spring / mass))

            if plotting:
                plt.figure(figsize=(8, 3))
                plt.xlabel('$t$', size=15)
                plt.ylabel('$L$', size=15)
                plt.plot(T_all, L_all, 'k.', label='L')
                t_plotting = np.linspace(min(T_all), max(T_all), 500)
                plt.plot(t_plotting, func1(t_plotting, a, b, c, d), 'b--', label='$f_S(t) = -mt+c+b$')
                plt.plot(T_all, func2(T_all, a, b, c, d), 'r-.', label='$f_N(t) = b$')
                plt.xlim(0, )
                plt.legend(loc='upper right')
                plt.show()

        if len(L) > 0:
            if params == False:
                zeta_one = (B[0] + D[0]) / (T_B[0] - T_D[0])
                mu_one = np.nan
            else:
                omega_n = np.sqrt(spring / mass)
                zeta_one = -omega_n * (B[0] + D[0]) / (np.pi)
                mu_one = zeta_one * np.pi * spring / (2 * N * np.sqrt(spring / mass))
            if len(L) == 1:
                mu_opt, zeta_opt, mu_fit, zeta_fit = np.nan, np.nan, np.nan, np.nan

    # Quadratic damping
    elif damping_type == 'quadratic':
        zeta_one, mu_one = 1, 1
        n = len(L)
        if n > 0:
            if n > 4:
                n = 4
            mu_q_opts = []
            zeta_q_opts = []
            x = np.linspace(0.0, 10, 2000)
            for it in range(n):
                V_i = B[it] + floor / 2
                P_i = D[it] - floor / 2
                L_i = L[it] - floor
                output = (L_i - (1 / (2 * x)) * np.log((2 * x * V_i - 1) / (2 * x * P_i - 1))) ** 2
                min_indice = np.nanargmin(output)
                zeta_q_opt = x[min_indice]
                zeta_q_opts.append(zeta_q_opt)

                if params == False:
                    mu_q_opt = np.nan
                    mu_q_opts.append(mu_q_opt)
                else:
                    output = (L_i - (mass / (2 * x)) * np.log((2 * x * V_i - mass) / (2 * x * P_i - mass))) ** 2
                    min_indice = np.nanargmin(output)
                    mu_q_opt = zeta_q_opt * mass
                    mu_q_opts.append(mu_q_opt)
            zeta_opt = np.nanmedian(zeta_q_opts)
            zeta_one = zeta_q_opts[0]
            mu_opt = np.nanmedian(mu_q_opts)
            mu_one = mu_q_opts[0]
        else:
            print('Error: insufficient time series for quadratic damping analysis')
            zeta_opt, zeta_one, mu_opt, mu_one = np.nan, np.nan, np.nan, np.nan
        zeta_fit = np.nan
        mu_fit = np.nan

    damping_results = {'zeta_opt': zeta_opt, 'zeta_fit': zeta_fit, 'zeta_one': zeta_one,
                       'mu_opt': mu_opt, 'mu_fit': mu_fit, 'mu_one': mu_one}
    return damping_results


def cutoff_from_lifetimes(L, len_ts, alpha, sigma):
    """Calculate significance cutoff for persistence features."""
    if len(L) == 0:
        cutoff = 0
    else:
        if sigma == False:
            mu_L = np.median(L)
            cutoff = 1.923 * mu_L * erfinv(2 * (1 - np.sqrt(alpha)) ** (1 / len_ts) - 1)
        else:
            cutoff = 2 ** (3 / 2) * sigma * erfinv(2 * (1 - np.sqrt(alpha)) ** (1 / len_ts) - 1)

        if cutoff > 0.3211 * max(L):
            cutoff = 0.3211 * max(L)

    return cutoff


def floor_from_lifetimes(L, t, L_sig, t_sig, cutoff, sigma):
    """Calculate noise floor for persistence features."""
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
    if cutoff == 0.3211 * max(L):
        floor = 0

    return floor


def damping_constant(t, ts, damping='viscous', params=False, sigma=False, alpha=0.001, noise_comp=True, plotting=False):
    """
    Main function to estimate damping parameters from time series.

    Parameters:
    -----------
    t : array - time vector
    ts : array - time series data (displacement/angle)
    damping : str - 'viscous', 'coulomb', or 'quadratic'
    params : list or False - [mass, N, spring] system parameters
    sigma : float or False - noise standard deviation (if known)
    alpha : float - significance level for cutoff
    noise_comp : bool - whether to apply noise floor compensation
    plotting : bool - whether to generate plots

    Returns:
    --------
    results : dict - estimated damping parameters and persistence features
    """
    if damping == 'quadratic' and np.abs(np.mean(ts)) > np.max(ts) - np.min(ts):
        print('Warning: for quadratic damping, time series should be zero-meaned.')

    feature_ind_1, feature_ind_2, persistenceDgm = Persistence0D(ts, 'localMin', edges=False)
    B = np.flip(persistenceDgm.T[0], axis=0)
    D = np.flip(persistenceDgm.T[1], axis=0)
    L = D - B
    I_B = np.array(feature_ind_1.astype(int)).T[0]
    I_D = np.array(feature_ind_2.astype(int)).T[0]
    I_D[I_D == len(ts)] = I_D[I_D == len(ts)] - 1
    T_B = np.flip(t[I_B], axis=0)
    T_D = np.flip(t[I_D], axis=0)
    cutoff = cutoff_from_lifetimes(L, len(ts), alpha, sigma)

    I_insig, I_sig = np.argwhere(L <= cutoff).T[0], np.argwhere(L > cutoff).T[0]
    L_sig, B_sig, D_sig, t_sig_B, t_sig_D = L[I_sig], B[I_sig], D[I_sig], T_B[I_sig], T_D[I_sig]
    L_noise, t_noise = L[I_insig], T_B[I_insig]
    I_sort = np.argsort(t_sig_B)
    L_sig, B_sig, D_sig, t_sig_B, t_sig_D = L_sig[I_sort], B_sig[I_sort], D_sig[I_sort], t_sig_B[I_sort], t_sig_D[I_sort]

    if noise_comp == True:
        floor = floor_from_lifetimes(L, t, L_sig, t_sig_B, cutoff, sigma)
    else:
        floor = 0

    damping_results = damping_param_estimation(damping_type=damping, L=L_sig, B=B_sig,
                                               D=D_sig, T_B=t_sig_B, T_D=t_sig_D, floor=floor, L_all=L,
                                               T_all=T_B, params=params, plotting=plotting)
    if plotting:
        gs = gridspec.GridSpec(2, 1)
        plt.figure(figsize=(10, 6))

        ax = plt.subplot(gs[0, 0])
        plt.ylabel('$\\theta(t)$', size=15)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.plot(t, ts, 'k', alpha=1, linewidth=1.25)
        plt.xlim(0, max(t))

        ax = plt.subplot(gs[1, 0])
        plt.xlabel('$t$', size=15)
        plt.ylabel('$L$', size=15)
        plt.plot(T_B, L, 'k.')
        plt.plot(t_noise, L_noise, 'r.', alpha=0.8, label='$L_N$ (noise)')
        plt.plot(t_sig_B, L_sig, 'bd', label='$L_F$ (signal)')
        plt.plot([0, max(T_B)], [floor, floor], 'k', label=f'Floor = {floor:.4f}')
        plt.plot([0, max(T_B)], [cutoff, cutoff], 'k--', label=f'Cutoff = {cutoff:.4f}')
        plt.legend(loc='upper right')
        plt.xlim(0, max(t))
        plt.subplots_adjust(hspace=0.03)
        plt.show()

    results = {'damping_params': damping_results, 'floor': floor, 'cutoff': cutoff,
               'L_sig': L_sig, 'B_sig': B_sig, 'D_sig': D_sig, 't_B': t_sig_B, 't_D': t_sig_D,
               'L': L, 'B': B, 'D': D, 'T_B': T_B, 'T_D': T_D}
    return results


# =============================================================================
# NONLINEAR PENDULUM SIMULATION
# =============================================================================

def nonlinear_pendulum_ode(t, y, k_th, Om, qh, qv, zeta, mu_c, mu_q):
    """
    Nonlinear pendulum equation of motion:
    θ̈ + 2ζθ̇ + μ_c·sign(θ̇) + μ_q·θ̇|θ̇| + k_θ·θ - cos(θ) + Ω²sin(Ωt)(q_h·sin(θ) - q_v·cos(θ)) = 0

    Parameters:
    -----------
    t : float - time
    y : array - state vector [theta, theta_dot]
    k_th : float - torsional spring stiffness
    Om : float - excitation frequency
    qh : float - horizontal excitation amplitude
    qv : float - vertical excitation amplitude
    zeta : float - viscous damping ratio
    mu_c : float - Coulomb friction coefficient
    mu_q : float - quadratic damping coefficient
    """
    theta = y[0]
    theta_dot = y[1]

    # Smooth sign function for Coulomb damping (avoid numerical issues)
    epsilon = 1e-6
    sign_smooth = np.tanh(theta_dot / epsilon)

    # Damping terms
    F_viscous = 2 * zeta * theta_dot
    F_coulomb = mu_c * sign_smooth
    F_quadratic = mu_q * theta_dot * np.abs(theta_dot)
    F_damping = F_viscous + F_coulomb + F_quadratic

    # Equation of motion
    A = F_damping + k_th * theta - np.cos(theta) + Om**2 * np.sin(Om * t) * (qh * np.sin(theta) - qv * np.cos(theta))

    dydt = np.zeros(2)
    dydt[0] = theta_dot
    dydt[1] = -A

    return dydt


def simulate_pendulum(damping_type='viscous', zeta=0.05, mu_c=0.03, mu_q=0.05,
                      k_th=20, qh=0, qv=0, Om=5,
                      theta0=120, theta_dot0=0,
                      tf_cycles=100, dt=0.01, noise_std=0.0):
    """
    Simulate nonlinear pendulum with specified damping.

    Parameters:
    -----------
    damping_type : str - 'viscous', 'coulomb', 'quadratic', or 'combined'
    zeta : float - viscous damping ratio
    mu_c : float - Coulomb friction coefficient
    mu_q : float - quadratic damping coefficient
    k_th : float - torsional spring stiffness
    qh, qv : float - horizontal/vertical excitation amplitudes
    Om : float - excitation frequency
    theta0 : float - initial angle (degrees)
    theta_dot0 : float - initial angular velocity (degrees/s)
    tf_cycles : int - number of oscillation cycles to simulate
    dt : float - time step
    noise_std : float - standard deviation of additive noise

    Returns:
    --------
    t : array - time vector
    theta : array - angle time series
    theta_dot : array - angular velocity time series
    params_used : dict - actual damping parameters used
    """
    # Set damping parameters based on type
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

    # Time settings
    T = 2 * np.pi / Om
    tf = tf_cycles * T
    t_span = (0, tf)
    t_eval = np.arange(0, tf, dt)

    # Initial conditions (convert to radians)
    y0 = [np.radians(theta0), np.radians(theta_dot0)]

    # Solve ODE
    sol = solve_ivp(
        lambda t, y: nonlinear_pendulum_ode(t, y, k_th, Om, qh, qv, zeta_use, mu_c_use, mu_q_use),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-9
    )

    t = sol.t
    theta = sol.y[0]  # in radians
    theta_dot = sol.y[1]

    # Add noise if specified
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


def run_inverse_estimation(damping_type='viscous', plotting=True, **sim_kwargs):
    """
    Run complete inverse parameter estimation pipeline.

    1. Simulate pendulum with known parameters
    2. Apply topological damping estimation
    3. Compare estimated vs true parameters
    """
    print(f"\n{'='*60}")
    print(f"INVERSE PARAMETER ESTIMATION: {damping_type.upper()} DAMPING")
    print('='*60)

    # Simulate pendulum
    t, theta, theta_dot, params_used = simulate_pendulum(damping_type=damping_type, **sim_kwargs)

    print(f"\nTrue Parameters:")
    print(f"  zeta = {params_used['zeta']:.4f}")
    print(f"  mu_c = {params_used['mu_c']:.4f}")
    print(f"  mu_q = {params_used['mu_q']:.4f}")
    print(f"  k_th = {params_used['k_th']:.4f}")

    # Center the signal (remove mean offset for better estimation)
    theta_centered = theta - np.mean(theta)

    # Apply topological damping estimation
    results = damping_constant(
        t, theta_centered,
        damping=damping_type,
        params=False,  # Don't use system params, estimate from signal only
        alpha=0.01,
        plotting=plotting
    )

    print(f"\nEstimated Parameters (Topological Method):")
    print(f"  zeta_opt = {results['damping_params']['zeta_opt']:.4f}")
    print(f"  zeta_fit = {results['damping_params']['zeta_fit']:.4f}")
    print(f"  zeta_one = {results['damping_params']['zeta_one']:.4f}")

    # Calculate estimation error
    if damping_type == 'viscous':
        true_val = params_used['zeta']
        est_val = results['damping_params']['zeta_opt']
        error_pct = abs(est_val - true_val) / true_val * 100
        print(f"\nEstimation Error: {error_pct:.2f}%")

    return t, theta, results, params_used


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Use non-interactive backend for matplotlib
    import matplotlib
    matplotlib.use('Agg')

    print("\n" + "="*70)
    print("NONLINEAR PENDULUM: INVERSE PARAMETER ESTIMATION")
    print("Using Topological Signal Processing")
    print("="*70)

    # Test 1: Viscous damping
    t_v, theta_v, results_v, params_v = run_inverse_estimation(
        damping_type='viscous',
        zeta=0.05,
        theta0=120,
        tf_cycles=50,
        dt=0.01,
        noise_std=0.001,
        plotting=False
    )

    # Test 2: Coulomb damping
    t_c, theta_c, results_c, params_c = run_inverse_estimation(
        damping_type='coulomb',
        mu_c=0.03,
        theta0=120,
        tf_cycles=50,
        dt=0.01,
        noise_std=0.001,
        plotting=False
    )

    # Test 3: Quadratic damping
    t_q, theta_q, results_q, params_q = run_inverse_estimation(
        damping_type='quadratic',
        mu_q=0.05,
        theta0=120,
        tf_cycles=50,
        dt=0.01,
        noise_std=0.001,
        plotting=False
    )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nDamping Type    | True Value | Estimated  | Error")
    print("-" * 55)
    print(f"Viscous (zeta)  | {params_v['zeta']:.4f}     | {results_v['damping_params']['zeta_opt']:.4f}     | {abs(results_v['damping_params']['zeta_opt']-params_v['zeta'])/params_v['zeta']*100:.1f}%")
    print(f"Coulomb (mu_c)  | {params_c['mu_c']:.4f}     | (see zeta) | N/A")
    print(f"Quadratic (mu_q)| {params_q['mu_q']:.4f}     | {results_q['damping_params']['zeta_opt']:.4f}     | N/A")
