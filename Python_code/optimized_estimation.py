"""
Optimized Damping Parameter Estimation for Nonlinear Pendulum

The key insight: Since the pendulum equation is nonlinear (contains -cos(θ)),
standard damping estimation formulas derived for linear oscillators do NOT apply.

Solution: Use optimization-based system identification:
1. Generate "observed" data with known (true) parameters
2. Define objective function: difference between simulated and observed envelopes
3. Use optimization to find parameters that minimize the difference

This approach works for ANY nonlinear system without needing analytical formulas.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks, hilbert
from scipy.optimize import minimize_scalar, minimize, differential_evolution
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

import os
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nonlinear_pendulum_ode(t, y, k_th, zeta, mu_c, mu_q):
    """Nonlinear pendulum ODE (free response)."""
    theta, theta_dot = y

    # Smooth sign function to avoid numerical issues
    epsilon = 1e-6
    sign_smooth = np.tanh(theta_dot / epsilon)

    # Damping forces
    F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)

    # Total acceleration: θ̈ = -(F_damping + k_θ·θ - cos(θ))
    A = F_damping + k_th * theta - np.cos(theta)

    return [theta_dot, -A]


def simulate(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt=0.002):
    """Simulate pendulum and return time and angle."""
    y0 = [np.radians(theta0_deg), 0]
    t_eval = np.arange(0, t_final, dt)

    sol = solve_ivp(
        lambda t, y: nonlinear_pendulum_ode(t, y, k_th, zeta, mu_c, mu_q),
        (0, t_final), y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10
    )

    return sol.t, sol.y[0]


def extract_envelope(t, theta):
    """Extract amplitude envelope using Hilbert transform and peak detection."""
    # Hilbert transform envelope
    analytic = hilbert(theta)
    env_hilbert = np.abs(analytic)

    # Peak-based envelope (more robust for decaying oscillations)
    peaks_pos, _ = find_peaks(theta, distance=10)
    peaks_neg, _ = find_peaks(-theta, distance=10)

    if len(peaks_pos) < 3:
        return t, env_hilbert, None, None

    t_peaks = t[peaks_pos]
    amp_peaks = theta[peaks_pos]

    return t, env_hilbert, t_peaks, amp_peaks


def envelope_error(param_value, param_type, t_obs, env_obs, k_th, theta0_deg, t_final, dt):
    """
    Compute error between observed and simulated envelopes.
    This is the objective function for optimization.
    """
    if param_type == 'viscous':
        zeta, mu_c, mu_q = param_value, 0, 0
    elif param_type == 'coulomb':
        zeta, mu_c, mu_q = 0, param_value, 0
    elif param_type == 'quadratic':
        zeta, mu_c, mu_q = 0, 0, param_value
    else:
        return 1e10

    try:
        t_sim, theta_sim = simulate(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt)
        _, env_sim, _, _ = extract_envelope(t_sim, theta_sim)

        # Interpolate to match time points
        env_sim_interp = np.interp(t_obs, t_sim, env_sim)

        # MSE on log-scale (emphasizes decay rate)
        valid = (env_obs > 0.01 * np.max(env_obs)) & (env_sim_interp > 0.01 * np.max(env_sim_interp))
        if np.sum(valid) < 10:
            return 1e10

        log_obs = np.log(env_obs[valid] + 1e-10)
        log_sim = np.log(env_sim_interp[valid] + 1e-10)

        error = np.mean((log_obs - log_sim) ** 2)
        return error

    except Exception as e:
        return 1e10


def estimate_parameter_optimization(t_obs, theta_obs, param_type, k_th, theta0_deg, t_final, dt,
                                     true_value=None, search_range=None):
    """
    Estimate damping parameter using optimization.

    Args:
        t_obs, theta_obs: Observed time series
        param_type: 'viscous', 'coulomb', or 'quadratic'
        k_th: Torsional stiffness
        theta0_deg: Initial angle
        true_value: True parameter value (for setting search range)
        search_range: Explicit (min, max) range for parameter search

    Returns:
        Dictionary with estimation results
    """
    _, env_obs, t_peaks, amp_peaks = extract_envelope(t_obs, theta_obs)

    # Set search range
    if search_range is None:
        if true_value is not None:
            # Search around true value (for testing)
            search_range = (true_value * 0.1, true_value * 3.0)
        else:
            # Default ranges
            if param_type == 'viscous':
                search_range = (0.001, 0.5)
            elif param_type == 'coulomb':
                search_range = (0.001, 0.3)
            elif param_type == 'quadratic':
                search_range = (0.001, 0.3)

    # Optimization using bounded scalar search (fast)
    result = minimize_scalar(
        lambda p: envelope_error(p, param_type, t_obs, env_obs, k_th, theta0_deg, t_final, dt),
        bounds=search_range,
        method='bounded',
        options={'xatol': 1e-6}
    )

    estimated = result.x

    return {
        'estimated': estimated,
        'true': true_value,
        'error_pct': abs(estimated - true_value) / true_value * 100 if true_value else None,
        't_obs': t_obs,
        'theta_obs': theta_obs,
        'env_obs': env_obs,
        't_peaks': t_peaks,
        'amp_peaks': amp_peaks,
        'optimization_result': result
    }


def estimate_all_parameters_joint(t_obs, theta_obs, k_th, theta0_deg, t_final, dt,
                                   true_zeta=None, true_mu_c=None, true_mu_q=None,
                                   damping_type='all'):
    """
    Joint estimation of multiple damping parameters.

    For mixed damping, uses differential evolution for global optimization.
    """
    _, env_obs, _, _ = extract_envelope(t_obs, theta_obs)

    def objective(params):
        if damping_type == 'viscous':
            zeta, mu_c, mu_q = params[0], 0, 0
        elif damping_type == 'coulomb':
            zeta, mu_c, mu_q = 0, params[0], 0
        elif damping_type == 'quadratic':
            zeta, mu_c, mu_q = 0, 0, params[0]
        else:  # all three
            zeta, mu_c, mu_q = params

        try:
            t_sim, theta_sim = simulate(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt)
            _, env_sim, _, _ = extract_envelope(t_sim, theta_sim)

            env_sim_interp = np.interp(t_obs, t_sim, env_sim)

            valid = env_obs > 0.01 * np.max(env_obs)
            if np.sum(valid) < 10:
                return 1e10

            log_obs = np.log(env_obs[valid] + 1e-10)
            log_sim = np.log(env_sim_interp[valid] + 1e-10)

            return np.mean((log_obs - log_sim) ** 2)
        except:
            return 1e10

    # Bounds for parameters
    if damping_type == 'all':
        bounds = [(0.001, 0.5), (0.001, 0.3), (0.001, 0.3)]
    else:
        bounds = [(0.001, 0.5)]

    # Differential evolution for global optimization
    result = differential_evolution(objective, bounds, seed=42,
                                     maxiter=100, tol=1e-6, workers=1)

    return result.x, result.fun


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("OPTIMIZATION-BASED DAMPING PARAMETER ESTIMATION")
    print("=" * 70)
    print("\nApproach: Direct optimization to match simulated envelope to observed data")
    print("This bypasses the need for analytical formulas (which fail for nonlinear systems)")

    # System parameters
    k_th = 20
    theta0_deg = 30  # Initial angle
    t_final = 60
    dt = 0.002

    TRUE_ZETA = 0.05
    TRUE_MU_C = 0.03
    TRUE_MU_Q = 0.05

    # Add realistic measurement noise
    NOISE_STD = 0.002  # 0.2% of 1 radian - typical sensor noise

    print(f"\nSystem: k_θ = {k_th}, θ₀ = {theta0_deg}°, t_final = {t_final}s")
    print(f"True parameters: ζ = {TRUE_ZETA}, μ_c = {TRUE_MU_C}, μ_q = {TRUE_MU_Q}")
    print(f"Measurement noise: σ = {NOISE_STD} rad ({NOISE_STD*100:.1f}% of 1 rad)")

    np.random.seed(42)  # For reproducibility
    results = {}

    # =========================================================================
    # VISCOUS DAMPING
    # =========================================================================
    print("\n" + "-" * 50)
    print("VISCOUS DAMPING ESTIMATION")
    print("-" * 50)

    # Generate "observed" data with noise
    t_obs, theta_obs = simulate(k_th, TRUE_ZETA, 0, 0, theta0_deg, t_final, dt)
    theta_obs = theta_obs + np.random.normal(0, NOISE_STD, len(theta_obs))

    # Estimate using optimization
    res = estimate_parameter_optimization(
        t_obs, theta_obs, 'viscous', k_th, theta0_deg, t_final, dt,
        true_value=TRUE_ZETA, search_range=(0.001, 0.3)
    )

    print(f"\n  True ζ      = {TRUE_ZETA:.5f}")
    print(f"  Estimated ζ = {res['estimated']:.5f}")
    print(f"  Error       = {res['error_pct']:.2f}%")

    results['viscous'] = res

    # =========================================================================
    # COULOMB DAMPING
    # =========================================================================
    print("\n" + "-" * 50)
    print("COULOMB DAMPING ESTIMATION")
    print("-" * 50)

    t_obs, theta_obs = simulate(k_th, 0, TRUE_MU_C, 0, theta0_deg, t_final, dt)
    theta_obs = theta_obs + np.random.normal(0, NOISE_STD, len(theta_obs))

    res = estimate_parameter_optimization(
        t_obs, theta_obs, 'coulomb', k_th, theta0_deg, t_final, dt,
        true_value=TRUE_MU_C, search_range=(0.001, 0.2)
    )

    print(f"\n  True μ_c      = {TRUE_MU_C:.5f}")
    print(f"  Estimated μ_c = {res['estimated']:.5f}")
    print(f"  Error         = {res['error_pct']:.2f}%")

    results['coulomb'] = res

    # =========================================================================
    # QUADRATIC DAMPING
    # =========================================================================
    print("\n" + "-" * 50)
    print("QUADRATIC DAMPING ESTIMATION")
    print("-" * 50)

    t_obs, theta_obs = simulate(k_th, 0, 0, TRUE_MU_Q, theta0_deg, t_final, dt)
    theta_obs = theta_obs + np.random.normal(0, NOISE_STD, len(theta_obs))

    res = estimate_parameter_optimization(
        t_obs, theta_obs, 'quadratic', k_th, theta0_deg, t_final, dt,
        true_value=TRUE_MU_Q, search_range=(0.001, 0.3)
    )

    print(f"\n  True μ_q      = {TRUE_MU_Q:.5f}")
    print(f"  Estimated μ_q = {res['estimated']:.5f}")
    print(f"  Error         = {res['error_pct']:.2f}%")

    results['quadratic'] = res

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: OPTIMIZATION-BASED ESTIMATION RESULTS")
    print("=" * 70)

    print(f"\n{'Type':<12} {'Parameter':<10} {'True':<12} {'Estimated':<12} {'Error':<10}")
    print("-" * 60)

    for dtype, res in results.items():
        param = 'ζ' if dtype == 'viscous' else ('μ_c' if dtype == 'coulomb' else 'μ_q')
        true_val = res['true']
        est_val = res['estimated']
        err = res['error_pct']
        status = "✓ GOOD" if err < 5 else ("~ OK" if err < 15 else "✗ HIGH")
        print(f"{dtype:<12} {param:<10} {true_val:<12.5f} {est_val:<12.5f} {err:<6.2f}%  {status}")

    avg_error = np.mean([res['error_pct'] for res in results.values()])
    print(f"\nAverage error: {avg_error:.2f}%")

    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    print("\n\nGenerating plots...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for idx, (dtype, res) in enumerate(results.items()):
        t = res['t_obs']
        theta = res['theta_obs']
        env = res['env_obs']
        t_peaks = res['t_peaks']
        amp_peaks = res['amp_peaks']

        param = 'ζ' if dtype == 'viscous' else ('μ_c' if dtype == 'coulomb' else 'μ_q')

        # Left: Time series with envelope
        ax = axes[idx, 0]
        ax.plot(t, np.degrees(theta), 'b-', linewidth=0.5, alpha=0.7, label='θ(t)')
        ax.plot(t, np.degrees(env), 'r-', linewidth=1.5, alpha=0.8, label='Envelope')
        if t_peaks is not None:
            ax.plot(t_peaks, np.degrees(amp_peaks), 'go', markersize=4, label='Peaks')

        ax.set_title(f"{dtype.capitalize()}: True {param} = {res['true']:.4f}, "
                     f"Est = {res['estimated']:.4f} (Error: {res['error_pct']:.1f}%)")
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('θ [deg]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Right: Simulated envelope comparison
        ax = axes[idx, 1]

        # Observed envelope
        ax.semilogy(t, env, 'b-', linewidth=1.5, alpha=0.7, label='Observed')

        # Simulated envelope with estimated parameter
        if dtype == 'viscous':
            t_sim, theta_sim = simulate(k_th, res['estimated'], 0, 0, theta0_deg, t_final, dt)
        elif dtype == 'coulomb':
            t_sim, theta_sim = simulate(k_th, 0, res['estimated'], 0, theta0_deg, t_final, dt)
        else:
            t_sim, theta_sim = simulate(k_th, 0, 0, res['estimated'], theta0_deg, t_final, dt)

        _, env_sim, _, _ = extract_envelope(t_sim, theta_sim)
        ax.semilogy(t_sim, env_sim, 'r--', linewidth=1.5, alpha=0.8, label='Simulated (estimated)')

        ax.set_title(f'{dtype.capitalize()}: Envelope Comparison')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude (log scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Optimization-Based Damping Parameter Estimation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_optimized_estimation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    width = 0.35

    true_vals = [results['viscous']['true'], results['coulomb']['true'], results['quadratic']['true']]
    est_vals = [results['viscous']['estimated'], results['coulomb']['estimated'], results['quadratic']['estimated']]
    errors = [results['viscous']['error_pct'], results['coulomb']['error_pct'], results['quadratic']['error_pct']]

    bars1 = ax.bar(x - width/2, true_vals, width, label='True', color='steelblue')
    bars2 = ax.bar(x + width/2, est_vals, width, label='Estimated', color='coral')

    for bar, err in zip(bars2, errors):
        color = 'green' if err < 5 else ('orange' if err < 15 else 'red')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{err:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

    ax.set_ylabel('Parameter Value')
    ax.set_title('Optimization-Based Estimation: True vs Estimated')
    ax.set_xticks(x)
    ax.set_xticklabels(['Viscous (ζ)', 'Coulomb (μ_c)', 'Quadratic (μ_q)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_optimized_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: fig_optimized_estimation.png, fig_optimized_comparison.png")

    # Check if all errors are reasonable
    all_good = all(res['error_pct'] < 5 for res in results.values())
    if all_good:
        print("\n✓ All estimation errors < 5% - EXCELLENT!")
    elif avg_error < 10:
        print(f"\n~ Average error {avg_error:.1f}% - Good, but can be improved")
    else:
        print(f"\n✗ Average error {avg_error:.1f}% - Needs improvement")

    print("\nDone!")
