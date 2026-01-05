"""
Least Squares Method for Damping Parameter Estimation
======================================================

The simplest and most direct approach: rearrange the ODE into a linear
system Ax = b and solve using least squares.

Pendulum ODE:
    θ̈ + 2ζθ̇ + μ_c·tanh(θ̇/ε) + μ_q·θ̇|θ̇| + k_θ·θ - cos(θ) = 0

Rearrange:
    θ̈ + k_θ·θ - cos(θ) = -F_damping(θ̇)

Define residual:
    b = θ̈ + k_θ·θ - cos(θ)

Then for each damping type:
    Viscous:   b = -2ζ·θ̇           → solve for 2ζ
    Coulomb:   b = -μ_c·tanh(θ̇/ε)  → solve for μ_c
    Quadratic: b = -μ_q·θ̇|θ̇|      → solve for μ_q

This is linear regression: A·x = b, solved via numpy.linalg.lstsq
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def pendulum_ode(t, y, k_th, zeta, mu_c, mu_q, epsilon=0.1):
    """Nonlinear pendulum ODE with mixed damping."""
    theta, theta_dot = y

    # Smoothed sign function for Coulomb friction
    sign_smooth = np.tanh(theta_dot / epsilon)

    # Damping force
    F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)

    # Equation of motion: θ̈ = -k_θ·θ + cos(θ) - F_damping
    theta_ddot = -k_th * theta + np.cos(theta) - F_damping

    return [theta_dot, theta_ddot]


def generate_data(k_th, zeta, mu_c, mu_q, theta0=0.3, t_span=(0, 10), n_points=5000):
    """Generate pendulum trajectory data."""
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        pendulum_ode,
        t_span,
        [theta0, 0.0],
        args=(k_th, zeta, mu_c, mu_q),
        method='RK45',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12
    )

    return sol.t, sol.y[0], sol.y[1]


def compute_acceleration(t, theta, theta_dot, method='savgol'):
    """
    Compute θ̈ from θ̇ using numerical differentiation.

    Methods:
    - 'savgol': Savitzky-Golay filter (smoothed derivative)
    - 'central': Central difference
    - 'spline': Spline interpolation + derivative
    """
    dt = t[1] - t[0]

    if method == 'savgol':
        # Savitzky-Golay filter: smooth derivative
        # Window size should be odd, polynomial order < window
        window = min(51, len(theta_dot) // 10)
        if window % 2 == 0:
            window += 1
        window = max(5, window)

        theta_ddot = savgol_filter(theta_dot, window, 3, deriv=1, delta=dt)

    elif method == 'central':
        # Central difference: (f(x+h) - f(x-h)) / 2h
        theta_ddot = np.gradient(theta_dot, dt, edge_order=2)

    elif method == 'spline':
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(t, theta_dot, s=0)
        theta_ddot = spline.derivative()(t)

    return theta_ddot


def least_squares_estimate(t, theta, theta_dot, theta_ddot, k_th, damping_type, epsilon=0.1):
    """
    Estimate damping parameter using ordinary least squares.

    The ODE gives us:
        θ̈ + k_θ·θ - cos(θ) = -F_damping

    So: b = θ̈ + k_θ·θ - cos(θ) = -F_damping

    For each damping type, we have:
        Viscous:   -F = -2ζ·θ̇           → A = -θ̇, x = 2ζ
        Coulomb:   -F = -μ_c·tanh(θ̇/ε)  → A = -tanh(θ̇/ε), x = μ_c
        Quadratic: -F = -μ_q·θ̇|θ̇|      → A = -θ̇|θ̇|, x = μ_q
    """
    # Compute the residual (what the damping should equal)
    b = theta_ddot + k_th * theta - np.cos(theta)

    # Build the feature matrix based on damping type
    if damping_type == 'viscous':
        # b = -2ζ·θ̇, so A = -θ̇
        A = -theta_dot.reshape(-1, 1)

    elif damping_type == 'coulomb':
        # b = -μ_c·tanh(θ̇/ε), so A = -tanh(θ̇/ε)
        A = -np.tanh(theta_dot / epsilon).reshape(-1, 1)

    elif damping_type == 'quadratic':
        # b = -μ_q·θ̇|θ̇|, so A = -θ̇|θ̇|
        A = -(theta_dot * np.abs(theta_dot)).reshape(-1, 1)

    # Solve least squares: A·x = b
    # Using normal equations: x = (A^T A)^(-1) A^T b
    # Or numpy's lstsq which is more numerically stable
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Extract the parameter
    if damping_type == 'viscous':
        # x[0] = 2ζ, so ζ = x[0]/2
        param = x[0] / 2
    else:
        param = x[0]

    return param


def weighted_least_squares_estimate(t, theta, theta_dot, theta_ddot, k_th, damping_type, epsilon=0.1):
    """
    Weighted least squares with weights based on signal strength.

    Weight points by |θ̇| to emphasize regions where damping effect is strongest.
    This reduces the impact of near-zero velocity points where the signal-to-noise is poor.
    """
    b = theta_ddot + k_th * theta - np.cos(theta)

    # Compute weights based on velocity magnitude
    weights = np.abs(theta_dot) + 1e-6  # Add small constant to avoid division by zero
    weights = weights / np.max(weights)  # Normalize

    # Build feature matrix
    if damping_type == 'viscous':
        A = -theta_dot.reshape(-1, 1)
    elif damping_type == 'coulomb':
        A = -np.tanh(theta_dot / epsilon).reshape(-1, 1)
    elif damping_type == 'quadratic':
        A = -(theta_dot * np.abs(theta_dot)).reshape(-1, 1)

    # Apply weights
    W = np.diag(weights)

    # Weighted least squares: (A^T W A)^(-1) A^T W b
    AtWA = A.T @ W @ A
    AtWb = A.T @ W @ b
    x = np.linalg.solve(AtWA, AtWb)

    if damping_type == 'viscous':
        param = x[0] / 2
    else:
        param = x[0]

    return param


def iteratively_reweighted_least_squares(t, theta, theta_dot, theta_ddot, k_th, damping_type,
                                          epsilon=0.1, n_iter=10):
    """
    Iteratively Reweighted Least Squares (IRLS) for robust estimation.

    Uses Huber weights to downweight outliers.
    """
    b = theta_ddot + k_th * theta - np.cos(theta)

    if damping_type == 'viscous':
        A = -theta_dot.reshape(-1, 1)
    elif damping_type == 'coulomb':
        A = -np.tanh(theta_dot / epsilon).reshape(-1, 1)
    elif damping_type == 'quadratic':
        A = -(theta_dot * np.abs(theta_dot)).reshape(-1, 1)

    # Initial estimate with OLS
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    for _ in range(n_iter):
        # Compute residuals
        residuals = b - A @ x

        # Huber weights
        c = 1.345 * np.std(residuals)  # Tuning constant
        weights = np.where(np.abs(residuals) <= c, 1.0, c / np.abs(residuals))

        # Weighted least squares
        W = np.diag(weights)
        AtWA = A.T @ W @ A
        AtWb = A.T @ W @ b
        x = np.linalg.solve(AtWA, AtWb)

    if damping_type == 'viscous':
        param = x[0] / 2
    else:
        param = x[0]

    return param


def total_least_squares_estimate(t, theta, theta_dot, theta_ddot, k_th, damping_type, epsilon=0.1):
    """
    Total Least Squares (TLS) using SVD.

    Accounts for errors in both A and b (errors-in-variables model).
    More appropriate when there's noise in all measurements.
    """
    b = theta_ddot + k_th * theta - np.cos(theta)

    if damping_type == 'viscous':
        A = -theta_dot.reshape(-1, 1)
    elif damping_type == 'coulomb':
        A = -np.tanh(theta_dot / epsilon).reshape(-1, 1)
    elif damping_type == 'quadratic':
        A = -(theta_dot * np.abs(theta_dot)).reshape(-1, 1)

    # Augmented matrix [A | b]
    C = np.hstack([A, b.reshape(-1, 1)])

    # SVD
    U, S, Vt = np.linalg.svd(C, full_matrices=True)

    # TLS solution from last row of V
    V = Vt.T
    x = -V[:-1, -1] / V[-1, -1]

    if damping_type == 'viscous':
        param = x[0] / 2
    else:
        param = x[0]

    return param


def analytical_acceleration(t, theta, theta_dot, k_th, zeta, mu_c, mu_q, epsilon=0.1):
    """
    Compute θ̈ analytically from the ODE (for validation/comparison).
    This gives us the "true" acceleration without numerical differentiation errors.
    """
    sign_smooth = np.tanh(theta_dot / epsilon)
    F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)
    theta_ddot = -k_th * theta + np.cos(theta) - F_damping
    return theta_ddot


def run_estimation(damping_type, true_value, k_th=20.0, theta0=0.3,
                   use_analytical_accel=False, diff_method='savgol'):
    """
    Run least squares estimation for a specific damping type.
    """
    print(f"\n{'='*60}")
    print(f"LEAST SQUARES ESTIMATION - {damping_type.upper()} DAMPING")
    print(f"{'='*60}")

    # Set up parameters based on damping type
    if damping_type == 'viscous':
        zeta, mu_c, mu_q = true_value, 0.0, 0.0
        param_name = 'zeta'
    elif damping_type == 'coulomb':
        zeta, mu_c, mu_q = 0.0, true_value, 0.0
        param_name = 'mu_c'
    elif damping_type == 'quadratic':
        zeta, mu_c, mu_q = 0.0, 0.0, true_value
        param_name = 'mu_q'

    # Generate data with high sampling rate for accurate differentiation
    t, theta, theta_dot = generate_data(k_th, zeta, mu_c, mu_q, theta0,
                                         t_span=(0, 15), n_points=10000)

    # Compute acceleration
    if use_analytical_accel:
        theta_ddot = analytical_acceleration(t, theta, theta_dot, k_th, zeta, mu_c, mu_q)
        print(f"Using analytical acceleration (for validation)")
    else:
        theta_ddot = compute_acceleration(t, theta, theta_dot, method=diff_method)
        print(f"Using numerical differentiation ({diff_method})")

    # Trim edges to avoid differentiation artifacts
    trim = 100
    t_trim = t[trim:-trim]
    theta_trim = theta[trim:-trim]
    theta_dot_trim = theta_dot[trim:-trim]
    theta_ddot_trim = theta_ddot[trim:-trim]

    # Run different least squares methods
    results = {}

    # 1. Ordinary Least Squares (OLS)
    print(f"\nMethod 1: Ordinary Least Squares (OLS)")
    est_ols = least_squares_estimate(t_trim, theta_trim, theta_dot_trim, theta_ddot_trim,
                                      k_th, damping_type)
    err_ols = abs(est_ols - true_value) / true_value * 100
    print(f"  Estimated {param_name}: {est_ols:.6f}")
    print(f"  True {param_name}: {true_value:.6f}")
    print(f"  Error: {err_ols:.4f}%")
    results['OLS'] = (est_ols, err_ols)

    # 2. Weighted Least Squares (WLS)
    print(f"\nMethod 2: Weighted Least Squares (WLS)")
    est_wls = weighted_least_squares_estimate(t_trim, theta_trim, theta_dot_trim, theta_ddot_trim,
                                               k_th, damping_type)
    err_wls = abs(est_wls - true_value) / true_value * 100
    print(f"  Estimated {param_name}: {est_wls:.6f}")
    print(f"  Error: {err_wls:.4f}%")
    results['WLS'] = (est_wls, err_wls)

    # 3. Iteratively Reweighted Least Squares (IRLS)
    print(f"\nMethod 3: Iteratively Reweighted Least Squares (IRLS)")
    est_irls = iteratively_reweighted_least_squares(t_trim, theta_trim, theta_dot_trim, theta_ddot_trim,
                                                     k_th, damping_type)
    err_irls = abs(est_irls - true_value) / true_value * 100
    print(f"  Estimated {param_name}: {est_irls:.6f}")
    print(f"  Error: {err_irls:.4f}%")
    results['IRLS'] = (est_irls, err_irls)

    # 4. Total Least Squares (TLS)
    print(f"\nMethod 4: Total Least Squares (TLS)")
    est_tls = total_least_squares_estimate(t_trim, theta_trim, theta_dot_trim, theta_ddot_trim,
                                            k_th, damping_type)
    err_tls = abs(est_tls - true_value) / true_value * 100
    print(f"  Estimated {param_name}: {est_tls:.6f}")
    print(f"  Error: {err_tls:.4f}%")
    results['TLS'] = (est_tls, err_tls)

    # Find best method
    best_method = min(results.keys(), key=lambda k: results[k][1])
    best_est, best_err = results[best_method]

    print(f"\n{'='*40}")
    print(f"BEST METHOD: {best_method}")
    print(f"  Estimated {param_name}: {best_est:.6f}")
    print(f"  True {param_name}: {true_value:.6f}")
    print(f"  Error: {best_err:.4f}%")
    print(f"{'='*40}")

    return best_est, best_err, results, (t, theta, theta_dot, theta_ddot)


def create_plots(all_results, all_data):
    """Create visualization of results."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    damping_types = ['viscous', 'coulomb', 'quadratic']
    true_values = [0.05, 0.03, 0.05]
    param_names = ['ζ', 'μ_c', 'μ_q']

    for i, (damping_type, true_val, param_name) in enumerate(zip(damping_types, true_values, param_names)):
        t, theta, theta_dot, theta_ddot = all_data[damping_type]
        results = all_results[damping_type]

        # Top row: Time series
        ax1 = axes[0, i]
        ax1.plot(t, theta, 'b-', linewidth=0.8, label='θ(t)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('θ (rad)')
        ax1.set_title(f'{damping_type.capitalize()} Damping')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom row: Method comparison
        ax2 = axes[1, i]
        methods = list(results.keys())
        errors = [results[m][1] for m in methods]
        colors = ['green' if e < 0.1 else 'orange' if e < 0.5 else 'red' for e in errors]

        bars = ax2.bar(methods, errors, color=colors, edgecolor='black')
        ax2.axhline(y=0.1, color='green', linestyle='--', linewidth=1.5, label='0.1% target')
        ax2.set_ylabel('Error (%)')
        ax2.set_title(f'{param_name} Estimation Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, err in zip(bars, errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{err:.3f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'least_squares_results.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to {os.path.join(SCRIPT_DIR, 'least_squares_results.png')}")


def main():
    """Main function to run all estimations."""
    print("="*70)
    print("LEAST SQUARES METHOD FOR DAMPING PARAMETER ESTIMATION")
    print("Simple, direct approach: rearrange ODE to Ax = b, solve with LSM")
    print("="*70)

    # True parameters
    k_th = 20.0
    true_zeta = 0.05
    true_mu_c = 0.03
    true_mu_q = 0.05

    all_results = {}
    all_data = {}
    final_results = []

    # Test each damping type
    for damping_type, true_val in [('viscous', true_zeta),
                                    ('coulomb', true_mu_c),
                                    ('quadratic', true_mu_q)]:
        best_est, best_err, results, data = run_estimation(
            damping_type, true_val, k_th,
            use_analytical_accel=False,  # Use numerical differentiation
            diff_method='savgol'
        )
        all_results[damping_type] = results
        all_data[damping_type] = data
        final_results.append((damping_type, true_val, best_est, best_err))

    # Create plots
    create_plots(all_results, all_data)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\n{'Damping Type':<15} {'True':<12} {'Estimated':<12} {'Error':<12} {'Status'}")
    print("-" * 65)

    all_below_target = True
    for damping_type, true_val, est_val, err in final_results:
        status = "PASS" if err < 0.1 else "FAIL"
        if err >= 0.1:
            all_below_target = False
        print(f"{damping_type:<15} {true_val:<12.6f} {est_val:<12.6f} {err:<10.4f}% {status}")

    print("\n" + "="*70)
    if all_below_target:
        print("SUCCESS: All errors are below 0.1%!")
    else:
        print("Some errors exceed 0.1% target. Optimization needed.")
    print("="*70)

    return all_below_target, final_results


if __name__ == "__main__":
    success, results = main()
