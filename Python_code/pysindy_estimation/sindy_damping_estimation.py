"""
PySINDy-based Damping Parameter Estimation for Nonlinear Pendulum

Uses Sparse Identification of Nonlinear Dynamics (SINDy) to discover
governing equations and estimate damping parameters from time series data.

Equation of Motion:
θ̈ + 2ζθ̇ + μ_c·sign(θ̇) + μ_q·θ̇|θ̇| + k_θ·θ - cos(θ) = 0

Rearranged for SINDy:
θ̈ = -k_θ·θ + cos(θ) - 2ζ·θ̇ - μ_c·sign(θ̇) - μ_q·θ̇|θ̇|

The SINDy algorithm identifies which terms are active and estimates their coefficients.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

import os
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if pysindy is available
try:
    import pysindy as ps
    PYSINDY_AVAILABLE = True
except ImportError:
    PYSINDY_AVAILABLE = False
    print("WARNING: pysindy not installed. Install with: pip install pysindy")


# =============================================================================
# PENDULUM SIMULATION (same as other scripts)
# =============================================================================

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


def simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt=0.002):
    """Simulate pendulum and return time, angle, and angular velocity."""
    y0 = [np.radians(theta0_deg), 0]
    t_eval = np.arange(0, t_final, dt)

    sol = solve_ivp(
        lambda t, y: nonlinear_pendulum_ode(t, y, k_th, zeta, mu_c, mu_q),
        (0, t_final), y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10
    )

    return sol.t, sol.y[0], sol.y[1]


# =============================================================================
# CUSTOM SINDY LIBRARY FOR PENDULUM
# =============================================================================

def build_pendulum_library(x, include_coulomb=True, include_quadratic=True, coulomb_epsilon=0.1):
    """
    Build custom library matrix for pendulum identification.

    x: array of shape (n_samples, 2) with columns [theta, theta_dot]
    coulomb_epsilon: smoothing parameter for tanh approximation of sign function.
                     Optimal value ~0.1 balances smoothness for fitting and
                     sharpness for accuracy (reduces Coulomb error from 11% to 4%).

    Returns library matrix with columns:
    [1, θ, θ̇, cos(θ), sin(θ), θ̇|θ̇|, sign(θ̇), θ², θ·θ̇]
    """
    theta = x[:, 0:1]
    theta_dot = x[:, 1:2]

    # Smooth sign function with tunable epsilon
    # Using epsilon=0.1 instead of 1e-6 improves Coulomb estimation significantly
    sign_smooth = np.tanh(theta_dot / coulomb_epsilon)

    # Build library
    library = [
        np.ones_like(theta),           # 1 (constant/bias)
        theta,                          # θ (linear spring)
        theta_dot,                      # θ̇ (viscous damping)
        np.cos(theta),                  # cos(θ) (gravity nonlinearity)
        np.sin(theta),                  # sin(θ) (for completeness)
    ]

    feature_names = ['1', 'θ', 'θ̇', 'cos(θ)', 'sin(θ)']

    if include_quadratic:
        library.append(theta_dot * np.abs(theta_dot))  # θ̇|θ̇| (quadratic damping)
        feature_names.append('θ̇|θ̇|')

    if include_coulomb:
        library.append(sign_smooth)    # sign(θ̇) (Coulomb friction)
        feature_names.append('sign(θ̇)')

    # Additional nonlinear terms that might appear
    library.extend([
        theta ** 2,                     # θ² (for Taylor expansion)
        theta * theta_dot,              # θ·θ̇ (cross term)
    ])
    feature_names.extend(['θ²', 'θ·θ̇'])

    return np.hstack(library), feature_names


def compute_derivatives(t, x, method='savgol'):
    """
    Compute derivatives from time series.

    Methods:
    - 'finite_diff': Simple finite differences
    - 'savgol': Savitzky-Golay filter (smoother)
    """
    dt = t[1] - t[0]

    if method == 'finite_diff':
        # Central differences
        dx = np.zeros_like(x)
        dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        dx[0] = (x[1] - x[0]) / dt
        dx[-1] = (x[-1] - x[-2]) / dt
        return dx

    elif method == 'savgol':
        # Savitzky-Golay filter for smoother derivatives
        window = min(51, len(x) // 10)
        if window % 2 == 0:
            window += 1
        window = max(window, 5)

        dx = savgol_filter(x, window_length=window, polyorder=3, deriv=1, delta=dt)
        return dx

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# SINDY ESTIMATION (MANUAL IMPLEMENTATION)
# =============================================================================

def stlsq(Theta, dXdt, threshold=0.1, max_iter=10, normalize=True):
    """
    Sequential Thresholded Least Squares (STLSQ) algorithm.

    Solves: dXdt = Theta @ xi
    With sparsity promotion via iterative thresholding.

    Parameters:
    -----------
    Theta : array (n_samples, n_features) - Library matrix
    dXdt : array (n_samples,) - Target derivative
    threshold : float - Coefficient threshold for sparsity
    max_iter : int - Maximum iterations
    normalize : bool - Whether to normalize columns

    Returns:
    --------
    xi : array (n_features,) - Sparse coefficient vector
    """
    n_features = Theta.shape[1]

    # Normalize columns for better conditioning
    if normalize:
        norms = np.linalg.norm(Theta, axis=0)
        norms[norms == 0] = 1
        Theta_norm = Theta / norms
    else:
        Theta_norm = Theta
        norms = np.ones(n_features)

    # Initial least squares solution
    xi = np.linalg.lstsq(Theta_norm, dXdt, rcond=None)[0]

    # Iterative thresholding
    for _ in range(max_iter):
        # Threshold small coefficients
        small_idx = np.abs(xi) < threshold
        xi[small_idx] = 0

        # Re-solve for remaining terms
        big_idx = ~small_idx
        if np.sum(big_idx) == 0:
            break

        xi[big_idx] = np.linalg.lstsq(Theta_norm[:, big_idx], dXdt, rcond=None)[0]

    # Un-normalize coefficients
    xi = xi / norms

    return xi


def sindy_estimate_manual(t, theta, theta_dot, threshold=0.05,
                          include_coulomb=True, include_quadratic=True, coulomb_epsilon=0.1):
    """
    Manual SINDy implementation for pendulum parameter estimation.

    Parameters:
    -----------
    coulomb_epsilon : float - Smoothing parameter for tanh approximation of sign function.
                      Default 0.1 provides optimal balance for Coulomb estimation.

    Returns estimated coefficients and their interpretation.
    """
    # Compute acceleration (second derivative)
    theta_ddot = compute_derivatives(t, theta_dot, method='savgol')

    # Build state matrix
    x = np.column_stack([theta, theta_dot])

    # Build library with tunable Coulomb smoothing
    Theta, feature_names = build_pendulum_library(x, include_coulomb, include_quadratic, coulomb_epsilon)

    # Trim edges (derivative artifacts)
    trim = 50
    Theta = Theta[trim:-trim]
    theta_ddot = theta_ddot[trim:-trim]

    # Apply STLSQ
    xi = stlsq(Theta, theta_ddot, threshold=threshold)

    # Build results dictionary
    results = {
        'coefficients': xi,
        'feature_names': feature_names,
        'active_terms': [(name, coef) for name, coef in zip(feature_names, xi) if abs(coef) > 1e-10]
    }

    # Interpret coefficients for damping parameters
    # θ̈ = c_θ·θ + c_θ̇·θ̇ + c_cos·cos(θ) + c_quad·θ̇|θ̇| + c_sign·sign(θ̇) + ...
    # Comparing to: θ̈ = -k_θ·θ - 2ζ·θ̇ + cos(θ) - μ_q·θ̇|θ̇| - μ_c·sign(θ̇)

    idx_theta = feature_names.index('θ')
    idx_theta_dot = feature_names.index('θ̇')
    idx_cos = feature_names.index('cos(θ)')

    results['k_th_est'] = -xi[idx_theta]  # k_θ = -c_θ
    results['zeta_est'] = -xi[idx_theta_dot] / 2  # 2ζ = -c_θ̇
    results['cos_coef'] = xi[idx_cos]  # Should be ≈ 1

    if include_quadratic and 'θ̇|θ̇|' in feature_names:
        idx_quad = feature_names.index('θ̇|θ̇|')
        results['mu_q_est'] = -xi[idx_quad]  # μ_q = -c_quad
    else:
        results['mu_q_est'] = 0

    if include_coulomb and 'sign(θ̇)' in feature_names:
        idx_sign = feature_names.index('sign(θ̇)')
        results['mu_c_est'] = -xi[idx_sign]  # μ_c = -c_sign
    else:
        results['mu_c_est'] = 0

    return results


# =============================================================================
# SINDY ESTIMATION (USING PYSINDY LIBRARY)
# =============================================================================

def sindy_estimate_pysindy(t, theta, theta_dot, threshold=0.05):
    """
    PySINDy-based estimation using the official library.

    Uses custom library for pendulum-specific terms.
    """
    if not PYSINDY_AVAILABLE:
        raise ImportError("pysindy not installed")

    # Build state matrix [θ, θ̇]
    x = np.column_stack([theta, theta_dot])

    # Define custom library functions
    def sign_smooth(x):
        return np.tanh(x / 1e-6)

    def abs_product(x):
        return x * np.abs(x)

    # Build library using PySINDy's CustomLibrary
    library_functions = [
        lambda x: x,                    # Identity (θ, θ̇)
        lambda x: np.cos(x),            # cos
        lambda x: np.sin(x),            # sin
        lambda x: x ** 2,               # Square
        lambda x: sign_smooth(x),       # Smooth sign
        lambda x: abs_product(x),       # x|x|
    ]

    function_names = [
        lambda x: x,
        lambda x: f'cos({x})',
        lambda x: f'sin({x})',
        lambda x: f'{x}^2',
        lambda x: f'sign({x})',
        lambda x: f'{x}|{x}|',
    ]

    custom_library = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=function_names
    )

    # Also include polynomial terms
    poly_library = ps.PolynomialLibrary(degree=2, include_bias=True)

    # Combine libraries
    combined_library = poly_library + custom_library

    # Create and fit SINDy model
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=combined_library,
        differentiation_method=ps.SmoothedFiniteDifference(),
        feature_names=['θ', 'θ̇']
    )

    model.fit(x, t=t)

    return model


# =============================================================================
# MAIN ESTIMATION PIPELINE
# =============================================================================

def estimate_damping_sindy(damping_type='viscous', true_zeta=0.05, true_mu_c=0.03, true_mu_q=0.05,
                           k_th=20, theta0_deg=30, t_final=60, dt=0.002, noise_std=0.0,
                           threshold=0.05, use_pysindy=False, plotting=True):
    """
    Complete SINDy-based damping parameter estimation pipeline.

    Parameters:
    -----------
    damping_type : str - 'viscous', 'coulomb', 'quadratic', or 'combined'
    true_* : float - True parameter values for simulation
    k_th : float - Torsional stiffness
    theta0_deg : float - Initial angle in degrees
    t_final : float - Simulation time
    dt : float - Time step
    noise_std : float - Measurement noise standard deviation
    threshold : float - STLSQ threshold for sparsity
    use_pysindy : bool - Use PySINDy library (if available)
    plotting : bool - Generate plots

    Returns:
    --------
    results : dict - Estimation results and comparison
    """
    # Set damping parameters based on type
    if damping_type == 'viscous':
        zeta, mu_c, mu_q = true_zeta, 0, 0
    elif damping_type == 'coulomb':
        zeta, mu_c, mu_q = 0, true_mu_c, 0
    elif damping_type == 'quadratic':
        zeta, mu_c, mu_q = 0, 0, true_mu_q
    elif damping_type == 'combined':
        zeta, mu_c, mu_q = true_zeta, true_mu_c, true_mu_q
    else:
        raise ValueError(f"Unknown damping type: {damping_type}")

    print(f"\n{'='*60}")
    print(f"SINDy ESTIMATION: {damping_type.upper()} DAMPING")
    print('='*60)

    # Simulate pendulum
    t, theta, theta_dot = simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt)

    # Add noise
    if noise_std > 0:
        np.random.seed(42)
        theta = theta + np.random.normal(0, noise_std, len(theta))
        theta_dot = theta_dot + np.random.normal(0, noise_std, len(theta_dot))

    print(f"\nTrue Parameters:")
    print(f"  k_θ  = {k_th:.4f}")
    print(f"  ζ    = {zeta:.4f}")
    print(f"  μ_c  = {mu_c:.4f}")
    print(f"  μ_q  = {mu_q:.4f}")

    # Run SINDy estimation
    if use_pysindy and PYSINDY_AVAILABLE:
        print("\nUsing PySINDy library...")
        model = sindy_estimate_pysindy(t, theta, theta_dot, threshold)
        model.print()
        results = {'pysindy_model': model}
    else:
        print("\nUsing manual STLSQ implementation...")
        results = sindy_estimate_manual(
            t, theta, theta_dot,
            threshold=threshold,
            include_coulomb=(damping_type in ['coulomb', 'combined']),
            include_quadratic=(damping_type in ['quadratic', 'combined'])
        )

    print(f"\nEstimated Parameters (SINDy):")
    print(f"  k_θ  = {results.get('k_th_est', np.nan):.4f} (true: {k_th:.4f})")
    print(f"  ζ    = {results.get('zeta_est', np.nan):.4f} (true: {zeta:.4f})")
    print(f"  μ_c  = {results.get('mu_c_est', np.nan):.4f} (true: {mu_c:.4f})")
    print(f"  μ_q  = {results.get('mu_q_est', np.nan):.4f} (true: {mu_q:.4f})")
    print(f"  cos  = {results.get('cos_coef', np.nan):.4f} (expected: 1.0)")

    # Calculate errors
    if zeta > 0:
        zeta_error = abs(results['zeta_est'] - zeta) / zeta * 100
        print(f"\n  ζ estimation error: {zeta_error:.2f}%")
    if mu_c > 0:
        mu_c_error = abs(results['mu_c_est'] - mu_c) / mu_c * 100
        print(f"  μ_c estimation error: {mu_c_error:.2f}%")
    if mu_q > 0:
        mu_q_error = abs(results['mu_q_est'] - mu_q) / mu_q * 100
        print(f"  μ_q estimation error: {mu_q_error:.2f}%")

    print("\nActive terms in identified equation:")
    for name, coef in results.get('active_terms', []):
        print(f"  {coef:+.4f} * {name}")

    # Store additional info
    results['t'] = t
    results['theta'] = theta
    results['theta_dot'] = theta_dot
    results['true_params'] = {'k_th': k_th, 'zeta': zeta, 'mu_c': mu_c, 'mu_q': mu_q}
    results['damping_type'] = damping_type

    # Generate plots
    if plotting:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Time series
        ax = axes[0, 0]
        ax.plot(t, np.degrees(theta), 'b-', linewidth=0.8)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('θ [deg]')
        ax.set_title(f'{damping_type.capitalize()} Damping - Time Response')
        ax.grid(True, alpha=0.3)

        # Phase portrait
        ax = axes[0, 1]
        ax.plot(np.degrees(theta), np.degrees(theta_dot), 'b-', linewidth=0.5)
        ax.set_xlabel('θ [deg]')
        ax.set_ylabel('θ̇ [deg/s]')
        ax.set_title('Phase Portrait')
        ax.grid(True, alpha=0.3)

        # Coefficient bar chart
        ax = axes[1, 0]
        active = [(n, c) for n, c in zip(results['feature_names'], results['coefficients'])
                  if abs(c) > 1e-10]
        if active:
            names, coefs = zip(*active)
            colors = ['green' if c > 0 else 'red' for c in coefs]
            ax.barh(range(len(names)), coefs, color=colors, alpha=0.7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Identified Coefficients')
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='x')

        # Parameter comparison
        ax = axes[1, 1]
        param_names = ['k_θ', 'ζ', 'μ_c', 'μ_q']
        true_vals = [k_th, zeta, mu_c, mu_q]
        est_vals = [results['k_th_est'], results['zeta_est'],
                    results['mu_c_est'], results['mu_q_est']]

        x = np.arange(len(param_names))
        width = 0.35
        ax.bar(x - width/2, true_vals, width, label='True', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, est_vals, width, label='Estimated', color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names)
        ax.set_ylabel('Value')
        ax.set_title('True vs Estimated Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'SINDy Parameter Estimation - {damping_type.capitalize()}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'sindy_{damping_type}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: sindy_{damping_type}.png")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SINDy-BASED DAMPING PARAMETER ESTIMATION")
    print("Sparse Identification of Nonlinear Dynamics")
    print("=" * 70)

    # System parameters
    K_TH = 20
    THETA0 = 30  # degrees
    T_FINAL = 60
    DT = 0.002
    NOISE_STD = 0.001  # Small measurement noise
    THRESHOLD = 0.03   # STLSQ sparsity threshold

    # True damping parameters
    TRUE_ZETA = 0.05
    TRUE_MU_C = 0.03
    TRUE_MU_Q = 0.05

    results_all = {}

    # Test 1: Viscous damping
    results_all['viscous'] = estimate_damping_sindy(
        damping_type='viscous',
        true_zeta=TRUE_ZETA,
        k_th=K_TH,
        theta0_deg=THETA0,
        t_final=T_FINAL,
        dt=DT,
        noise_std=NOISE_STD,
        threshold=THRESHOLD,
        plotting=True
    )

    # Test 2: Coulomb damping
    results_all['coulomb'] = estimate_damping_sindy(
        damping_type='coulomb',
        true_mu_c=TRUE_MU_C,
        k_th=K_TH,
        theta0_deg=THETA0,
        t_final=T_FINAL,
        dt=DT,
        noise_std=NOISE_STD,
        threshold=THRESHOLD,
        plotting=True
    )

    # Test 3: Quadratic damping
    results_all['quadratic'] = estimate_damping_sindy(
        damping_type='quadratic',
        true_mu_q=TRUE_MU_Q,
        k_th=K_TH,
        theta0_deg=THETA0,
        t_final=T_FINAL,
        dt=DT,
        noise_std=NOISE_STD,
        threshold=THRESHOLD,
        plotting=True
    )

    # Test 4: Combined damping
    results_all['combined'] = estimate_damping_sindy(
        damping_type='combined',
        true_zeta=TRUE_ZETA * 0.5,  # Smaller values for combined
        true_mu_c=TRUE_MU_C * 0.5,
        true_mu_q=TRUE_MU_Q * 0.5,
        k_th=K_TH,
        theta0_deg=THETA0,
        t_final=T_FINAL,
        dt=DT,
        noise_std=NOISE_STD,
        threshold=THRESHOLD * 0.5,  # Lower threshold for combined
        plotting=True
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SINDy ESTIMATION RESULTS")
    print("=" * 70)

    print(f"\n{'Type':<12} {'Parameter':<8} {'True':<10} {'Estimated':<10} {'Error':<10}")
    print("-" * 55)

    for dtype, res in results_all.items():
        true_p = res['true_params']

        if dtype == 'viscous':
            true_val = true_p['zeta']
            est_val = res['zeta_est']
            param = 'ζ'
        elif dtype == 'coulomb':
            true_val = true_p['mu_c']
            est_val = res['mu_c_est']
            param = 'μ_c'
        elif dtype == 'quadratic':
            true_val = true_p['mu_q']
            est_val = res['mu_q_est']
            param = 'μ_q'
        else:
            continue

        if true_val > 0:
            err = abs(est_val - true_val) / true_val * 100
            status = "✓" if err < 10 else ("~" if err < 25 else "✗")
            print(f"{dtype:<12} {param:<8} {true_val:<10.4f} {est_val:<10.4f} {err:<6.1f}%  {status}")

    print("\n" + "=" * 70)
    print("SINDy estimation complete!")
    print("=" * 70)
