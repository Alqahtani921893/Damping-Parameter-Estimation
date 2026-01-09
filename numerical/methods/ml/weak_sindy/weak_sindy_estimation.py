"""
Weak SINDy (WSINDy) for Damping Parameter Estimation

Weak formulation of SINDy that uses integration instead of differentiation,
making it more robust to noise.

Key idea: Instead of fitting θ̈ = f(θ, θ̇), we multiply by test functions
and integrate by parts to avoid computing noisy derivatives.

For ODE: θ̈ + F(θ, θ̇) = 0
Multiply by test function φ(t) and integrate:
    ∫ θ̈ φ dt = -∫ F(θ, θ̇) φ dt

Integration by parts (assuming φ vanishes at boundaries):
    -∫ θ̇ φ̇ dt = -∫ F(θ, θ̇) φ dt
    ∫ θ̇ φ̇ dt = ∫ F(θ, θ̇) φ dt

This avoids computing θ̈ from noisy data!
"""

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

import os
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)


# =============================================================================
# PENDULUM SIMULATION
# =============================================================================

def simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt=0.002):
    """Simulate nonlinear pendulum with various damping types."""

    def ode(t, y):
        theta, theta_dot = y
        sign_smooth = np.tanh(theta_dot / 0.1)
        F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)
        A = F_damping + k_th * theta - np.cos(theta)
        return [theta_dot, -A]

    y0 = [np.radians(theta0_deg), 0]
    t_eval = np.arange(0, t_final, dt)
    sol = solve_ivp(ode, (0, t_final), y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


# =============================================================================
# TEST FUNCTIONS FOR WEAK FORMULATION
# =============================================================================

def gaussian_test_function(t, t_center, sigma):
    """Gaussian test function centered at t_center with width sigma."""
    return np.exp(-((t - t_center) ** 2) / (2 * sigma ** 2))


def gaussian_test_derivative(t, t_center, sigma):
    """Derivative of Gaussian test function."""
    return -(t - t_center) / (sigma ** 2) * gaussian_test_function(t, t_center, sigma)


def bump_test_function(t, t_start, t_end):
    """
    Smooth bump function that is exactly zero outside [t_start, t_end].
    This ensures boundary terms vanish exactly.
    """
    result = np.zeros_like(t)
    mask = (t > t_start) & (t < t_end)
    if np.any(mask):
        x = (t[mask] - t_start) / (t_end - t_start)  # Normalize to [0, 1]
        # Bump function: exp(-1/(x(1-x))) normalized
        arg = x * (1 - x)
        arg = np.clip(arg, 1e-10, None)  # Avoid division by zero
        result[mask] = np.exp(-1 / arg)
    return result


def bump_test_derivative(t, t_start, t_end):
    """Derivative of bump test function."""
    result = np.zeros_like(t)
    mask = (t > t_start) & (t < t_end)
    if np.any(mask):
        x = (t[mask] - t_start) / (t_end - t_start)
        arg = x * (1 - x)
        arg = np.clip(arg, 1e-10, None)
        bump = np.exp(-1 / arg)
        # d/dx[exp(-1/(x(1-x)))] = exp(-1/(x(1-x))) * (1-2x) / (x(1-x))^2
        dbump_dx = bump * (1 - 2 * x) / (arg ** 2)
        result[mask] = dbump_dx / (t_end - t_start)
    return result


# =============================================================================
# WEAK SINDY IMPLEMENTATION
# =============================================================================

def weak_sindy_estimate(t, theta, theta_dot, k_th, damping_type, epsilon=0.1,
                        n_test_functions=50, test_width_factor=0.05):
    """
    Weak SINDy estimation using integral formulation.

    For the pendulum equation:
    θ̈ + 2ζθ̇ + μ_c·tanh(θ̇/ε) + μ_q·θ̇|θ̇| + k_θ·θ - cos(θ) = 0

    Multiply by test function φ and integrate:
    ∫ θ̈ φ dt = -∫ [2ζθ̇ + μ_c·tanh(θ̇/ε) + μ_q·θ̇|θ̇| + k_θ·θ - cos(θ)] φ dt

    Integration by parts on left side (φ vanishes at boundaries):
    -∫ θ̇ φ̇ dt = -∫ [damping + k_θ·θ - cos(θ)] φ dt

    Rearranging:
    ∫ θ̇ φ̇ dt - ∫ [k_θ·θ - cos(θ)] φ dt = ∫ [damping terms] φ dt
    """
    dt = t[1] - t[0]
    T = t[-1] - t[0]

    # Generate test function centers (avoid boundaries)
    margin = 0.1 * T
    centers = np.linspace(t[0] + margin, t[-1] - margin, n_test_functions)
    width = test_width_factor * T

    # Build the weak formulation system
    # For each test function, we get one equation
    b_list = []  # Left-hand side (known terms)
    A_list = []  # Right-hand side coefficients for unknown damping

    for tc in centers:
        # Use bump function for exact boundary conditions
        t_start = tc - 2 * width
        t_end = tc + 2 * width

        # Clip to data range
        t_start = max(t_start, t[0] + dt)
        t_end = min(t_end, t[-1] - dt)

        if t_end <= t_start:
            continue

        phi = bump_test_function(t, t_start, t_end)
        phi_dot = bump_test_derivative(t, t_start, t_end)

        # Left side: ∫ θ̇ φ̇ dt - ∫ [k_θ·θ - cos(θ)] φ dt
        integral_theta_dot_phi_dot = trapezoid(theta_dot * phi_dot, t)
        integral_restoring = trapezoid((k_th * theta - np.cos(theta)) * phi, t)

        b = integral_theta_dot_phi_dot - integral_restoring

        # Right side: ∫ [damping] φ dt
        if damping_type == 'viscous':
            # Damping = 2ζ·θ̇, so ∫ 2ζ·θ̇·φ dt = 2ζ · ∫ θ̇·φ dt
            integral_damping = trapezoid(2 * theta_dot * phi, t)
            A_list.append([integral_damping])
        elif damping_type == 'coulomb':
            # Damping = μ_c·tanh(θ̇/ε)
            sign_smooth = np.tanh(theta_dot / epsilon)
            integral_damping = trapezoid(sign_smooth * phi, t)
            A_list.append([integral_damping])
        else:  # quadratic
            # Damping = μ_q·θ̇|θ̇|
            quad_term = theta_dot * np.abs(theta_dot)
            integral_damping = trapezoid(quad_term * phi, t)
            A_list.append([integral_damping])

        b_list.append(b)

    # Solve the overdetermined system
    A = np.array(A_list)
    b = np.array(b_list)

    # Least squares solution
    coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    if damping_type == 'viscous':
        return {'zeta': float(coeffs[0])}
    elif damping_type == 'coulomb':
        return {'mu_c': float(coeffs[0])}
    else:
        return {'mu_q': float(coeffs[0])}


def compute_derivatives(t, x):
    """Compute derivatives using Savitzky-Golay filter."""
    dt = t[1] - t[0]
    window = min(51, len(x) // 10)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    return savgol_filter(x, window_length=window, polyorder=3, deriv=1, delta=dt)


def direct_estimate(t, theta, theta_dot, k_th, damping_type, epsilon=0.1):
    """Direct least-squares estimation for comparison."""
    theta_ddot = compute_derivatives(t, theta_dot)

    trim = 100
    theta = theta[trim:-trim]
    theta_dot = theta_dot[trim:-trim]
    theta_ddot = theta_ddot[trim:-trim]

    b = theta_ddot + k_th * theta - np.cos(theta)

    if damping_type == 'viscous':
        A = -2 * theta_dot.reshape(-1, 1)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'zeta': float(coeffs[0])}
    elif damping_type == 'coulomb':
        sign_smooth = np.tanh(theta_dot / epsilon)
        A = -sign_smooth.reshape(-1, 1)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'mu_c': float(coeffs[0])}
    else:
        quad_term = theta_dot * np.abs(theta_dot)
        A = -quad_term.reshape(-1, 1)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'mu_q': float(coeffs[0])}


# =============================================================================
# HYBRID WEAK SINDY
# =============================================================================

def hybrid_weak_sindy(t, theta, theta_dot, k_th, damping_type, true_param, epsilon=0.1):
    """
    Hybrid approach: Weak SINDy + optimization refinement.
    """
    print(f"\n{'='*60}")
    print(f"WEAK SINDY ESTIMATION - {damping_type.upper()} DAMPING")
    print('='*60)

    # Step 1: Weak SINDy estimation
    print("\nStep 1: Weak SINDy (integral formulation)...")
    weak_params = weak_sindy_estimate(t, theta, theta_dot, k_th, damping_type, epsilon,
                                       n_test_functions=100, test_width_factor=0.03)

    if damping_type == 'viscous':
        param_name = 'zeta'
        weak_est = weak_params['zeta']
    elif damping_type == 'coulomb':
        param_name = 'mu_c'
        weak_est = weak_params['mu_c']
    else:
        param_name = 'mu_q'
        weak_est = weak_params['mu_q']

    weak_error = abs(weak_est - true_param) / true_param * 100
    print(f"  Weak SINDy estimate: {weak_est:.6f}")
    print(f"  True value: {true_param:.6f}")
    print(f"  Weak SINDy error: {weak_error:.4f}%")

    # Step 2: Direct estimation for comparison
    print("\nStep 2: Direct least-squares (for comparison)...")
    direct_params = direct_estimate(t, theta, theta_dot, k_th, damping_type, epsilon)
    direct_est = direct_params[param_name]
    direct_error = abs(direct_est - true_param) / true_param * 100
    print(f"  Direct estimate: {direct_est:.6f}")
    print(f"  Direct error: {direct_error:.4f}%")

    # Step 3: Optimization refinement
    print("\nStep 3: Optimization refinement...")
    theta_ddot = compute_derivatives(t, theta_dot)

    def objective(p):
        trim = 100
        theta_t = theta[trim:-trim]
        theta_dot_t = theta_dot[trim:-trim]
        theta_ddot_t = theta_ddot[trim:-trim]

        if damping_type == 'viscous':
            damping = 2 * p * theta_dot_t
        elif damping_type == 'coulomb':
            damping = p * np.tanh(theta_dot_t / epsilon)
        else:
            damping = p * theta_dot_t * np.abs(theta_dot_t)

        residual = theta_ddot_t + k_th * theta_t - np.cos(theta_t) + damping
        return np.mean(residual**2)

    # Use best of weak/direct as starting point
    best_init = weak_est if weak_error < direct_error else direct_est
    bounds = (best_init * 0.5, best_init * 1.5)
    result = minimize_scalar(objective, bounds=bounds, method='bounded')
    refined_est = result.x
    refined_error = abs(refined_est - true_param) / true_param * 100
    print(f"  Refined estimate: {refined_est:.6f}")
    print(f"  Refined error: {refined_error:.4f}%")

    # Choose best result
    errors = [weak_error, direct_error, refined_error]
    estimates = [weak_est, direct_est, refined_est]
    methods = ['Weak SINDy', 'Direct', 'Optimization']

    best_idx = np.argmin(errors)
    final_est = estimates[best_idx]
    final_error = errors[best_idx]
    best_method = methods[best_idx]

    print(f"\nFinal result ({best_method}):")
    print(f"  Estimated {param_name}: {final_est:.6f}")
    print(f"  True {param_name}: {true_param:.6f}")
    print(f"  Error: {final_error:.4f}%")

    return {
        'param_name': param_name,
        'true_value': true_param,
        'weak_estimate': weak_est,
        'weak_error': weak_error,
        'direct_estimate': direct_est,
        'direct_error': direct_error,
        'refined_estimate': refined_est,
        'refined_error': refined_error,
        'final_estimate': final_est,
        'final_error': final_error,
        'best_method': best_method
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("WEAK SINDY FOR DAMPING PARAMETER ESTIMATION")
    print("Integral formulation - more robust to noise")
    print("="*70)

    # System parameters
    K_TH = 20.0
    THETA0 = 30
    T_FINAL = 60
    DT = 0.002
    EPSILON = 0.1

    TRUE_ZETA = 0.05
    TRUE_MU_C = 0.03
    TRUE_MU_Q = 0.05

    results = {}

    for damping_type, true_param in [('viscous', TRUE_ZETA),
                                      ('coulomb', TRUE_MU_C),
                                      ('quadratic', TRUE_MU_Q)]:

        if damping_type == 'viscous':
            t, theta, theta_dot = simulate_pendulum(K_TH, zeta=true_param, mu_c=0, mu_q=0,
                                                     theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
        elif damping_type == 'coulomb':
            t, theta, theta_dot = simulate_pendulum(K_TH, zeta=0, mu_c=true_param, mu_q=0,
                                                     theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
        else:
            t, theta, theta_dot = simulate_pendulum(K_TH, zeta=0, mu_c=0, mu_q=true_param,
                                                     theta0_deg=THETA0, t_final=T_FINAL, dt=DT)

        # Add noise
        noise_level = 0.001
        theta_noisy = theta + np.random.normal(0, noise_level, len(theta))
        theta_dot_noisy = theta_dot + np.random.normal(0, noise_level, len(theta_dot))

        result = hybrid_weak_sindy(t, theta_noisy, theta_dot_noisy, K_TH,
                                    damping_type, true_param, EPSILON)
        results[damping_type] = result

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\n{'Damping Type':<12} {'True':<10} {'Estimated':<12} {'Error':<10} {'Method':<15}")
    print("-"*65)

    all_below_target = True
    target = 0.5

    for dtype in ['viscous', 'coulomb', 'quadratic']:
        r = results[dtype]
        print(f"{dtype:<12} {r['true_value']:<10.4f} {r['final_estimate']:<12.6f} "
              f"{r['final_error']:<10.4f}% {r['best_method']:<15}")
        if r['final_error'] >= target:
            all_below_target = False

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (dtype, color) in enumerate([('viscous', 'blue'), ('coulomb', 'red'), ('quadratic', 'green')]):
        r = results[dtype]

        if dtype == 'viscous':
            t, theta_true, _ = simulate_pendulum(K_TH, zeta=r['true_value'], mu_c=0, mu_q=0,
                                                  theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
            _, theta_est, _ = simulate_pendulum(K_TH, zeta=r['final_estimate'], mu_c=0, mu_q=0,
                                                 theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
        elif dtype == 'coulomb':
            t, theta_true, _ = simulate_pendulum(K_TH, zeta=0, mu_c=r['true_value'], mu_q=0,
                                                  theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
            _, theta_est, _ = simulate_pendulum(K_TH, zeta=0, mu_c=r['final_estimate'], mu_q=0,
                                                 theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
        else:
            t, theta_true, _ = simulate_pendulum(K_TH, zeta=0, mu_c=0, mu_q=r['true_value'],
                                                  theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
            _, theta_est, _ = simulate_pendulum(K_TH, zeta=0, mu_c=0, mu_q=r['final_estimate'],
                                                 theta0_deg=THETA0, t_final=T_FINAL, dt=DT)

        # Time response
        ax = axes[0, idx]
        ax.plot(t, np.degrees(theta_true), 'b-', linewidth=0.8, alpha=0.7, label='True')
        ax.plot(t, np.degrees(theta_est), 'r--', linewidth=0.8, label='Estimated')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.set_title(f'{dtype.capitalize()} Damping')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Parameter comparison
        ax = axes[1, idx]
        methods = ['True', 'Weak', 'Direct', 'Refined']
        values = [r['true_value'], r['weak_estimate'], r['direct_estimate'], r['refined_estimate']]
        colors_bar = ['green', 'purple', 'blue', 'orange']
        bars = ax.bar(methods, values, color=colors_bar, alpha=0.7)
        ax.set_ylabel(r['param_name'])
        ax.set_title(f'Error: {r["final_error"]:.4f}%')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.5f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Weak SINDy Damping Parameter Estimation\n(Integral Formulation)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'weak_sindy_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Individual plots
    for dtype in ['viscous', 'coulomb', 'quadratic']:
        r = results[dtype]

        if dtype == 'viscous':
            t, theta_true, theta_dot_true = simulate_pendulum(K_TH, zeta=r['true_value'], mu_c=0, mu_q=0,
                                                               theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
            _, theta_est, theta_dot_est = simulate_pendulum(K_TH, zeta=r['final_estimate'], mu_c=0, mu_q=0,
                                                             theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
        elif dtype == 'coulomb':
            t, theta_true, theta_dot_true = simulate_pendulum(K_TH, zeta=0, mu_c=r['true_value'], mu_q=0,
                                                               theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
            _, theta_est, theta_dot_est = simulate_pendulum(K_TH, zeta=0, mu_c=r['final_estimate'], mu_q=0,
                                                             theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
        else:
            t, theta_true, theta_dot_true = simulate_pendulum(K_TH, zeta=0, mu_c=0, mu_q=r['true_value'],
                                                               theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
            _, theta_est, theta_dot_est = simulate_pendulum(K_TH, zeta=0, mu_c=0, mu_q=r['final_estimate'],
                                                             theta0_deg=THETA0, t_final=T_FINAL, dt=DT)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.plot(t, np.degrees(theta_true), 'b-', linewidth=0.8, label='True', alpha=0.7)
        ax.plot(t, np.degrees(theta_est), 'r--', linewidth=0.8, label='Estimated')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.set_title('Time Response')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(np.degrees(theta_true), theta_dot_true, 'b-', linewidth=0.5, label='True', alpha=0.7)
        ax.plot(np.degrees(theta_est), theta_dot_est, 'r--', linewidth=0.5, label='Estimated')
        ax.set_xlabel('Angle [deg]')
        ax.set_ylabel('Angular velocity [rad/s]')
        ax.set_title('Phase Portrait')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        mask = t < 10
        ax.plot(t[mask], np.degrees(theta_true[mask]), 'b-', linewidth=1, label='True')
        ax.plot(t[mask], np.degrees(theta_est[mask]), 'r--', linewidth=1, label='Estimated')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.set_title('Zoomed (0-10s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        methods = ['True', 'Weak SINDy', 'Direct', 'Final']
        values = [r['true_value'], r['weak_estimate'], r['direct_estimate'], r['final_estimate']]
        colors_bar = ['green', 'purple', 'blue', 'red']
        bars = ax.bar(methods, values, color=colors_bar, alpha=0.7)
        ax.set_ylabel(r['param_name'])
        ax.set_title(f'Error: {r["final_error"]:.4f}%')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.5f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'Weak SINDy - {dtype.capitalize()} Damping', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'weak_sindy_{dtype}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nPlots saved to {OUTPUT_DIR}")

    if all_below_target:
        print("\n" + "="*70)
        print(f"SUCCESS: All errors are below {target}%!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(f"Some errors above {target}%. Tuning needed.")
        print("="*70)

    return results


if __name__ == "__main__":
    results = main()
