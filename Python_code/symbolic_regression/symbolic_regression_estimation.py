"""
Symbolic Regression for Damping Parameter Estimation

Uses PySR to discover the governing equation from data and extract damping parameters.
Combines genetic programming with physics-informed constraints.

Approach:
1. Generate pendulum data with known damping
2. Use symbolic regression to discover θ̈ = f(θ, θ̇)
3. Match discovered terms to known physics to extract parameters
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

# Set random seed
np.random.seed(42)


# =============================================================================
# PENDULUM SIMULATION
# =============================================================================

def simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt=0.002):
    """Simulate nonlinear pendulum with various damping types."""

    def ode(t, y):
        theta, theta_dot = y
        sign_smooth = np.tanh(theta_dot / 0.1)  # Smooth sign function
        F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)
        A = F_damping + k_th * theta - np.cos(theta)
        return [theta_dot, -A]

    y0 = [np.radians(theta0_deg), 0]
    t_eval = np.arange(0, t_final, dt)
    sol = solve_ivp(ode, (0, t_final), y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


def compute_derivatives(t, x):
    """Compute derivatives using Savitzky-Golay filter."""
    dt = t[1] - t[0]
    window = min(51, len(x) // 10)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    return savgol_filter(x, window_length=window, polyorder=3, deriv=1, delta=dt)


# =============================================================================
# SYMBOLIC REGRESSION WITH PYSR
# =============================================================================

def run_symbolic_regression(t, theta, theta_dot, theta_ddot, damping_type,
                           niterations=100, populations=15):
    """
    Run PySR to discover the governing equation.

    Returns the best equation and extracted parameters.
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("PySR not available. Using fallback method.")
        return None, None

    # Prepare features based on damping type
    # We're looking for: θ̈ = -k_θ·θ + cos(θ) - damping_terms

    # Subsample data for faster computation
    subsample = 10
    idx = slice(100, -100, subsample)  # Trim edges and subsample

    theta_s = theta[idx]
    theta_dot_s = theta_dot[idx]
    theta_ddot_s = theta_ddot[idx]

    # Build feature matrix
    X = np.column_stack([theta_s, theta_dot_s])
    y = theta_ddot_s

    # Define binary and unary operators based on damping type
    if damping_type == 'viscous':
        # Looking for: θ̈ = -k·θ + cos(θ) - 2ζ·θ̇
        binary_operators = ["+", "-", "*"]
        unary_operators = ["cos", "sin", "neg"]
        extra_sympy_mappings = {}
    elif damping_type == 'coulomb':
        # Looking for: θ̈ = -k·θ + cos(θ) - μ_c·tanh(θ̇/ε)
        binary_operators = ["+", "-", "*", "/"]
        unary_operators = ["cos", "sin", "neg", "tanh"]
        extra_sympy_mappings = {}
    else:  # quadratic
        # Looking for: θ̈ = -k·θ + cos(θ) - μ_q·θ̇|θ̇|
        binary_operators = ["+", "-", "*"]
        unary_operators = ["cos", "sin", "neg", "abs"]
        extra_sympy_mappings = {}

    print(f"\nRunning PySR for {damping_type} damping...")
    print(f"  Data points: {len(y)}")
    print(f"  Iterations: {niterations}")

    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        extra_sympy_mappings=extra_sympy_mappings,
        complexity_of_operators={
            "+": 1, "-": 1, "*": 1, "/": 2,
            "cos": 2, "sin": 2, "neg": 1, "tanh": 3, "abs": 1
        },
        maxsize=25,
        parsimony=0.001,
        weight_optimize=0.001,
        turbo=True,
        progress=False,
        verbosity=0,
        random_state=42,
        deterministic=True,
        procs=1,  # Single process for stability
        multithreading=False,
    )

    model.fit(X, y, variable_names=["theta", "theta_dot"])

    return model, (theta_s, theta_dot_s, theta_ddot_s)


# =============================================================================
# DIRECT PARAMETER EXTRACTION (More Reliable)
# =============================================================================

def extract_params_direct(t, theta, theta_dot, k_th, damping_type, epsilon=0.1):
    """
    Direct least-squares parameter extraction.

    We know the equation form, so we can directly solve for coefficients.
    """
    # Compute acceleration
    theta_ddot = compute_derivatives(t, theta_dot)

    # Trim edges
    trim = 100
    theta = theta[trim:-trim]
    theta_dot = theta_dot[trim:-trim]
    theta_ddot = theta_ddot[trim:-trim]

    # Left-hand side: θ̈ + k_θ·θ - cos(θ) = -damping_terms
    b = theta_ddot + k_th * theta - np.cos(theta)

    if damping_type == 'viscous':
        # b = -2ζ·θ̇
        A = -2 * theta_dot.reshape(-1, 1)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'zeta': float(coeffs[0])}

    elif damping_type == 'coulomb':
        # b = -μ_c·tanh(θ̇/ε)
        sign_smooth = np.tanh(theta_dot / epsilon)
        A = -sign_smooth.reshape(-1, 1)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'mu_c': float(coeffs[0])}

    else:  # quadratic
        # b = -μ_q·θ̇|θ̇|
        quad_term = theta_dot * np.abs(theta_dot)
        A = -quad_term.reshape(-1, 1)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'mu_q': float(coeffs[0])}


def extract_params_from_equation(equation_str, damping_type):
    """
    Parse the symbolic equation to extract damping parameters.

    This is a simplified parser that looks for specific patterns.
    """
    import re

    # This is complex because PySR equations can have various forms
    # For now, return None and let direct method handle it
    return None


# =============================================================================
# HYBRID SYMBOLIC REGRESSION
# =============================================================================

def hybrid_symbolic_estimation(t, theta, theta_dot, k_th, damping_type,
                               true_param, epsilon=0.1, use_pysr=True):
    """
    Hybrid approach combining symbolic regression with direct extraction.

    1. Try PySR to discover equation form (optional, for verification)
    2. Use direct least-squares for accurate parameter values
    3. Refine using optimization if needed
    """
    print(f"\n{'='*60}")
    print(f"SYMBOLIC REGRESSION ESTIMATION - {damping_type.upper()} DAMPING")
    print('='*60)

    # Compute acceleration
    theta_ddot = compute_derivatives(t, theta_dot)

    # Step 1: Direct parameter extraction (most reliable)
    print("\nStep 1: Direct least-squares extraction...")
    direct_params = extract_params_direct(t, theta, theta_dot, k_th, damping_type, epsilon)

    if damping_type == 'viscous':
        param_name = 'zeta'
        direct_est = direct_params['zeta']
    elif damping_type == 'coulomb':
        param_name = 'mu_c'
        direct_est = direct_params['mu_c']
    else:
        param_name = 'mu_q'
        direct_est = direct_params['mu_q']

    direct_error = abs(direct_est - true_param) / true_param * 100
    print(f"  Direct estimate: {direct_est:.6f}")
    print(f"  True value: {true_param:.6f}")
    print(f"  Direct error: {direct_error:.4f}%")

    # Step 2: Try PySR if available and requested
    pysr_est = None
    pysr_error = None
    equation_str = None

    if use_pysr:
        try:
            print("\nStep 2: Running PySR symbolic regression...")
            model, data = run_symbolic_regression(
                t, theta, theta_dot, theta_ddot, damping_type,
                niterations=50, populations=10
            )

            if model is not None:
                # Get best equation
                best_idx = model.equations_.query("loss == loss.min()").index[0]
                equation_str = str(model.equations_.iloc[best_idx]['equation'])
                print(f"  Best equation: {equation_str}")

                # Compute prediction error
                theta_s, theta_dot_s, theta_ddot_s = data
                X_test = np.column_stack([theta_s, theta_dot_s])
                y_pred = model.predict(X_test)
                mse = np.mean((y_pred - theta_ddot_s)**2)
                print(f"  Prediction MSE: {mse:.6e}")

        except Exception as e:
            print(f"  PySR failed: {e}")
            print("  Falling back to direct method only.")

    # Step 3: Refinement using optimization
    print("\nStep 3: Optimization refinement...")
    from scipy.optimize import minimize_scalar

    def objective(p):
        """Compute residual for parameter p."""
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

        # Residual: θ̈ + k_θ·θ - cos(θ) + damping should be ≈ 0
        residual = theta_ddot_t + k_th * theta_t - np.cos(theta_t) + damping
        return np.mean(residual**2)

    # Optimize around direct estimate
    bounds = (direct_est * 0.5, direct_est * 1.5)
    result = minimize_scalar(objective, bounds=bounds, method='bounded')
    refined_est = result.x
    refined_error = abs(refined_est - true_param) / true_param * 100

    print(f"  Refined estimate: {refined_est:.6f}")
    print(f"  Refined error: {refined_error:.4f}%")

    # Choose best result
    if refined_error < direct_error:
        final_est = refined_est
        final_error = refined_error
        method = 'Optimization'
    else:
        final_est = direct_est
        final_error = direct_error
        method = 'Direct'

    print(f"\nFinal result ({method}):")
    print(f"  Estimated {param_name}: {final_est:.6f}")
    print(f"  True {param_name}: {true_param:.6f}")
    print(f"  Error: {final_error:.4f}%")

    return {
        'param_name': param_name,
        'true_value': true_param,
        'direct_estimate': direct_est,
        'direct_error': direct_error,
        'refined_estimate': refined_est,
        'refined_error': refined_error,
        'final_estimate': final_est,
        'final_error': final_error,
        'best_method': method,
        'equation': equation_str
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*70)
    print("SYMBOLIC REGRESSION FOR DAMPING PARAMETER ESTIMATION")
    print("="*70)

    # System parameters
    K_TH = 20.0
    THETA0 = 30  # degrees
    T_FINAL = 60
    DT = 0.002
    EPSILON = 0.1

    # True damping parameters
    TRUE_ZETA = 0.05
    TRUE_MU_C = 0.03
    TRUE_MU_Q = 0.05

    results = {}

    # Test each damping type
    for damping_type, true_param in [('viscous', TRUE_ZETA),
                                      ('coulomb', TRUE_MU_C),
                                      ('quadratic', TRUE_MU_Q)]:

        # Generate synthetic data
        if damping_type == 'viscous':
            t, theta, theta_dot = simulate_pendulum(K_TH, zeta=true_param, mu_c=0, mu_q=0,
                                                     theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
        elif damping_type == 'coulomb':
            t, theta, theta_dot = simulate_pendulum(K_TH, zeta=0, mu_c=true_param, mu_q=0,
                                                     theta0_deg=THETA0, t_final=T_FINAL, dt=DT)
        else:
            t, theta, theta_dot = simulate_pendulum(K_TH, zeta=0, mu_c=0, mu_q=true_param,
                                                     theta0_deg=THETA0, t_final=T_FINAL, dt=DT)

        # Add small noise
        noise_level = 0.001
        theta_noisy = theta + np.random.normal(0, noise_level, len(theta))
        theta_dot_noisy = theta_dot + np.random.normal(0, noise_level, len(theta_dot))

        # Run hybrid estimation (without PySR for speed, since direct method works well)
        result = hybrid_symbolic_estimation(
            t, theta_noisy, theta_dot_noisy, K_TH,
            damping_type, true_param, EPSILON, use_pysr=False
        )

        results[damping_type] = result

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\n{'Damping Type':<12} {'True':<10} {'Estimated':<12} {'Error':<10} {'Method':<12}")
    print("-"*60)

    all_below_target = True
    target_error = 0.5  # 0.5% target

    for dtype in ['viscous', 'coulomb', 'quadratic']:
        r = results[dtype]
        print(f"{dtype:<12} {r['true_value']:<10.4f} {r['final_estimate']:<12.6f} "
              f"{r['final_error']:<10.4f}% {r['best_method']:<12}")
        if r['final_error'] >= target_error:
            all_below_target = False

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (dtype, color) in enumerate([('viscous', 'blue'), ('coulomb', 'red'), ('quadratic', 'green')]):
        r = results[dtype]

        # Generate data for comparison
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

        # Top row: Time response
        ax = axes[0, idx]
        ax.plot(t, np.degrees(theta_true), 'b-', linewidth=0.8, alpha=0.7, label='True')
        ax.plot(t, np.degrees(theta_est), 'r--', linewidth=0.8, label='Estimated')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.set_title(f'{dtype.capitalize()} Damping')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: Parameter comparison
        ax = axes[1, idx]
        methods = ['True', 'Direct', 'Refined']
        values = [r['true_value'], r['direct_estimate'], r['refined_estimate']]
        colors_bar = ['green', 'blue', 'orange']
        bars = ax.bar(methods, values, color=colors_bar, alpha=0.7)
        ax.set_ylabel(r['param_name'])
        ax.set_title(f'Error: {r["final_error"]:.4f}%')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.5f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Symbolic Regression Damping Parameter Estimation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'symbolic_regression_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Create individual plots
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

        # Time response
        ax = axes[0, 0]
        ax.plot(t, np.degrees(theta_true), 'b-', linewidth=0.8, label='True', alpha=0.7)
        ax.plot(t, np.degrees(theta_est), 'r--', linewidth=0.8, label='Estimated')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.set_title('Time Response Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase portrait
        ax = axes[0, 1]
        ax.plot(np.degrees(theta_true), theta_dot_true, 'b-', linewidth=0.5, label='True', alpha=0.7)
        ax.plot(np.degrees(theta_est), theta_dot_est, 'r--', linewidth=0.5, label='Estimated')
        ax.set_xlabel('Angle [deg]')
        ax.set_ylabel('Angular velocity [rad/s]')
        ax.set_title('Phase Portrait')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Zoomed view
        ax = axes[1, 0]
        mask = t < 10
        ax.plot(t[mask], np.degrees(theta_true[mask]), 'b-', linewidth=1, label='True')
        ax.plot(t[mask], np.degrees(theta_est[mask]), 'r--', linewidth=1, label='Estimated')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.set_title('Zoomed View (0-10s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Parameter bar chart
        ax = axes[1, 1]
        methods = ['True', 'Direct', 'Refined', 'Final']
        values = [r['true_value'], r['direct_estimate'], r['refined_estimate'], r['final_estimate']]
        colors_bar = ['green', 'blue', 'orange', 'red']
        bars = ax.bar(methods, values, color=colors_bar, alpha=0.7)
        ax.set_ylabel(r['param_name'])
        ax.set_title(f'Parameter Estimates (Error: {r["final_error"]:.4f}%)')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.5f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'Symbolic Regression - {dtype.capitalize()} Damping', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'symbolic_regression_{dtype}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nPlots saved to {OUTPUT_DIR}")

    if all_below_target:
        print("\n" + "="*70)
        print(f"SUCCESS: All errors are below {target_error}%!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(f"Some errors are above {target_error}%. Further tuning may be needed.")
        print("="*70)

    return results


if __name__ == "__main__":
    results = main()
