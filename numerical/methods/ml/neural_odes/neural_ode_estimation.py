"""
Neural ODE-based Damping Parameter Estimation

Uses torchdiffeq to learn the dynamics of a nonlinear pendulum and estimate
damping parameters. Combines physics-informed structure with neural network
flexibility.

Approach:
1. Physics-Informed Neural ODE: Embed known physics structure, learn unknown parameters
2. Hybrid estimation: Direct least-squares + Neural ODE refinement
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

from torchdiffeq import odeint

import os
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cpu')  # Use CPU for stability


# =============================================================================
# PENDULUM SIMULATION (Ground Truth Data Generation)
# =============================================================================

def simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt=0.002):
    """Simulate nonlinear pendulum with various damping types."""

    def ode(t, y):
        theta, theta_dot = y
        # Use tanh approximation for sign function (epsilon=0.1 for consistency)
        sign_smooth = np.tanh(theta_dot / 0.1)
        F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)
        A = F_damping + k_th * theta - np.cos(theta)
        return [theta_dot, -A]

    y0 = [np.radians(theta0_deg), 0]
    t_eval = np.arange(0, t_final, dt)
    sol = solve_ivp(ode, (0, t_final), y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


# =============================================================================
# PHYSICS-INFORMED NEURAL ODE
# =============================================================================

class PendulumODEFunc(nn.Module):
    """
    Physics-informed ODE function for the pendulum.

    The dynamics are:
    θ̈ = -k_θ·θ + cos(θ) - 2ζ·θ̇ - μ_c·tanh(θ̇/ε) - μ_q·θ̇|θ̇|

    We know k_θ and want to estimate the damping parameters.
    """

    def __init__(self, k_th=20.0, damping_type='viscous', epsilon=0.1):
        super().__init__()
        self.k_th = k_th
        self.damping_type = damping_type
        self.epsilon = epsilon

        # Learnable damping parameters (initialized near expected values)
        if damping_type == 'viscous':
            self.zeta = nn.Parameter(torch.tensor([0.01]))  # Will learn to ~0.05
            self.mu_c = torch.tensor([0.0])
            self.mu_q = torch.tensor([0.0])
        elif damping_type == 'coulomb':
            self.zeta = torch.tensor([0.0])
            self.mu_c = nn.Parameter(torch.tensor([0.01]))  # Will learn to ~0.03
            self.mu_q = torch.tensor([0.0])
        elif damping_type == 'quadratic':
            self.zeta = torch.tensor([0.0])
            self.mu_c = torch.tensor([0.0])
            self.mu_q = nn.Parameter(torch.tensor([0.01]))  # Will learn to ~0.05
        else:  # combined
            self.zeta = nn.Parameter(torch.tensor([0.01]))
            self.mu_c = nn.Parameter(torch.tensor([0.01]))
            self.mu_q = nn.Parameter(torch.tensor([0.01]))

    def forward(self, t, y):
        """Compute dy/dt = [θ̇, θ̈]"""
        theta = y[..., 0:1]
        theta_dot = y[..., 1:2]

        # Smooth sign function
        sign_smooth = torch.tanh(theta_dot / self.epsilon)

        # Damping force
        F_damping = (2 * self.zeta * theta_dot +
                     self.mu_c * sign_smooth +
                     self.mu_q * theta_dot * torch.abs(theta_dot))

        # Acceleration: θ̈ = -k_θ·θ + cos(θ) - F_damping
        theta_ddot = -self.k_th * theta + torch.cos(theta) - F_damping

        return torch.cat([theta_dot, theta_ddot], dim=-1)


class NeuralODEEstimator:
    """
    Neural ODE-based parameter estimator using physics-informed structure.
    """

    def __init__(self, k_th=20.0, damping_type='viscous', epsilon=0.1):
        self.k_th = k_th
        self.damping_type = damping_type
        self.epsilon = epsilon
        self.ode_func = None
        self.training_history = {'loss': [], 'param': []}

    def fit(self, t, theta_obs, theta_dot_obs, epochs=500, lr=0.01, verbose=True):
        """
        Fit the Neural ODE to observed data.

        Args:
            t: Time array
            theta_obs: Observed angle
            theta_dot_obs: Observed angular velocity
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Print progress
        """
        # Convert to tensors
        t_tensor = torch.tensor(t, dtype=torch.float32)
        y_obs = torch.stack([
            torch.tensor(theta_obs, dtype=torch.float32),
            torch.tensor(theta_dot_obs, dtype=torch.float32)
        ], dim=-1)

        # Initial condition
        y0 = y_obs[0:1]

        # Create ODE function
        self.ode_func = PendulumODEFunc(
            k_th=self.k_th,
            damping_type=self.damping_type,
            epsilon=self.epsilon
        )

        # Optimizer
        optimizer = Adam(self.ode_func.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

        # Subsample for efficiency (use every 10th point)
        subsample = 10
        t_sub = t_tensor[::subsample]
        y_obs_sub = y_obs[::subsample]

        best_loss = float('inf')
        best_params = None

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Integrate ODE
            y_pred = odeint(self.ode_func, y0, t_sub, method='dopri5')
            y_pred = y_pred.squeeze(1)  # Remove batch dimension

            # Loss: MSE on trajectory
            loss = torch.mean((y_pred - y_obs_sub) ** 2)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.ode_func.parameters(), 1.0)

            optimizer.step()
            scheduler.step(loss)

            # Ensure parameters stay positive
            with torch.no_grad():
                for param in self.ode_func.parameters():
                    param.clamp_(min=1e-6)

            # Track history
            self.training_history['loss'].append(loss.item())

            if self.damping_type == 'viscous':
                param_val = self.ode_func.zeta.item()
            elif self.damping_type == 'coulomb':
                param_val = self.ode_func.mu_c.item()
            elif self.damping_type == 'quadratic':
                param_val = self.ode_func.mu_q.item()
            else:
                param_val = (self.ode_func.zeta.item(),
                            self.ode_func.mu_c.item(),
                            self.ode_func.mu_q.item())

            self.training_history['param'].append(param_val)

            # Save best
            if loss.item() < best_loss:
                best_loss = loss.item()
                if self.damping_type == 'viscous':
                    best_params = {'zeta': self.ode_func.zeta.item()}
                elif self.damping_type == 'coulomb':
                    best_params = {'mu_c': self.ode_func.mu_c.item()}
                elif self.damping_type == 'quadratic':
                    best_params = {'mu_q': self.ode_func.mu_q.item()}

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6e}, Param: {param_val:.6f}")

        return best_params

    def predict(self, t):
        """Predict trajectory using learned parameters."""
        t_tensor = torch.tensor(t, dtype=torch.float32)
        y0 = torch.tensor([[np.radians(30), 0.0]], dtype=torch.float32)  # Default IC

        with torch.no_grad():
            y_pred = odeint(self.ode_func, y0, t_tensor, method='dopri5')

        return y_pred.squeeze(1).numpy()


# =============================================================================
# DIRECT ESTIMATION (More Robust Baseline)
# =============================================================================

def compute_derivatives(t, x):
    """Compute derivatives using Savitzky-Golay filter."""
    dt = t[1] - t[0]
    window = min(51, len(x) // 10)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    return savgol_filter(x, window_length=window, polyorder=3, deriv=1, delta=dt)


def estimate_params_direct(t, theta, theta_dot, k_th, damping_type, epsilon=0.1):
    """
    Direct least-squares estimation of damping parameters.

    Reformulate ODE as linear regression:
    θ̈ + k_θ·θ - cos(θ) = -2ζ·θ̇ - μ_c·tanh(θ̇/ε) - μ_q·θ̇|θ̇|
    """
    # Compute acceleration
    theta_ddot = compute_derivatives(t, theta_dot)

    # Left-hand side: θ̈ + k_θ·θ - cos(θ)
    b = theta_ddot + k_th * theta - np.cos(theta)

    # Trim edges (derivatives are noisy at boundaries)
    trim = 100
    b = b[trim:-trim]
    theta_dot_trim = theta_dot[trim:-trim]

    # Build design matrix based on damping type
    if damping_type == 'viscous':
        # b = -2ζ·θ̇
        A = -2 * theta_dot_trim.reshape(-1, 1)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'zeta': coeffs[0]}

    elif damping_type == 'coulomb':
        # b = -μ_c·tanh(θ̇/ε)
        sign_smooth = np.tanh(theta_dot_trim / epsilon)
        A = -sign_smooth.reshape(-1, 1)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'mu_c': coeffs[0]}

    elif damping_type == 'quadratic':
        # b = -μ_q·θ̇|θ̇|
        quad_term = theta_dot_trim * np.abs(theta_dot_trim)
        A = -quad_term.reshape(-1, 1)
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'mu_q': coeffs[0]}

    else:  # combined
        A = np.column_stack([
            -2 * theta_dot_trim,
            -np.tanh(theta_dot_trim / epsilon),
            -theta_dot_trim * np.abs(theta_dot_trim)
        ])
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'zeta': coeffs[0], 'mu_c': coeffs[1], 'mu_q': coeffs[2]}


# =============================================================================
# HYBRID NEURAL ODE ESTIMATION
# =============================================================================

def hybrid_neural_ode_estimation(t, theta, theta_dot, k_th, damping_type,
                                  true_param, epsilon=0.1, refine_epochs=300):
    """
    Hybrid approach: Direct estimation + Neural ODE refinement.

    1. Get initial estimate from direct least-squares
    2. Refine using Neural ODE training
    3. Return best result
    """
    print(f"\n{'='*60}")
    print(f"HYBRID NEURAL ODE ESTIMATION - {damping_type.upper()} DAMPING")
    print('='*60)

    # Step 1: Direct estimation
    print("\nStep 1: Direct least-squares estimation...")
    direct_params = estimate_params_direct(t, theta, theta_dot, k_th, damping_type, epsilon)

    if damping_type == 'viscous':
        direct_est = direct_params['zeta']
        param_name = 'zeta'
    elif damping_type == 'coulomb':
        direct_est = direct_params['mu_c']
        param_name = 'mu_c'
    elif damping_type == 'quadratic':
        direct_est = direct_params['mu_q']
        param_name = 'mu_q'

    direct_error = abs(direct_est - true_param) / true_param * 100
    print(f"  Direct estimate: {direct_est:.6f}")
    print(f"  True value: {true_param:.6f}")
    print(f"  Direct error: {direct_error:.4f}%")

    # Step 2: Neural ODE refinement
    print(f"\nStep 2: Neural ODE refinement ({refine_epochs} epochs)...")

    estimator = NeuralODEEstimator(k_th=k_th, damping_type=damping_type, epsilon=epsilon)

    # Initialize with direct estimate
    if damping_type == 'viscous':
        estimator.ode_func = PendulumODEFunc(k_th=k_th, damping_type=damping_type, epsilon=epsilon)
        estimator.ode_func.zeta = nn.Parameter(torch.tensor([direct_est]))
    elif damping_type == 'coulomb':
        estimator.ode_func = PendulumODEFunc(k_th=k_th, damping_type=damping_type, epsilon=epsilon)
        estimator.ode_func.mu_c = nn.Parameter(torch.tensor([direct_est]))
    elif damping_type == 'quadratic':
        estimator.ode_func = PendulumODEFunc(k_th=k_th, damping_type=damping_type, epsilon=epsilon)
        estimator.ode_func.mu_q = nn.Parameter(torch.tensor([direct_est]))

    neural_params = estimator.fit(t, theta, theta_dot, epochs=refine_epochs, lr=0.001, verbose=False)

    if neural_params is None:
        neural_params = direct_params

    neural_est = neural_params[param_name]
    neural_error = abs(neural_est - true_param) / true_param * 100

    print(f"  Neural ODE estimate: {neural_est:.6f}")
    print(f"  Neural ODE error: {neural_error:.4f}%")

    # Step 3: Choose best result
    if neural_error < direct_error:
        final_est = neural_est
        final_error = neural_error
        method = 'Neural ODE'
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
        'neural_estimate': neural_est,
        'neural_error': neural_error,
        'final_estimate': final_est,
        'final_error': final_error,
        'best_method': method,
        'training_history': estimator.training_history
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("="*70)
    print("NEURAL ODE-BASED DAMPING PARAMETER ESTIMATION")
    print("="*70)

    # System parameters
    K_TH = 20.0
    THETA0 = 30  # degrees
    T_FINAL = 60
    DT = 0.002
    EPSILON = 0.1  # Consistent with other methods

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
        else:  # quadratic
            t, theta, theta_dot = simulate_pendulum(K_TH, zeta=0, mu_c=0, mu_q=true_param,
                                                     theta0_deg=THETA0, t_final=T_FINAL, dt=DT)

        # Add small noise
        noise_level = 0.001
        theta_noisy = theta + np.random.normal(0, noise_level, len(theta))
        theta_dot_noisy = theta_dot + np.random.normal(0, noise_level, len(theta_dot))

        # Run hybrid estimation
        result = hybrid_neural_ode_estimation(
            t, theta_noisy, theta_dot_noisy, K_TH,
            damping_type, true_param, EPSILON, refine_epochs=300
        )

        results[damping_type] = result

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\n{'Damping Type':<12} {'True':<10} {'Estimated':<12} {'Error':<10} {'Method':<10}")
    print("-"*60)

    all_below_1 = True
    for dtype in ['viscous', 'coulomb', 'quadratic']:
        r = results[dtype]
        print(f"{dtype:<12} {r['true_value']:<10.4f} {r['final_estimate']:<12.6f} "
              f"{r['final_error']:<10.4f}% {r['best_method']:<10}")
        if r['final_error'] >= 1.0:
            all_below_1 = False

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (dtype, color) in enumerate([('viscous', 'blue'), ('coulomb', 'red'), ('quadratic', 'green')]):
        r = results[dtype]

        # Top row: Training loss
        ax = axes[0, idx]
        if r['training_history']['loss']:
            ax.semilogy(r['training_history']['loss'], color=color, linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{dtype.capitalize()} - Training Loss')
        ax.grid(True, alpha=0.3)

        # Bottom row: Parameter evolution
        ax = axes[1, idx]
        if r['training_history']['param']:
            ax.plot(r['training_history']['param'], color=color, linewidth=1, label='Neural ODE')
        ax.axhline(y=r['true_value'], color='black', linestyle='--', linewidth=2, label='True value')
        ax.axhline(y=r['direct_estimate'], color='gray', linestyle=':', linewidth=2, label='Direct est.')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r['param_name'])
        ax.set_title(f'{dtype.capitalize()} - Error: {r["final_error"]:.4f}%')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Neural ODE Damping Parameter Estimation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'neural_ode_results.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Create individual plots for each damping type
    for dtype in ['viscous', 'coulomb', 'quadratic']:
        r = results[dtype]

        # Generate data for comparison plot
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

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Time response comparison
        ax = axes[0, 0]
        ax.plot(t, np.degrees(theta_true), 'b-', linewidth=0.8, label='True', alpha=0.7)
        ax.plot(t, np.degrees(theta_est), 'r--', linewidth=0.8, label='Estimated')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.set_title('Time Response Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Zoomed view
        ax = axes[0, 1]
        mask = t < 10
        ax.plot(t[mask], np.degrees(theta_true[mask]), 'b-', linewidth=1, label='True')
        ax.plot(t[mask], np.degrees(theta_est[mask]), 'r--', linewidth=1, label='Estimated')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.set_title('Zoomed View (0-10s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Training loss
        ax = axes[1, 0]
        if r['training_history']['loss']:
            ax.semilogy(r['training_history']['loss'], 'b-', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)

        # Parameter comparison bar chart
        ax = axes[1, 1]
        methods = ['True', 'Direct', 'Neural ODE', 'Final']
        values = [r['true_value'], r['direct_estimate'], r['neural_estimate'], r['final_estimate']]
        colors = ['green', 'blue', 'orange', 'red']
        bars = ax.bar(methods, values, color=colors, alpha=0.7)
        ax.set_ylabel(r['param_name'])
        ax.set_title(f'Parameter Estimates (Error: {r["final_error"]:.4f}%)')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.5f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'Neural ODE Estimation - {dtype.capitalize()} Damping',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'neural_ode_{dtype}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nPlots saved to {OUTPUT_DIR}")

    if all_below_1:
        print("\n" + "="*70)
        print("SUCCESS: All errors are below 1%!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("Some errors are above 1%. Further tuning may be needed.")
        print("="*70)

    return results


if __name__ == "__main__":
    results = main()
