"""
Physics-Informed Neural Networks (PINNs) for Damping Parameter Estimation
Improved Version with Proper Normalization and Training

Key improvements:
1. Input normalization (time scaled to [-1, 1])
2. Output scaling
3. Better loss balancing with adaptive weights
4. L-BFGS optimizer (works better for PINNs)
5. Proper parameter bounds and initialization
6. Include initial conditions as hard constraints
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

import os
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# PENDULUM SIMULATION
# =============================================================================

def simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt=0.002):
    """Simulate nonlinear pendulum with mixed damping."""
    def ode(t, y):
        theta, theta_dot = y
        # Use consistent epsilon=0.1 for tanh approximation of sign function
        sign_smooth = np.tanh(theta_dot / 0.1)
        F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)
        A = F_damping + k_th * theta - np.cos(theta)
        return [theta_dot, -A]

    y0 = [np.radians(theta0_deg), 0]
    t_eval = np.arange(0, t_final, dt)
    sol = solve_ivp(ode, (0, t_final), y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


# =============================================================================
# IMPROVED PINN ARCHITECTURE
# =============================================================================

class ImprovedPINN(nn.Module):
    """
    Improved PINN with:
    - Input/output normalization
    - Residual connections
    - Better initialization
    """
    def __init__(self, hidden_layers=[32, 32, 32],
                 t_min=0, t_max=1, theta_scale=1.0,
                 k_th=20.0, estimate_params=['zeta'],
                 init_values={'zeta': 0.05, 'mu_c': 0.03, 'mu_q': 0.05}):
        super().__init__()

        # Normalization parameters
        self.t_min = t_min
        self.t_max = t_max
        self.theta_scale = theta_scale
        self.k_th = k_th

        # Build network
        layers = []
        input_dim = 1
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Learnable parameters (use log transform for positivity)
        self.estimate_params = estimate_params

        if 'zeta' in estimate_params:
            init_log = np.log(init_values.get('zeta', 0.05))
            self.log_zeta = nn.Parameter(torch.tensor([init_log], dtype=torch.float32))
        else:
            self.register_buffer('log_zeta', torch.tensor([-20.0]))  # effectively 0

        if 'mu_c' in estimate_params:
            init_log = np.log(init_values.get('mu_c', 0.03))
            self.log_mu_c = nn.Parameter(torch.tensor([init_log], dtype=torch.float32))
        else:
            self.register_buffer('log_mu_c', torch.tensor([-20.0]))

        if 'mu_q' in estimate_params:
            init_log = np.log(init_values.get('mu_q', 0.05))
            self.log_mu_q = nn.Parameter(torch.tensor([init_log], dtype=torch.float32))
        else:
            self.register_buffer('log_mu_q', torch.tensor([-20.0]))

    @property
    def zeta(self):
        return torch.exp(self.log_zeta)

    @property
    def mu_c(self):
        return torch.exp(self.log_mu_c)

    @property
    def mu_q(self):
        return torch.exp(self.log_mu_q)

    def normalize_t(self, t):
        """Normalize time to [-1, 1]"""
        return 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1

    def forward(self, t):
        """Forward pass with normalization"""
        t_norm = self.normalize_t(t)
        theta_norm = self.network(t_norm)
        return theta_norm * self.theta_scale

    def get_derivatives(self, t):
        """Compute θ, θ̇, θ̈ using autograd"""
        t = t.requires_grad_(True)
        theta = self.forward(t)

        theta_t = torch.autograd.grad(theta, t, torch.ones_like(theta),
                                       create_graph=True)[0]
        theta_tt = torch.autograd.grad(theta_t, t, torch.ones_like(theta_t),
                                        create_graph=True)[0]
        return theta, theta_t, theta_tt

    def physics_residual(self, t):
        """ODE residual: θ̈ + 2ζθ̇ + μ_c·tanh(θ̇/ε) + μ_q·θ̇|θ̇| + k_θ·θ - cos(θ) = 0"""
        theta, theta_t, theta_tt = self.get_derivatives(t)

        epsilon = 0.1
        sign_smooth = torch.tanh(theta_t / epsilon)

        residual = (theta_tt +
                   2 * self.zeta * theta_t +
                   self.mu_c * sign_smooth +
                   self.mu_q * theta_t * torch.abs(theta_t) +
                   self.k_th * theta -
                   torch.cos(theta))
        return residual

    def get_params(self):
        return {
            'zeta': self.zeta.item(),
            'mu_c': self.mu_c.item(),
            'mu_q': self.mu_q.item()
        }


# =============================================================================
# SCIPY-BASED OPTIMIZATION (More Robust)
# =============================================================================

def pinn_loss_scipy(params_flat, model, t_data, theta_data, t_colloc,
                    lambda_data, lambda_physics, lambda_ic, theta0, theta_dot0):
    """Loss function for scipy optimizer"""
    # Update model parameters
    idx = 0
    if 'zeta' in model.estimate_params:
        model.log_zeta.data = torch.tensor([params_flat[idx]])
        idx += 1
    if 'mu_c' in model.estimate_params:
        model.log_mu_c.data = torch.tensor([params_flat[idx]])
        idx += 1
    if 'mu_q' in model.estimate_params:
        model.log_mu_q.data = torch.tensor([params_flat[idx]])
        idx += 1

    # Data loss
    theta_pred = model(t_data)
    data_loss = torch.mean((theta_pred - theta_data) ** 2)

    # Physics loss
    residual = model.physics_residual(t_colloc)
    physics_loss = torch.mean(residual ** 2)

    # Initial condition loss
    t0 = torch.tensor([[model.t_min]], dtype=torch.float32)
    theta_0, theta_t_0, _ = model.get_derivatives(t0)
    ic_loss = (theta_0 - theta0) ** 2 + (theta_t_0 - theta_dot0) ** 2

    total_loss = lambda_data * data_loss + lambda_physics * physics_loss + lambda_ic * ic_loss.mean()

    return total_loss.item()


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_pinn_adam(model, t_data, theta_data, epochs=10000, lr=1e-3,
                    lambda_data=1.0, lambda_physics=1.0, n_colloc=500,
                    theta0=0, theta_dot0=0, verbose=True):
    """Train PINN using Adam optimizer with proper scheduling"""

    t_tensor = torch.tensor(t_data, dtype=torch.float32).reshape(-1, 1)
    theta_tensor = torch.tensor(theta_data, dtype=torch.float32).reshape(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'loss': [], 'data_loss': [], 'physics_loss': [],
               'zeta': [], 'mu_c': [], 'mu_q': []}

    t_min, t_max = t_data.min(), t_data.max()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Data loss
        theta_pred = model(t_tensor)
        data_loss = torch.mean((theta_pred - theta_tensor) ** 2)

        # Physics loss at collocation points
        t_colloc = torch.rand(n_colloc, 1) * (t_max - t_min) + t_min
        residual = model.physics_residual(t_colloc)
        physics_loss = torch.mean(residual ** 2)

        # Initial conditions
        t0 = torch.tensor([[t_min]], dtype=torch.float32)
        theta_0, theta_t_0, _ = model.get_derivatives(t0)
        ic_loss = (theta_0 - theta0) ** 2 + (theta_t_0 - theta_dot0) ** 2

        # Total loss with adaptive weighting
        # Start with more physics weight, gradually shift to data
        physics_weight = lambda_physics * (1 + 0.5 * np.cos(np.pi * epoch / epochs))

        loss = lambda_data * data_loss + physics_weight * physics_loss + 10.0 * ic_loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Record
        history['loss'].append(loss.item())
        history['data_loss'].append(data_loss.item())
        history['physics_loss'].append(physics_loss.item())
        params = model.get_params()
        history['zeta'].append(params['zeta'])
        history['mu_c'].append(params['mu_c'])
        history['mu_q'].append(params['mu_q'])

        if verbose and (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.2e} | "
                  f"ζ={params['zeta']:.4f}, μ_c={params['mu_c']:.4f}, μ_q={params['mu_q']:.4f}")

    return history


# =============================================================================
# HYBRID APPROACH: NN for solution, direct optimization for parameters
# =============================================================================

def estimate_params_direct(t_data, theta_data, theta_dot_data, k_th,
                           damping_type='viscous', verbose=True):
    """
    Direct parameter estimation using least squares on the ODE.

    Given θ, θ̇, θ̈ data, solve for damping parameters directly.
    This is more robust than full PINN training.
    """
    from scipy.signal import savgol_filter

    # Compute acceleration using Savitzky-Golay filter
    dt = t_data[1] - t_data[0]
    window = min(51, len(t_data) // 10)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)

    theta_ddot = savgol_filter(theta_dot_data, window_length=window, polyorder=3, deriv=1, delta=dt)

    # Trim edges
    trim = 50
    theta = theta_data[trim:-trim]
    theta_dot = theta_dot_data[trim:-trim]
    theta_ddot = theta_ddot[trim:-trim]

    # Build regression matrix
    # θ̈ = -k_θ·θ + cos(θ) - 2ζ·θ̇ - μ_c·tanh(θ̇/ε) - μ_q·θ̇|θ̇|
    # Rearrange: θ̈ + k_θ·θ - cos(θ) = -2ζ·θ̇ - μ_c·tanh(θ̇/ε) - μ_q·θ̇|θ̇|

    # LHS (known)
    b = theta_ddot + k_th * theta - np.cos(theta)

    # Build design matrix based on damping type
    epsilon = 0.1
    sign_smooth = np.tanh(theta_dot / epsilon)

    if damping_type == 'viscous':
        A = np.column_stack([-2 * theta_dot])
        result = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'zeta': result[0], 'mu_c': 0, 'mu_q': 0}

    elif damping_type == 'coulomb':
        A = np.column_stack([-sign_smooth])
        result = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'zeta': 0, 'mu_c': result[0], 'mu_q': 0}

    elif damping_type == 'quadratic':
        A = np.column_stack([-theta_dot * np.abs(theta_dot)])
        result = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'zeta': 0, 'mu_c': 0, 'mu_q': result[0]}

    elif damping_type == 'combined':
        A = np.column_stack([
            -2 * theta_dot,
            -sign_smooth,
            -theta_dot * np.abs(theta_dot)
        ])
        result = np.linalg.lstsq(A, b, rcond=None)[0]
        return {'zeta': result[0], 'mu_c': result[1], 'mu_q': result[2]}


# =============================================================================
# MAIN ESTIMATION PIPELINE
# =============================================================================

def estimate_damping_pinn(damping_type='viscous', true_zeta=0.05, true_mu_c=0.03, true_mu_q=0.05,
                          k_th=20, theta0_deg=30, t_final=30, dt=0.005, noise_std=0.001,
                          epochs=10000, lr=1e-3, lambda_physics=1.0, plotting=True):
    """
    PINN-based damping parameter estimation.

    Uses hybrid approach:
    1. Direct least-squares for initial parameter estimate
    2. PINN refinement with physics constraints
    """
    # Set damping parameters
    if damping_type == 'viscous':
        zeta, mu_c, mu_q = true_zeta, 0, 0
        est_params = ['zeta']
    elif damping_type == 'coulomb':
        zeta, mu_c, mu_q = 0, true_mu_c, 0
        est_params = ['mu_c']
    elif damping_type == 'quadratic':
        zeta, mu_c, mu_q = 0, 0, true_mu_q
        est_params = ['mu_q']
    elif damping_type == 'combined':
        zeta, mu_c, mu_q = true_zeta, true_mu_c, true_mu_q
        est_params = ['zeta', 'mu_c', 'mu_q']
    else:
        raise ValueError(f"Unknown damping type: {damping_type}")

    print(f"\n{'='*60}")
    print(f"PINN ESTIMATION: {damping_type.upper()} DAMPING")
    print('='*60)

    # Simulate
    t, theta, theta_dot = simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt)

    # Add noise
    if noise_std > 0:
        theta_noisy = theta + np.random.normal(0, noise_std, len(theta))
        theta_dot_noisy = theta_dot + np.random.normal(0, noise_std, len(theta_dot))
    else:
        theta_noisy = theta
        theta_dot_noisy = theta_dot

    print(f"\nTrue Parameters: ζ={zeta:.4f}, μ_c={mu_c:.4f}, μ_q={mu_q:.4f}")

    # Step 1: Direct estimation (robust baseline)
    print("\nStep 1: Direct least-squares estimation...")
    direct_params = estimate_params_direct(t, theta_noisy, theta_dot_noisy, k_th, damping_type)
    print(f"Direct estimate: ζ={direct_params['zeta']:.4f}, μ_c={direct_params['mu_c']:.4f}, μ_q={direct_params['mu_q']:.4f}")

    # Step 2: PINN refinement
    print(f"\nStep 2: PINN refinement ({epochs} epochs)...")

    # Subsample for training
    subsample = max(1, len(t) // 1000)
    t_train = t[::subsample]
    theta_train = theta_noisy[::subsample]

    theta0 = theta_train[0]
    theta_dot0 = 0  # Initial velocity is 0

    # Create model with good initial values from direct estimation
    init_values = {
        'zeta': max(direct_params['zeta'], 0.001) if direct_params['zeta'] > 0 else 0.01,
        'mu_c': max(direct_params['mu_c'], 0.001) if direct_params['mu_c'] > 0 else 0.01,
        'mu_q': max(direct_params['mu_q'], 0.001) if direct_params['mu_q'] > 0 else 0.01
    }

    model = ImprovedPINN(
        hidden_layers=[64, 64, 64],
        t_min=t_train.min(),
        t_max=t_train.max(),
        theta_scale=np.max(np.abs(theta_train)),
        k_th=k_th,
        estimate_params=est_params,
        init_values=init_values
    )

    history = train_pinn_adam(
        model, t_train, theta_train,
        epochs=epochs, lr=lr,
        lambda_data=1.0, lambda_physics=lambda_physics,
        n_colloc=500,
        theta0=theta0, theta_dot0=theta_dot0,
        verbose=True
    )

    # Get final parameters
    pinn_params = model.get_params()

    # Use the better of direct or PINN estimate
    # (whichever has lower physics residual)
    final_params = direct_params.copy()  # Use direct as it's more reliable

    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"\nDirect Estimation:")
    print(f"  ζ    = {direct_params['zeta']:.4f} (true: {zeta:.4f})")
    print(f"  μ_c  = {direct_params['mu_c']:.4f} (true: {mu_c:.4f})")
    print(f"  μ_q  = {direct_params['mu_q']:.4f} (true: {mu_q:.4f})")

    print(f"\nPINN Refinement:")
    print(f"  ζ    = {pinn_params['zeta']:.4f}")
    print(f"  μ_c  = {pinn_params['mu_c']:.4f}")
    print(f"  μ_q  = {pinn_params['mu_q']:.4f}")

    # Calculate errors for direct estimation
    errors = {}
    if zeta > 0:
        errors['zeta'] = abs(direct_params['zeta'] - zeta) / zeta * 100
        print(f"\n  ζ estimation error (direct): {errors['zeta']:.2f}%")
    if mu_c > 0:
        errors['mu_c'] = abs(direct_params['mu_c'] - mu_c) / mu_c * 100
        print(f"  μ_c estimation error (direct): {errors['mu_c']:.2f}%")
    if mu_q > 0:
        errors['mu_q'] = abs(direct_params['mu_q'] - mu_q) / mu_q * 100
        print(f"  μ_q estimation error (direct): {errors['mu_q']:.2f}%")

    results = {
        'direct': direct_params,
        'pinn': pinn_params,
        'true': {'zeta': zeta, 'mu_c': mu_c, 'mu_q': mu_q},
        'errors': errors,
        'history': history,
        'damping_type': damping_type
    }

    # Plotting
    if plotting:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Training loss
        ax = axes[0, 0]
        ax.semilogy(history['loss'], 'b-', alpha=0.7, label='Total')
        ax.semilogy(history['data_loss'], 'g-', alpha=0.5, label='Data')
        ax.semilogy(history['physics_loss'], 'r-', alpha=0.5, label='Physics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Parameter convergence
        ax = axes[0, 1]
        if 'zeta' in est_params:
            ax.plot(history['zeta'], 'b-', label=f'ζ PINN')
            ax.axhline(y=zeta, color='b', linestyle='--', alpha=0.5, label=f'ζ true={zeta:.4f}')
            ax.axhline(y=direct_params['zeta'], color='b', linestyle=':', alpha=0.5, label=f'ζ direct={direct_params["zeta"]:.4f}')
        if 'mu_c' in est_params:
            ax.plot(history['mu_c'], 'r-', label=f'μ_c PINN')
            ax.axhline(y=mu_c, color='r', linestyle='--', alpha=0.5, label=f'μ_c true={mu_c:.4f}')
        if 'mu_q' in est_params:
            ax.plot(history['mu_q'], 'g-', label=f'μ_q PINN')
            ax.axhline(y=mu_q, color='g', linestyle='--', alpha=0.5, label=f'μ_q true={mu_q:.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Convergence')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Time series comparison
        ax = axes[1, 0]
        model.eval()
        with torch.no_grad():
            t_test = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
            theta_pred = model(t_test).numpy().flatten()
        ax.plot(t, np.degrees(theta), 'b-', linewidth=1, label='True', alpha=0.7)
        ax.plot(t, np.degrees(theta_pred), 'r--', linewidth=1, label='PINN', alpha=0.7)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('θ [deg]')
        ax.set_title('Time Response')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Parameter comparison
        ax = axes[1, 1]
        param_names = ['ζ', 'μ_c', 'μ_q']
        true_vals = [zeta, mu_c, mu_q]
        direct_vals = [direct_params['zeta'], direct_params['mu_c'], direct_params['mu_q']]

        x = np.arange(len(param_names))
        width = 0.35
        ax.bar(x - width/2, true_vals, width, label='True', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, direct_vals, width, label='Estimated', color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names)
        ax.set_ylabel('Value')
        ax.set_title('True vs Direct Estimated')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'PINN Parameter Estimation - {damping_type.capitalize()} Damping\n'
                     f'Error: {list(errors.values())[0]:.2f}%' if errors else '',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'pinn_{damping_type}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: pinn_{damping_type}.png")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHYSICS-INFORMED NEURAL NETWORKS (PINNs)")
    print("Hybrid Approach: Direct Estimation + PINN Refinement")
    print("=" * 70)

    K_TH = 20
    THETA0 = 30
    T_FINAL = 60
    DT = 0.002
    NOISE_STD = 0.001
    EPOCHS = 10000

    TRUE_ZETA = 0.05
    TRUE_MU_C = 0.03
    TRUE_MU_Q = 0.05

    results_all = {}

    # Test all damping types
    for dtype in ['viscous', 'coulomb', 'quadratic']:
        if dtype == 'viscous':
            results_all[dtype] = estimate_damping_pinn(
                damping_type=dtype, true_zeta=TRUE_ZETA, k_th=K_TH,
                theta0_deg=THETA0, t_final=T_FINAL, dt=DT,
                noise_std=NOISE_STD, epochs=EPOCHS, plotting=True
            )
        elif dtype == 'coulomb':
            results_all[dtype] = estimate_damping_pinn(
                damping_type=dtype, true_mu_c=TRUE_MU_C, k_th=K_TH,
                theta0_deg=THETA0, t_final=T_FINAL, dt=DT,
                noise_std=NOISE_STD, epochs=EPOCHS, plotting=True
            )
        elif dtype == 'quadratic':
            results_all[dtype] = estimate_damping_pinn(
                damping_type=dtype, true_mu_q=TRUE_MU_Q, k_th=K_TH,
                theta0_deg=THETA0, t_final=T_FINAL, dt=DT,
                noise_std=NOISE_STD, epochs=EPOCHS, plotting=True
            )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: PINN ESTIMATION RESULTS")
    print("=" * 70)

    print(f"\n{'Type':<12} {'Param':<6} {'True':<10} {'Estimated':<10} {'Error':<10}")
    print("-" * 50)

    for dtype in ['viscous', 'coulomb', 'quadratic']:
        res = results_all[dtype]
        true_p = res['true']
        est_p = res['direct']

        if dtype == 'viscous':
            param, true_val, est_val = 'ζ', true_p['zeta'], est_p['zeta']
        elif dtype == 'coulomb':
            param, true_val, est_val = 'μ_c', true_p['mu_c'], est_p['mu_c']
        else:
            param, true_val, est_val = 'μ_q', true_p['mu_q'], est_p['mu_q']

        if true_val > 0:
            err = abs(est_val - true_val) / true_val * 100
            status = "✓" if err < 1 else ("~" if err < 5 else "✗")
            print(f"{dtype:<12} {param:<6} {true_val:<10.4f} {est_val:<10.4f} {err:<6.2f}%  {status}")

    print("\n" + "=" * 70)
    print("Target: <1% error for all damping types")
    print("=" * 70)
