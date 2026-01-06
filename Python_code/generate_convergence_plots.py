#!/usr/bin/env python3
"""
Convergence Plots for Iterative Damping Parameter Estimation Methods

Generates parameter value vs epoch/iteration plots for all iterative methods:
1. Physics-Informed Neural Networks (PINNs)
2. Neural ODEs
3. Genetic Algorithm
4. RNN (LSTM/GRU)

Shows how parameter estimates evolve during training, demonstrating
convergence behavior for each method and damping type.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter, hilbert
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Device
device = torch.device('cpu')

# System parameters
K_TH = 20.0
THETA0_DEG = 30
T_FINAL = 30
DT = 0.002
EPSILON = 0.1

# True parameters
TRUE_PARAMS = {
    'viscous': 0.05,
    'coulomb': 0.03,
    'quadratic': 0.05
}


# =============================================================================
# COMMON SIMULATION FUNCTION
# =============================================================================

def simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0_deg, t_final, dt=0.002):
    """Simulate nonlinear pendulum with various damping types."""
    def ode(t, y):
        theta, theta_dot = y
        sign_smooth = np.tanh(theta_dot / EPSILON)
        F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)
        return [theta_dot, -F_damping - k_th * theta + np.cos(theta)]

    y0 = [np.radians(theta0_deg), 0]
    t_eval = np.arange(0, t_final, dt)
    sol = solve_ivp(ode, (0, t_final), y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10)
    return sol.t, sol.y[0], sol.y[1]


def get_damping_params(damping_type, true_param):
    """Return (zeta, mu_c, mu_q) tuple based on damping type."""
    if damping_type == 'viscous':
        return true_param, 0, 0
    elif damping_type == 'coulomb':
        return 0, true_param, 0
    else:  # quadratic
        return 0, 0, true_param


# =============================================================================
# PINN WITH CONVERGENCE TRACKING
# =============================================================================

class PINNConvergence:
    """PINN with parameter convergence tracking."""

    def __init__(self, k_th, damping_type, hidden_layers=[64, 64, 64]):
        self.k_th = k_th
        self.damping_type = damping_type
        self.hidden_layers = hidden_layers

    def build_network(self, t_min, t_max, theta_scale):
        """Build PINN network."""
        layers = []
        input_dim = 1
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)
        self.t_min = t_min
        self.t_max = t_max
        self.theta_scale = theta_scale

        # Initialize weights
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Learnable parameter (log-transform for positivity)
        self.log_param = nn.Parameter(torch.tensor([np.log(0.01)]))

    @property
    def param(self):
        return torch.exp(self.log_param)

    def normalize_t(self, t):
        return 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1

    def forward(self, t):
        t_norm = self.normalize_t(t)
        theta_norm = self.network(t_norm)
        return theta_norm * self.theta_scale

    def get_derivatives(self, t):
        t = t.requires_grad_(True)
        theta = self.forward(t)
        theta_t = torch.autograd.grad(theta, t, torch.ones_like(theta), create_graph=True)[0]
        theta_tt = torch.autograd.grad(theta_t, t, torch.ones_like(theta_t), create_graph=True)[0]
        return theta, theta_t, theta_tt

    def physics_residual(self, t):
        theta, theta_t, theta_tt = self.get_derivatives(t)

        if self.damping_type == 'viscous':
            F_damp = 2 * self.param * theta_t
        elif self.damping_type == 'coulomb':
            F_damp = self.param * torch.tanh(theta_t / EPSILON)
        else:  # quadratic
            F_damp = self.param * theta_t * torch.abs(theta_t)

        residual = theta_tt + F_damp + self.k_th * theta - torch.cos(theta)
        return residual

    def train(self, t_data, theta_data, epochs=5000, lr=1e-3):
        """Train PINN and track parameter convergence."""
        self.build_network(t_data.min(), t_data.max(), np.max(np.abs(theta_data)))

        t_tensor = torch.tensor(t_data, dtype=torch.float32).reshape(-1, 1)
        theta_tensor = torch.tensor(theta_data, dtype=torch.float32).reshape(-1, 1)

        params = list(self.network.parameters()) + [self.log_param]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'epoch': [], 'param': [], 'loss': []}
        t_min, t_max = t_data.min(), t_data.max()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Data loss
            theta_pred = self.forward(t_tensor)
            data_loss = torch.mean((theta_pred - theta_tensor) ** 2)

            # Physics loss
            t_colloc = torch.rand(200, 1) * (t_max - t_min) + t_min
            residual = self.physics_residual(t_colloc)
            physics_loss = torch.mean(residual ** 2)

            loss = data_loss + 0.1 * physics_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            # Record every 10 epochs
            if epoch % 10 == 0:
                history['epoch'].append(epoch)
                history['param'].append(self.param.item())
                history['loss'].append(loss.item())

        return history


# =============================================================================
# NEURAL ODE WITH CONVERGENCE TRACKING
# =============================================================================

class NeuralODEConvergence:
    """Neural ODE with parameter convergence tracking."""

    def __init__(self, k_th, damping_type):
        self.k_th = k_th
        self.damping_type = damping_type

    def train(self, t_data, theta_data, theta_dot_data, epochs=500, lr=0.01):
        """Train Neural ODE and track convergence."""
        from torchdiffeq import odeint

        class ODEFunc(nn.Module):
            def __init__(self, k_th, damping_type):
                super().__init__()
                self.k_th = k_th
                self.damping_type = damping_type
                self.param = nn.Parameter(torch.tensor([0.01]))

            def forward(self, t, y):
                theta = y[..., 0:1]
                theta_dot = y[..., 1:2]

                if self.damping_type == 'viscous':
                    F_damp = 2 * self.param * theta_dot
                elif self.damping_type == 'coulomb':
                    F_damp = self.param * torch.tanh(theta_dot / EPSILON)
                else:
                    F_damp = self.param * theta_dot * torch.abs(theta_dot)

                theta_ddot = -self.k_th * theta + torch.cos(theta) - F_damp
                return torch.cat([theta_dot, theta_ddot], dim=-1)

        ode_func = ODEFunc(self.k_th, self.damping_type)

        t_tensor = torch.tensor(t_data, dtype=torch.float32)
        y_obs = torch.stack([
            torch.tensor(theta_data, dtype=torch.float32),
            torch.tensor(theta_dot_data, dtype=torch.float32)
        ], dim=-1)

        y0 = y_obs[0:1]

        optimizer = torch.optim.Adam(ode_func.parameters(), lr=lr)

        # Subsample for efficiency
        subsample = 20
        t_sub = t_tensor[::subsample]
        y_obs_sub = y_obs[::subsample]

        history = {'epoch': [], 'param': [], 'loss': []}

        for epoch in range(epochs):
            optimizer.zero_grad()

            y_pred = odeint(ode_func, y0, t_sub, method='dopri5')
            y_pred = y_pred.squeeze(1)

            loss = torch.mean((y_pred - y_obs_sub) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                ode_func.param.clamp_(min=1e-6)

            # Record every 5 epochs
            if epoch % 5 == 0:
                history['epoch'].append(epoch)
                history['param'].append(ode_func.param.item())
                history['loss'].append(loss.item())

        return history


# =============================================================================
# GENETIC ALGORITHM WITH CONVERGENCE TRACKING
# =============================================================================

class GAConvergence:
    """Genetic Algorithm with parameter convergence tracking."""

    def __init__(self, k_th, damping_type):
        self.k_th = k_th
        self.damping_type = damping_type

    def train(self, t_obs, theta_obs, generations=100, pop_size=50):
        """Run GA and track convergence."""
        # Get observed envelope
        env_obs = np.abs(hilbert(theta_obs))

        def fitness(param):
            try:
                zeta, mu_c, mu_q = get_damping_params(self.damping_type, param)
                t_sim, theta_sim, _ = simulate_pendulum(
                    self.k_th, zeta, mu_c, mu_q, THETA0_DEG, T_FINAL, DT
                )
                env_sim = np.abs(hilbert(theta_sim))
                env_sim_interp = np.interp(t_obs, t_sim, env_sim)

                env_obs_safe = np.maximum(env_obs, 1e-10)
                env_sim_safe = np.maximum(env_sim_interp, 1e-10)
                mse = np.mean((np.log(env_obs_safe) - np.log(env_sim_safe))**2)
                return -mse  # GA maximizes
            except:
                return -1e10

        # GA parameters
        if self.damping_type == 'viscous':
            bounds = (0.001, 0.2)
        elif self.damping_type == 'coulomb':
            bounds = (0.001, 0.1)
        else:
            bounds = (0.001, 0.2)

        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], pop_size)

        history = {'generation': [], 'param': [], 'fitness': []}

        for gen in range(generations):
            # Evaluate fitness
            fit = np.array([fitness(ind) for ind in population])

            # Sort by fitness
            sorted_idx = np.argsort(fit)[::-1]
            population = population[sorted_idx]
            fit = fit[sorted_idx]

            # Record
            history['generation'].append(gen)
            history['param'].append(population[0])
            history['fitness'].append(fit[0])

            # Check convergence
            if np.std(population) < 1e-8:
                break

            # Create new population
            new_pop = [population[0], population[1]]  # Elitism

            while len(new_pop) < pop_size:
                # Tournament selection
                idx1 = np.argmax(fit[np.random.choice(len(population), 3, replace=False)])
                idx2 = np.argmax(fit[np.random.choice(len(population), 3, replace=False)])
                p1, p2 = population[idx1], population[idx2]

                # Crossover
                if np.random.random() < 0.9:
                    d = abs(p1 - p2)
                    c1 = np.random.uniform(max(bounds[0], min(p1, p2) - 0.5*d),
                                           min(bounds[1], max(p1, p2) + 0.5*d))
                else:
                    c1 = p1

                # Mutation
                if np.random.random() < 0.2:
                    c1 += np.random.normal(0, 0.05 * (bounds[1] - bounds[0]))
                    c1 = np.clip(c1, bounds[0], bounds[1])

                new_pop.append(c1)

            population = np.array(new_pop)

        return history


# =============================================================================
# RNN WITH CONVERGENCE TRACKING
# =============================================================================

class RNNConvergence:
    """RNN with parameter convergence tracking via periodic extraction."""

    def __init__(self, k_th, damping_type):
        self.k_th = k_th
        self.damping_type = damping_type

    def train(self, t_data, theta_data, theta_dot_data, epochs=200, seq_length=15):
        """Train RNN and track parameter convergence by extracting estimates periodically."""

        class LSTMDynamics(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(2, 64, 2, batch_first=True, dropout=0.1)
                self.fc = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])

        # Compute acceleration
        dt = t_data[1] - t_data[0]
        theta_ddot = savgol_filter(theta_data, 51, 3, deriv=2, delta=dt)

        # Create sequences
        n = len(theta_data)
        X, y = [], []
        for i in range(n - seq_length):
            seq = np.column_stack([theta_data[i:i+seq_length], theta_dot_data[i:i+seq_length]])
            X.append(seq)
            y.append(theta_ddot[i + seq_length - 1])
        X, y = np.array(X), np.array(y)

        model = LSTMDynamics().to(device)
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        history = {'epoch': [], 'param': [], 'loss': []}

        def extract_param_from_model(model):
            """Extract damping parameter from trained model predictions."""
            model.eval()
            with torch.no_grad():
                theta_ddot_pred = model(X_tensor).cpu().numpy().flatten()

            theta_seq = theta_data[seq_length-1:-1]
            theta_dot_seq = theta_dot_data[seq_length-1:-1]

            F_damp_pred = -self.k_th * theta_seq + np.cos(theta_seq) - theta_ddot_pred

            if self.damping_type == 'viscous':
                valid = np.abs(theta_dot_seq) > 0.01
                if np.sum(valid) > 10:
                    estimates = F_damp_pred[valid] / (2 * theta_dot_seq[valid])
                    return np.median(estimates)
            elif self.damping_type == 'coulomb':
                tanh_vals = np.tanh(theta_dot_seq / EPSILON)
                valid = np.abs(tanh_vals) > 0.1
                if np.sum(valid) > 10:
                    estimates = F_damp_pred[valid] / tanh_vals[valid]
                    return np.median(estimates)
            else:  # quadratic
                quad_term = theta_dot_seq * np.abs(theta_dot_seq)
                valid = np.abs(quad_term) > 0.001
                if np.sum(valid) > 10:
                    estimates = F_damp_pred[valid] / quad_term[valid]
                    return np.median(estimates)
            return 0.0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)

            # Extract parameter every 10 epochs
            if epoch % 10 == 0:
                param_est = extract_param_from_model(model)
                history['epoch'].append(epoch)
                history['param'].append(param_est)
                history['loss'].append(avg_loss)

        return history


# =============================================================================
# MAIN: GENERATE CONVERGENCE PLOTS
# =============================================================================

def generate_convergence_plots():
    """Generate convergence plots for all iterative methods."""
    print("=" * 70)
    print("GENERATING CONVERGENCE PLOTS")
    print("=" * 70)

    damping_types = ['viscous', 'coulomb', 'quadratic']
    methods = ['PINN', 'Neural ODE', 'Genetic Algorithm', 'RNN (LSTM)']

    # Store all histories
    all_histories = {method: {} for method in methods}

    for damping_type in damping_types:
        true_param = TRUE_PARAMS[damping_type]
        zeta, mu_c, mu_q = get_damping_params(damping_type, true_param)

        print(f"\n{'='*60}")
        print(f"Processing {damping_type.upper()} damping (true = {true_param})")
        print('='*60)

        # Generate data
        t, theta, theta_dot = simulate_pendulum(K_TH, zeta, mu_c, mu_q, THETA0_DEG, T_FINAL, DT)

        # Add small noise
        theta_noisy = theta + np.random.normal(0, 0.001, len(theta))
        theta_dot_noisy = theta_dot + np.random.normal(0, 0.001, len(theta_dot))

        # Subsample for training
        subsample = 5
        t_sub = t[::subsample]
        theta_sub = theta_noisy[::subsample]
        theta_dot_sub = theta_dot_noisy[::subsample]

        # 1. PINN
        print(f"\n  Training PINN...")
        pinn = PINNConvergence(K_TH, damping_type)
        history_pinn = pinn.train(t_sub, theta_sub, epochs=3000, lr=1e-3)
        all_histories['PINN'][damping_type] = history_pinn
        print(f"    Final estimate: {history_pinn['param'][-1]:.6f}")

        # 2. Neural ODE
        print(f"\n  Training Neural ODE...")
        try:
            node = NeuralODEConvergence(K_TH, damping_type)
            history_node = node.train(t_sub, theta_sub, theta_dot_sub, epochs=300, lr=0.01)
            all_histories['Neural ODE'][damping_type] = history_node
            print(f"    Final estimate: {history_node['param'][-1]:.6f}")
        except ImportError:
            print("    (torchdiffeq not available, skipping)")
            all_histories['Neural ODE'][damping_type] = None

        # 3. Genetic Algorithm
        print(f"\n  Running Genetic Algorithm...")
        ga = GAConvergence(K_TH, damping_type)
        history_ga = ga.train(t, theta_noisy, generations=100, pop_size=50)
        all_histories['Genetic Algorithm'][damping_type] = history_ga
        print(f"    Final estimate: {history_ga['param'][-1]:.6f}")

        # 4. RNN
        print(f"\n  Training RNN (LSTM)...")
        rnn = RNNConvergence(K_TH, damping_type)
        history_rnn = rnn.train(t_sub, theta_sub, theta_dot_sub, epochs=150, seq_length=15)
        all_histories['RNN (LSTM)'][damping_type] = history_rnn
        print(f"    Final estimate: {history_rnn['param'][-1]:.6f}")

    # ==========================================================================
    # Create Plots
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Creating convergence plots...")
    print("=" * 70)

    # Plot 1: Combined convergence plot (3x4 grid)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    colors = {'PINN': 'blue', 'Neural ODE': 'green', 'Genetic Algorithm': 'red', 'RNN (LSTM)': 'purple'}
    param_names = {'viscous': r'$\zeta$', 'coulomb': r'$\mu_c$', 'quadratic': r'$\mu_q$'}

    for i, damping_type in enumerate(damping_types):
        true_param = TRUE_PARAMS[damping_type]

        for j, method in enumerate(methods):
            ax = axes[i, j]
            history = all_histories[method][damping_type]

            if history is None:
                ax.text(0.5, 0.5, 'Not available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method}\n{damping_type.capitalize()}')
                continue

            # Get x-axis data
            if 'epoch' in history:
                x_data = history['epoch']
                x_label = 'Epoch'
            else:
                x_data = history['generation']
                x_label = 'Generation'

            # Plot parameter convergence
            ax.plot(x_data, history['param'], color=colors[method], linewidth=1.5, label='Estimated')
            ax.axhline(y=true_param, color='black', linestyle='--', linewidth=2, label=f'True = {true_param}')

            # Final estimate
            final_est = history['param'][-1]
            error = abs(final_est - true_param) / true_param * 100

            ax.set_xlabel(x_label)
            ax.set_ylabel(param_names[damping_type])
            ax.set_title(f'{method}\n{damping_type.capitalize()}: Error = {error:.2f}%')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.suptitle('Parameter Convergence During Training\nAll Iterative Methods',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_convergence_all_methods.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_convergence_all_methods.png")

    # Plot 2: Method comparison for each damping type (3 separate plots)
    for damping_type in damping_types:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        true_param = TRUE_PARAMS[damping_type]

        # Left: Parameter convergence
        ax = axes[0]
        for method in methods:
            history = all_histories[method][damping_type]
            if history is None:
                continue

            x_data = history.get('epoch', history.get('generation'))
            ax.plot(x_data, history['param'], color=colors[method], linewidth=1.5,
                    label=method, alpha=0.8)

        ax.axhline(y=true_param, color='black', linestyle='--', linewidth=2,
                   label=f'True {param_names[damping_type]} = {true_param}')
        ax.set_xlabel('Epoch / Generation')
        ax.set_ylabel(f'Parameter Value ({param_names[damping_type]})')
        ax.set_title(f'{damping_type.capitalize()} Damping: Parameter Convergence')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Right: Loss/Fitness convergence
        ax = axes[1]
        for method in methods:
            history = all_histories[method][damping_type]
            if history is None:
                continue

            x_data = history.get('epoch', history.get('generation'))

            if 'loss' in history:
                y_data = history['loss']
                ax.semilogy(x_data, y_data, color=colors[method], linewidth=1.5,
                           label=method, alpha=0.8)
            elif 'fitness' in history:
                y_data = [-f for f in history['fitness']]  # Convert to MSE (negative fitness)
                ax.semilogy(x_data, y_data, color=colors[method], linewidth=1.5,
                           label=method, alpha=0.8)

        ax.set_xlabel('Epoch / Generation')
        ax.set_ylabel('Loss / MSE (log scale)')
        ax.set_title(f'{damping_type.capitalize()} Damping: Training Loss')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Convergence Comparison: {damping_type.capitalize()} Damping',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'fig_convergence_{damping_type}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: fig_convergence_{damping_type}.png")

    # Plot 3: Final estimates comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(damping_types))
    width = 0.15
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    for j, method in enumerate(methods):
        final_values = []
        for damping_type in damping_types:
            history = all_histories[method][damping_type]
            if history is not None:
                final_values.append(history['param'][-1])
            else:
                final_values.append(0)

        ax.bar(x + offsets[j], final_values, width, label=method, color=colors[method], alpha=0.8)

    # True values
    true_values = [TRUE_PARAMS[dt] for dt in damping_types]
    for i, (tv, dt) in enumerate(zip(true_values, damping_types)):
        ax.scatter(i, tv, s=200, marker='*', color='gold', edgecolors='black',
                   zorder=5, label='True' if i == 0 else '')
        ax.annotate(f'{tv}', (i, tv), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Damping Type')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Final Parameter Estimates by Method')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{dt.capitalize()}\n({param_names[dt]})' for dt in damping_types])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_convergence_final_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_convergence_final_comparison.png")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CONVERGENCE RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Damping':<12} {'True':<10} {'Final Est':<12} {'Error %':<10}")
    print("-" * 70)

    for method in methods:
        for damping_type in damping_types:
            history = all_histories[method][damping_type]
            true_param = TRUE_PARAMS[damping_type]

            if history is not None:
                final_est = history['param'][-1]
                error = abs(final_est - true_param) / true_param * 100
                print(f"{method:<20} {damping_type:<12} {true_param:<10.4f} {final_est:<12.6f} {error:<10.2f}")
            else:
                print(f"{method:<20} {damping_type:<12} {true_param:<10.4f} {'N/A':<12} {'N/A':<10}")

    print("\n" + "=" * 70)
    print(f"Plots saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_convergence_plots()
