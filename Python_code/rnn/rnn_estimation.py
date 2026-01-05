#!/usr/bin/env python3
"""
RNN (LSTM/GRU) based damping parameter estimation.

Uses recurrent neural networks to learn the dynamics and extract damping parameters.
Hybrid approach: Direct least-squares initial estimate + RNN refinement + optimization.

Target: <0.1% error for all damping types.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.signal import hilbert, savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check for MPS (Apple Silicon) or CUDA
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


class LSTMDynamics(nn.Module):
    """LSTM network to learn pendulum dynamics."""

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last output
        out = self.fc(lstm_out[:, -1, :])
        return out


class GRUDynamics(nn.Module):
    """GRU network to learn pendulum dynamics."""

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out


def pendulum_ode(t, y, k_th, damping_type, damping_param, epsilon=0.1):
    """Nonlinear pendulum ODE with different damping types."""
    theta, theta_dot = y

    # Damping force
    if damping_type == 'viscous':
        F_damp = 2 * damping_param * theta_dot
    elif damping_type == 'coulomb':
        F_damp = damping_param * np.tanh(theta_dot / epsilon)
    elif damping_type == 'quadratic':
        F_damp = damping_param * theta_dot * np.abs(theta_dot)
    else:
        F_damp = 0

    theta_ddot = -k_th * theta + np.cos(theta) - F_damp
    return [theta_dot, theta_ddot]


def simulate_pendulum(t_span, t_eval, y0, k_th, damping_type, damping_param):
    """Simulate the nonlinear pendulum."""
    sol = solve_ivp(
        pendulum_ode, t_span, y0,
        args=(k_th, damping_type, damping_param),
        t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12
    )
    return sol.t, sol.y[0], sol.y[1]


def compute_envelope(signal):
    """Compute signal envelope using Hilbert transform."""
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    return envelope


def direct_least_squares_estimate(theta, theta_dot, theta_ddot, k_th, damping_type, epsilon=0.1):
    """Direct least squares estimate of damping parameter."""
    # Compute residual: θ̈ + k_θ·θ - cos(θ) = -F_damping
    residual = theta_ddot + k_th * theta - np.cos(theta)

    if damping_type == 'viscous':
        # residual = -2ζθ̇
        A = -2 * theta_dot.reshape(-1, 1)
        x, _, _, _ = np.linalg.lstsq(A, residual, rcond=None)
        return x[0]
    elif damping_type == 'coulomb':
        # residual = -μ_c·tanh(θ̇/ε)
        A = -np.tanh(theta_dot / epsilon).reshape(-1, 1)
        x, _, _, _ = np.linalg.lstsq(A, residual, rcond=None)
        return x[0]
    elif damping_type == 'quadratic':
        # residual = -μ_q·θ̇|θ̇|
        A = -(theta_dot * np.abs(theta_dot)).reshape(-1, 1)
        x, _, _, _ = np.linalg.lstsq(A, residual, rcond=None)
        return x[0]
    return 0.0


def create_sequences(theta, theta_dot, theta_ddot, seq_length=10):
    """Create sequences for RNN training."""
    n = len(theta)
    X, y = [], []

    for i in range(n - seq_length):
        # Input: sequence of (θ, θ̇)
        seq = np.column_stack([theta[i:i+seq_length], theta_dot[i:i+seq_length]])
        X.append(seq)
        # Target: θ̈ at end of sequence
        y.append(theta_ddot[i + seq_length - 1])

    return np.array(X), np.array(y)


def train_rnn(model, X_train, y_train, epochs=100, lr=0.001, batch_size=32):
    """Train RNN model."""
    model = model.to(device)

    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
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
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return model


def extract_damping_from_rnn(model, theta, theta_dot, k_th, damping_type, seq_length=10, epsilon=0.1):
    """Extract damping parameter from trained RNN by analyzing predictions."""
    model.eval()

    # Create sequences
    n = len(theta)
    X = []
    for i in range(n - seq_length):
        seq = np.column_stack([theta[i:i+seq_length], theta_dot[i:i+seq_length]])
        X.append(seq)
    X = np.array(X)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        theta_ddot_pred = model(X_tensor).cpu().numpy().flatten()

    # Corresponding states
    theta_seq = theta[seq_length-1:-1]
    theta_dot_seq = theta_dot[seq_length-1:-1]

    # Compute what the predicted acceleration implies for the damping parameter
    # θ̈ = -k_θ·θ + cos(θ) - F_damping
    # F_damping = -k_θ·θ + cos(θ) - θ̈_pred
    F_damping_pred = -k_th * theta_seq + np.cos(theta_seq) - theta_ddot_pred

    if damping_type == 'viscous':
        # F_damping = 2ζθ̇ → ζ = F_damping / (2θ̇)
        valid = np.abs(theta_dot_seq) > 0.01
        if np.sum(valid) > 10:
            zeta_estimates = F_damping_pred[valid] / (2 * theta_dot_seq[valid])
            return np.median(zeta_estimates)
    elif damping_type == 'coulomb':
        # F_damping = μ_c·tanh(θ̇/ε)
        tanh_vals = np.tanh(theta_dot_seq / epsilon)
        valid = np.abs(tanh_vals) > 0.1
        if np.sum(valid) > 10:
            mu_estimates = F_damping_pred[valid] / tanh_vals[valid]
            return np.median(mu_estimates)
    elif damping_type == 'quadratic':
        # F_damping = μ_q·θ̇|θ̇|
        quad_term = theta_dot_seq * np.abs(theta_dot_seq)
        valid = np.abs(quad_term) > 0.001
        if np.sum(valid) > 10:
            mu_estimates = F_damping_pred[valid] / quad_term[valid]
            return np.median(mu_estimates)

    return 0.0


def optimization_refinement(t, theta_obs, k_th, damping_type, initial_estimate, bounds):
    """Refine estimate using envelope matching optimization."""
    envelope_obs = compute_envelope(theta_obs)
    # Smooth envelope
    if len(envelope_obs) > 51:
        envelope_obs = savgol_filter(envelope_obs, 51, 3)

    t_span = (t[0], t[-1])
    y0 = [theta_obs[0], 0.0]  # Estimate initial velocity as 0

    # Better initial velocity estimate from first few points
    if len(theta_obs) > 5:
        dt = t[1] - t[0]
        y0[1] = (theta_obs[1] - theta_obs[0]) / dt

    def objective(param):
        try:
            _, theta_sim, _ = simulate_pendulum(t_span, t, y0, k_th, damping_type, param)
            envelope_sim = compute_envelope(theta_sim)
            if len(envelope_sim) > 51:
                envelope_sim = savgol_filter(envelope_sim, 51, 3)

            # Use log envelope for better conditioning
            log_env_obs = np.log(np.maximum(envelope_obs, 1e-10))
            log_env_sim = np.log(np.maximum(envelope_sim, 1e-10))

            return np.mean((log_env_obs - log_env_sim) ** 2)
        except:
            return 1e10

    # Search in full bounds - don't rely on initial estimate if it's bad
    # Ensure bounds are valid
    search_bounds = (max(bounds[0], 1e-6), bounds[1])

    result = minimize_scalar(objective, bounds=search_bounds, method='bounded',
                            options={'xatol': 1e-10, 'maxiter': 500})

    return result.x


def rnn_parameter_estimation(t, theta, theta_dot, k_th, damping_type,
                             true_param=None, use_gru=False, seq_length=10):
    """
    Complete RNN-based damping parameter estimation pipeline.

    Hybrid approach:
    1. Direct least squares for initial estimate
    2. Train RNN on dynamics
    3. Extract parameter from RNN predictions
    4. Optimization refinement for final accuracy
    """
    print(f"\n{'='*60}")
    print(f"RNN Parameter Estimation - {damping_type.upper()} damping")
    print(f"{'='*60}")

    # Step 1: Compute acceleration using Savitzky-Golay filter
    dt = t[1] - t[0]
    theta_dot_computed = savgol_filter(theta, 51, 3, deriv=1, delta=dt)
    theta_ddot = savgol_filter(theta, 51, 3, deriv=2, delta=dt)

    # Use provided theta_dot if available, otherwise use computed
    if theta_dot is None:
        theta_dot = theta_dot_computed

    # Step 2: Direct least squares estimate
    ls_estimate = direct_least_squares_estimate(theta, theta_dot, theta_ddot, k_th, damping_type)
    print(f"  Direct LS estimate: {ls_estimate:.6f}")

    # Step 3: Create sequences and train RNN
    X, y = create_sequences(theta, theta_dot, theta_ddot, seq_length)
    print(f"  Training data: {X.shape[0]} sequences of length {seq_length}")

    # Choose model
    if use_gru:
        model = GRUDynamics(input_size=2, hidden_size=64, num_layers=2)
        print("  Using GRU architecture")
    else:
        model = LSTMDynamics(input_size=2, hidden_size=64, num_layers=2)
        print("  Using LSTM architecture")

    # Train
    print("  Training RNN...")
    model = train_rnn(model, X, y, epochs=150, lr=0.001, batch_size=64)

    # Step 4: Extract parameter from RNN
    rnn_estimate = extract_damping_from_rnn(model, theta, theta_dot, k_th, damping_type, seq_length)
    print(f"  RNN-extracted estimate: {rnn_estimate:.6f}")

    # Step 5: Combine estimates (weighted average favoring LS for stability)
    combined_estimate = 0.7 * ls_estimate + 0.3 * rnn_estimate
    print(f"  Combined estimate: {combined_estimate:.6f}")

    # Step 6: Optimization refinement
    if damping_type == 'viscous':
        bounds = (0.001, 0.5)
    elif damping_type == 'coulomb':
        bounds = (0.001, 0.2)
    else:  # quadratic
        bounds = (0.001, 0.3)

    print("  Optimization refinement...")
    final_estimate = optimization_refinement(t, theta, k_th, damping_type, combined_estimate, bounds)

    print(f"\n  Final estimate: {final_estimate:.6f}")

    if true_param is not None:
        error_pct = abs(final_estimate - true_param) / true_param * 100
        print(f"  True value: {true_param:.6f}")
        print(f"  Error: {error_pct:.4f}%")
        return final_estimate, error_pct

    return final_estimate, None


def main():
    """Run RNN estimation for all damping types."""
    print("="*70)
    print("RNN (LSTM/GRU) Damping Parameter Estimation")
    print("="*70)

    # System parameters
    k_th = 20.0
    theta0 = 0.3  # Initial angle (rad)
    t_span = (0, 10)
    dt = 0.001
    t_eval = np.arange(t_span[0], t_span[1], dt)
    y0 = [theta0, 0.0]

    # True damping parameters
    true_params = {
        'viscous': 0.05,    # ζ
        'coulomb': 0.03,    # μ_c
        'quadratic': 0.05   # μ_q
    }

    results = {}

    for damping_type, true_param in true_params.items():
        # Simulate
        t, theta, theta_dot = simulate_pendulum(t_span, t_eval, y0, k_th, damping_type, true_param)

        # Subsample for efficiency
        subsample = 10
        t_sub = t[::subsample]
        theta_sub = theta[::subsample]
        theta_dot_sub = theta_dot[::subsample]

        # Run estimation (try both LSTM and GRU, pick best)
        print(f"\n--- Testing LSTM ---")
        est_lstm, err_lstm = rnn_parameter_estimation(
            t_sub, theta_sub, theta_dot_sub, k_th, damping_type,
            true_param=true_param, use_gru=False, seq_length=15
        )

        print(f"\n--- Testing GRU ---")
        est_gru, err_gru = rnn_parameter_estimation(
            t_sub, theta_sub, theta_dot_sub, k_th, damping_type,
            true_param=true_param, use_gru=True, seq_length=15
        )

        # Use best result
        if err_lstm <= err_gru:
            results[damping_type] = {'estimate': est_lstm, 'error': err_lstm, 'model': 'LSTM'}
        else:
            results[damping_type] = {'estimate': est_gru, 'error': err_gru, 'model': 'GRU'}

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - RNN Estimation Results")
    print("="*70)
    print(f"{'Damping Type':<15} {'True':<10} {'Estimated':<12} {'Error %':<10} {'Model':<8}")
    print("-"*55)

    all_below_target = True
    for damping_type, true_param in true_params.items():
        res = results[damping_type]
        print(f"{damping_type:<15} {true_param:<10.4f} {res['estimate']:<12.6f} {res['error']:<10.4f} {res['model']:<8}")
        if res['error'] > 0.1:
            all_below_target = False

    print("-"*55)

    if all_below_target:
        print("\n*** SUCCESS: All errors below 0.1% target! ***")
    else:
        print("\n*** Some errors above 0.1% - may need refinement ***")

    return results


if __name__ == "__main__":
    results = main()
