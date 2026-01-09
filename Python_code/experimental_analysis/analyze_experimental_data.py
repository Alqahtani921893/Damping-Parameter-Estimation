#!/usr/bin/env python3
"""
Experimental Data Analysis for Horizontal Pendulum
Estimates stiffness (k_θ) and damping parameters from real experimental data.

Author: Generated for damping parameter estimation project
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, hilbert
from scipy.optimize import minimize_scalar, curve_fit, differential_evolution
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures', 'experimental')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Experimental data files
EXPERIMENT_DIR = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Experiment'

DATA_FILES = {
    '60': '60.txt',
    '70': '70.txt',
    '80': '80P.txt',
    '90': '90p.txt',
    '100': '100.txt',
    '110': '110Pnedulum-120-14-1.6-2.txt',
    '120_1': '120Pnedulum-120-14-1.6.txt',
    '120_2': '120Pnedulum-120-14-1.6-2.txt',
}


def load_experimental_data(filename):
    """Load experimental data from file."""
    filepath = os.path.join(EXPERIMENT_DIR, filename)

    times = []
    angles = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip header lines
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                t = float(parts[1].strip())
                angle = float(parts[2].strip())
                times.append(t)
                angles.append(angle)
            except ValueError:
                continue

    times = np.array(times)
    angles = np.array(angles)

    # Normalize time to start from 0
    times = times - times[0]

    # Convert angles to radians
    angles_rad = np.radians(angles)

    return times, angles, angles_rad


def resample_uniform(t, y, dt=0.001):
    """Resample data to uniform time steps."""
    t_uniform = np.arange(t[0], t[-1], dt)
    y_uniform = np.interp(t_uniform, t, y)
    return t_uniform, y_uniform


def estimate_natural_frequency(t, theta, method='peaks'):
    """
    Estimate natural frequency from oscillation period.

    Returns:
        omega_n: Natural frequency (rad/s)
        period: Average period (s)
        frequency: Frequency (Hz)
    """
    if method == 'peaks':
        # Find peaks in the signal
        peaks, _ = find_peaks(theta, distance=len(t)//100, prominence=0.01)

        if len(peaks) < 2:
            # Try with smoothed signal
            theta_smooth = savgol_filter(theta, min(51, len(theta)//10*2+1), 3)
            peaks, _ = find_peaks(theta_smooth, distance=len(t)//100, prominence=0.01)

        if len(peaks) >= 2:
            # Calculate periods between consecutive peaks
            peak_times = t[peaks]
            periods = np.diff(peak_times)

            # Filter out outliers
            median_period = np.median(periods)
            valid_periods = periods[np.abs(periods - median_period) < 0.5 * median_period]

            if len(valid_periods) > 0:
                avg_period = np.mean(valid_periods)
            else:
                avg_period = median_period
        else:
            # Fallback: use zero crossings
            zero_crossings = np.where(np.diff(np.sign(theta)))[0]
            if len(zero_crossings) >= 2:
                crossing_times = t[zero_crossings]
                half_periods = np.diff(crossing_times)
                avg_period = 2 * np.median(half_periods)
            else:
                avg_period = 1.0  # Default fallback

    elif method == 'fft':
        # FFT-based frequency estimation
        dt = np.mean(np.diff(t))
        n = len(theta)
        fft_vals = np.abs(np.fft.rfft(theta - np.mean(theta)))
        freqs = np.fft.rfftfreq(n, dt)

        # Find dominant frequency (skip DC component)
        peak_idx = np.argmax(fft_vals[1:]) + 1
        dominant_freq = freqs[peak_idx]
        avg_period = 1.0 / dominant_freq if dominant_freq > 0 else 1.0

    frequency = 1.0 / avg_period
    omega_n = 2 * np.pi * frequency

    return omega_n, avg_period, frequency


def estimate_stiffness(omega_n, include_gravity=True):
    """
    Estimate stiffness parameter k_θ from natural frequency.

    For horizontal pendulum: θ̈ + damping + k_θ*θ - cos(θ) = 0

    Linearized around θ=0: θ̈ + damping + (k_θ - 1)*θ = 0
    (since -cos(θ) ≈ -1 + θ²/2, the linear part contributes -1 to effective stiffness)

    So: ω_n² = k_θ - 1 (for small angles)
    Therefore: k_θ = ω_n² + 1

    For larger angles, the effective frequency is amplitude-dependent.
    """
    if include_gravity:
        # Accounting for the -cos(θ) term linearized
        k_theta = omega_n**2 + 1
    else:
        # Simple harmonic oscillator
        k_theta = omega_n**2

    return k_theta


def extract_envelope(t, theta, smooth_window=51):
    """Extract amplitude envelope using Hilbert transform."""
    # Smooth the signal first
    if len(theta) > smooth_window:
        theta_smooth = savgol_filter(theta, smooth_window, 3)
    else:
        theta_smooth = theta

    # Hilbert transform for envelope
    analytic_signal = hilbert(theta_smooth)
    envelope = np.abs(analytic_signal)

    # Additional smoothing of envelope
    if len(envelope) > smooth_window:
        envelope = savgol_filter(envelope, smooth_window, 3)

    return envelope


def analyze_decay_pattern(t, envelope):
    """
    Analyze the decay pattern to determine damping type.

    Viscous: exponential decay A(t) = A0 * exp(-ζ*ω_n*t)
    Coulomb: linear decay A(t) = A0 - c*t
    Quadratic: 1/A decay, i.e., A(t) = A0 / (1 + b*t)

    Returns:
        best_type: 'viscous', 'coulomb', or 'quadratic'
        fit_results: dict with fit parameters and R² values
    """
    # Normalize time
    t_norm = t - t[0]

    # Filter valid envelope points (positive and reasonable)
    valid = (envelope > 0.01) & (envelope < envelope[0] * 1.5)
    t_fit = t_norm[valid]
    env_fit = envelope[valid]

    if len(t_fit) < 10:
        return 'unknown', {}

    results = {}

    # 1. Exponential fit (viscous)
    try:
        log_env = np.log(env_fit)
        # Linear fit to log(A) = log(A0) - decay_rate * t
        coeffs = np.polyfit(t_fit, log_env, 1)
        decay_rate = -coeffs[0]
        A0_exp = np.exp(coeffs[1])

        exp_pred = A0_exp * np.exp(-decay_rate * t_fit)
        ss_res = np.sum((env_fit - exp_pred)**2)
        ss_tot = np.sum((env_fit - np.mean(env_fit))**2)
        r2_exp = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results['viscous'] = {
            'decay_rate': decay_rate,
            'A0': A0_exp,
            'R2': r2_exp
        }
    except:
        results['viscous'] = {'R2': -1}

    # 2. Linear fit (Coulomb)
    try:
        coeffs = np.polyfit(t_fit, env_fit, 1)
        linear_pred = coeffs[0] * t_fit + coeffs[1]

        ss_res = np.sum((env_fit - linear_pred)**2)
        ss_tot = np.sum((env_fit - np.mean(env_fit))**2)
        r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results['coulomb'] = {
            'slope': -coeffs[0],
            'A0': coeffs[1],
            'R2': r2_lin
        }
    except:
        results['coulomb'] = {'R2': -1}

    # 3. Hyperbolic fit (quadratic damping)
    try:
        # A(t) = A0 / (1 + b*t) => 1/A = 1/A0 + (b/A0)*t
        inv_env = 1.0 / env_fit
        coeffs = np.polyfit(t_fit, inv_env, 1)

        b_over_A0 = coeffs[0]
        inv_A0 = coeffs[1]
        A0_hyp = 1.0 / inv_A0 if inv_A0 > 0 else env_fit[0]
        b_param = b_over_A0 * A0_hyp

        hyp_pred = A0_hyp / (1 + b_param * t_fit)

        ss_res = np.sum((env_fit - hyp_pred)**2)
        ss_tot = np.sum((env_fit - np.mean(env_fit))**2)
        r2_hyp = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results['quadratic'] = {
            'b': b_param,
            'A0': A0_hyp,
            'R2': r2_hyp
        }
    except:
        results['quadratic'] = {'R2': -1}

    # Determine best fit
    r2_values = {k: v.get('R2', -1) for k, v in results.items()}
    best_type = max(r2_values, key=r2_values.get)

    return best_type, results


def pendulum_ode(t, y, k_theta, zeta, mu_c, mu_q, epsilon=0.1):
    """ODE for horizontal pendulum with multiple damping types."""
    theta, theta_dot = y

    # Damping force
    F_damp = 2 * zeta * theta_dot  # Viscous
    F_damp += mu_c * np.tanh(theta_dot / epsilon)  # Coulomb
    F_damp += mu_q * theta_dot * np.abs(theta_dot)  # Quadratic

    # Equation of motion
    theta_ddot = -F_damp - k_theta * theta + np.cos(theta)

    return [theta_dot, theta_ddot]


def simulate_pendulum(k_theta, zeta, mu_c, mu_q, theta0, t_span, dt=0.001):
    """Simulate pendulum motion."""
    t_eval = np.arange(t_span[0], t_span[1], dt)

    sol = solve_ivp(
        lambda t, y: pendulum_ode(t, y, k_theta, zeta, mu_c, mu_q),
        t_span, [theta0, 0], t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10
    )

    return sol.t, sol.y[0], sol.y[1]


def estimate_damping_least_squares(t, theta, theta_dot, theta_ddot, k_theta, damping_type='viscous'):
    """
    Estimate damping using least squares on the equation of motion.

    θ̈ + F_damp + k_θ*θ - cos(θ) = 0
    => F_damp = -θ̈ - k_θ*θ + cos(θ)
    """
    # Compute the residual (what damping force should be)
    F_damp_observed = -theta_ddot - k_theta * theta + np.cos(theta)

    if damping_type == 'viscous':
        # F_damp = 2*ζ*θ̇  =>  ζ = F_damp / (2*θ̇)
        # Least squares: minimize ||F_damp - 2*ζ*θ̇||²
        A = 2 * theta_dot.reshape(-1, 1)
        b = F_damp_observed

        # Weighted by |θ̇| to focus on high-velocity points
        weights = np.abs(theta_dot) + 0.01
        W = np.diag(weights)

        # Solve weighted least squares
        zeta = np.linalg.lstsq(W @ A, W @ b, rcond=None)[0][0]
        return max(0, zeta)

    elif damping_type == 'coulomb':
        # F_damp = μ_c * tanh(θ̇/ε)
        epsilon = 0.1
        A = np.tanh(theta_dot / epsilon).reshape(-1, 1)
        b = F_damp_observed

        mu_c = np.linalg.lstsq(A, b, rcond=None)[0][0]
        return max(0, mu_c)

    elif damping_type == 'quadratic':
        # F_damp = μ_q * θ̇ * |θ̇|
        A = (theta_dot * np.abs(theta_dot)).reshape(-1, 1)
        b = F_damp_observed

        mu_q = np.linalg.lstsq(A, b, rcond=None)[0][0]
        return max(0, mu_q)

    elif damping_type == 'combined':
        # Estimate all three simultaneously
        epsilon = 0.1
        A = np.column_stack([
            2 * theta_dot,
            np.tanh(theta_dot / epsilon),
            theta_dot * np.abs(theta_dot)
        ])
        b = F_damp_observed

        params = np.linalg.lstsq(A, b, rcond=None)[0]
        return {
            'zeta': max(0, params[0]),
            'mu_c': max(0, params[1]),
            'mu_q': max(0, params[2])
        }


def estimate_damping_optimization(t, theta_obs, k_theta, theta0, damping_type='viscous'):
    """
    Estimate damping by optimizing simulation match.
    """
    t_span = (t[0], t[-1])

    def objective(param):
        if damping_type == 'viscous':
            zeta, mu_c, mu_q = param, 0, 0
        elif damping_type == 'coulomb':
            zeta, mu_c, mu_q = 0, param, 0
        elif damping_type == 'quadratic':
            zeta, mu_c, mu_q = 0, 0, param
        else:
            return 1e10

        try:
            t_sim, theta_sim, _ = simulate_pendulum(k_theta, zeta, mu_c, mu_q, theta0, t_span)
            theta_interp = np.interp(t, t_sim, theta_sim)

            # Match both signal and envelope
            error = np.mean((theta_interp - theta_obs)**2)
            return error
        except:
            return 1e10

    # Bounded optimization
    result = minimize_scalar(objective, bounds=(0.001, 0.5), method='bounded')

    return result.x, result.fun


def analyze_single_experiment(name, filename, plot=True):
    """Analyze a single experimental dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name} ({filename})")
    print('='*60)

    # Load data
    t_raw, angles_deg, angles_rad = load_experimental_data(filename)
    print(f"  Data points: {len(t_raw)}")
    print(f"  Duration: {t_raw[-1]:.2f} s")
    print(f"  Angle range: [{angles_deg.min():.1f}°, {angles_deg.max():.1f}°]")

    # Resample to uniform time steps
    t, theta = resample_uniform(t_raw, angles_rad, dt=0.002)
    print(f"  Resampled points: {len(t)}")

    # Compute derivatives
    dt = t[1] - t[0]
    theta_dot = np.gradient(theta, dt)
    theta_ddot = np.gradient(theta_dot, dt)

    # Smooth derivatives
    theta_dot = savgol_filter(theta_dot, 51, 3)
    theta_ddot = savgol_filter(theta_ddot, 51, 3)

    # Step 1: Estimate natural frequency and stiffness
    omega_n, period, freq = estimate_natural_frequency(t, theta, method='peaks')
    k_theta = estimate_stiffness(omega_n, include_gravity=True)

    print(f"\n  Natural Frequency Estimation:")
    print(f"    Period: {period:.4f} s")
    print(f"    Frequency: {freq:.4f} Hz")
    print(f"    ω_n: {omega_n:.4f} rad/s")
    print(f"    Estimated k_θ: {k_theta:.4f}")

    # Step 2: Extract envelope and analyze decay
    envelope = extract_envelope(t, theta)
    best_type, decay_results = analyze_decay_pattern(t, envelope)

    print(f"\n  Decay Pattern Analysis:")
    for dtype, result in decay_results.items():
        r2 = result.get('R2', -1)
        print(f"    {dtype}: R² = {r2:.4f}")
    print(f"    Best fit: {best_type}")

    # Step 3: Estimate damping parameters
    theta0 = theta[0]

    # Least squares estimation
    ls_results = {}
    for dtype in ['viscous', 'coulomb', 'quadratic']:
        param = estimate_damping_least_squares(t, theta, theta_dot, theta_ddot, k_theta, dtype)
        ls_results[dtype] = param

    # Combined estimation
    combined = estimate_damping_least_squares(t, theta, theta_dot, theta_ddot, k_theta, 'combined')

    print(f"\n  Damping Estimation (Least Squares):")
    print(f"    Viscous (ζ): {ls_results['viscous']:.6f}")
    print(f"    Coulomb (μ_c): {ls_results['coulomb']:.6f}")
    print(f"    Quadratic (μ_q): {ls_results['quadratic']:.6f}")
    print(f"\n  Combined estimation:")
    print(f"    ζ: {combined['zeta']:.6f}")
    print(f"    μ_c: {combined['mu_c']:.6f}")
    print(f"    μ_q: {combined['mu_q']:.6f}")

    # Optimization-based refinement for the best damping type
    opt_param, opt_error = estimate_damping_optimization(t, theta, k_theta, theta0, best_type)
    print(f"\n  Optimization Refinement ({best_type}):")
    print(f"    Parameter: {opt_param:.6f}")
    print(f"    MSE: {opt_error:.6e}")

    # Store results
    results = {
        'name': name,
        'filename': filename,
        'n_points': len(t_raw),
        'duration': t_raw[-1],
        'angle_range': (angles_deg.min(), angles_deg.max()),
        'initial_angle_deg': np.degrees(theta0),
        'omega_n': omega_n,
        'period': period,
        'frequency': freq,
        'k_theta': k_theta,
        'best_damping_type': best_type,
        'decay_R2': decay_results,
        'ls_viscous': ls_results['viscous'],
        'ls_coulomb': ls_results['coulomb'],
        'ls_quadratic': ls_results['quadratic'],
        'combined': combined,
        'opt_param': opt_param,
        'opt_error': opt_error,
        't': t,
        'theta': theta,
        'envelope': envelope
    }

    # Plotting
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Raw oscillation
        ax = axes[0, 0]
        ax.plot(t, np.degrees(theta), 'b-', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{name}: Oscillation')
        ax.grid(True, alpha=0.3)

        # Plot 2: Envelope and decay fit
        ax = axes[0, 1]
        ax.plot(t, np.degrees(envelope), 'b-', linewidth=1.5, label='Envelope')

        # Overlay best fit
        t_norm = t - t[0]
        if best_type == 'viscous' and 'decay_rate' in decay_results.get('viscous', {}):
            dr = decay_results['viscous']
            fit = dr['A0'] * np.exp(-dr['decay_rate'] * t_norm)
            ax.plot(t, np.degrees(fit), 'r--', linewidth=2, label=f'Exponential (R²={dr["R2"]:.3f})')
        elif best_type == 'coulomb' and 'slope' in decay_results.get('coulomb', {}):
            dr = decay_results['coulomb']
            fit = dr['A0'] - dr['slope'] * t_norm
            ax.plot(t, np.degrees(fit), 'g--', linewidth=2, label=f'Linear (R²={dr["R2"]:.3f})')
        elif best_type == 'quadratic' and 'b' in decay_results.get('quadratic', {}):
            dr = decay_results['quadratic']
            fit = dr['A0'] / (1 + dr['b'] * t_norm)
            ax.plot(t, np.degrees(fit), 'm--', linewidth=2, label=f'Hyperbolic (R²={dr["R2"]:.3f})')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (degrees)')
        ax.set_title(f'{name}: Envelope Decay Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Phase portrait
        ax = axes[1, 0]
        ax.plot(np.degrees(theta), np.degrees(theta_dot), 'b-', linewidth=0.3, alpha=0.7)
        ax.scatter([np.degrees(theta[0])], [np.degrees(theta_dot[0])], c='g', s=100, zorder=5, label='Start')
        ax.scatter([np.degrees(theta[-1])], [np.degrees(theta_dot[-1])], c='r', s=100, zorder=5, label='End')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Angular velocity (deg/s)')
        ax.set_title(f'{name}: Phase Portrait')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Comparison with simulation
        ax = axes[1, 1]
        ax.plot(t, np.degrees(theta), 'b-', linewidth=1, alpha=0.7, label='Experimental')

        # Simulate with estimated parameters
        try:
            t_sim, theta_sim, _ = simulate_pendulum(
                k_theta,
                combined['zeta'],
                combined['mu_c'],
                combined['mu_q'],
                theta0, (t[0], t[-1])
            )
            theta_sim_interp = np.interp(t, t_sim, theta_sim)
            ax.plot(t, np.degrees(theta_sim_interp), 'r--', linewidth=1.5, label='Simulated (combined)')
        except Exception as e:
            print(f"    Simulation error: {e}")

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{name}: Experimental vs Simulated')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Experimental Analysis: {name}\n' +
                     f'k_θ={k_theta:.2f}, ζ={combined["zeta"]:.4f}, ' +
                     f'μ_c={combined["mu_c"]:.4f}, μ_q={combined["mu_q"]:.4f}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'exp_analysis_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: exp_analysis_{name}.png")

    return results


def main():
    """Main analysis function."""
    print("=" * 70)
    print("EXPERIMENTAL DATA ANALYSIS")
    print("Estimating Stiffness and Damping Parameters")
    print("=" * 70)

    all_results = []

    for name, filename in DATA_FILES.items():
        try:
            result = analyze_single_experiment(name, filename, plot=True)
            all_results.append(result)
        except Exception as e:
            print(f"\nError analyzing {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)

    print(f"\n{'Name':<10} {'θ₀(°)':<8} {'k_θ':<8} {'ζ':<10} {'μ_c':<10} {'μ_q':<10} {'Best Type':<12}")
    print("-" * 80)

    k_values = []
    zeta_values = []
    mu_c_values = []
    mu_q_values = []

    for r in all_results:
        name = r['name']
        theta0 = r['initial_angle_deg']
        k_theta = r['k_theta']
        zeta = r['combined']['zeta']
        mu_c = r['combined']['mu_c']
        mu_q = r['combined']['mu_q']
        best_type = r['best_damping_type']

        print(f"{name:<10} {theta0:<8.1f} {k_theta:<8.2f} {zeta:<10.6f} {mu_c:<10.6f} {mu_q:<10.6f} {best_type:<12}")

        k_values.append(k_theta)
        zeta_values.append(zeta)
        mu_c_values.append(mu_c)
        mu_q_values.append(mu_q)

    print("-" * 80)
    print(f"{'Average':<10} {'':<8} {np.mean(k_values):<8.2f} {np.mean(zeta_values):<10.6f} " +
          f"{np.mean(mu_c_values):<10.6f} {np.mean(mu_q_values):<10.6f}")
    print(f"{'Std Dev':<10} {'':<8} {np.std(k_values):<8.2f} {np.std(zeta_values):<10.6f} " +
          f"{np.std(mu_c_values):<10.6f} {np.std(mu_q_values):<10.6f}")

    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r['name'] for r in all_results]
    theta0s = [r['initial_angle_deg'] for r in all_results]

    # Plot k_θ vs initial angle
    ax = axes[0, 0]
    ax.scatter(theta0s, k_values, s=100, c='blue', edgecolors='black')
    ax.axhline(np.mean(k_values), color='red', linestyle='--', label=f'Mean = {np.mean(k_values):.2f}')
    ax.fill_between([min(theta0s)-5, max(theta0s)+5],
                    np.mean(k_values) - np.std(k_values),
                    np.mean(k_values) + np.std(k_values),
                    alpha=0.2, color='red')
    ax.set_xlabel('Initial Angle (degrees)')
    ax.set_ylabel('Stiffness k_θ')
    ax.set_title('Stiffness vs Initial Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot damping parameters
    ax = axes[0, 1]
    x = np.arange(len(names))
    width = 0.25
    ax.bar(x - width, zeta_values, width, label='ζ (viscous)', color='blue', alpha=0.7)
    ax.bar(x, mu_c_values, width, label='μ_c (Coulomb)', color='green', alpha=0.7)
    ax.bar(x + width, mu_q_values, width, label='μ_q (quadratic)', color='red', alpha=0.7)
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Damping Parameters by Experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # All oscillations overlay
    ax = axes[1, 0]
    for r in all_results:
        ax.plot(r['t'], np.degrees(r['theta']), linewidth=0.5, alpha=0.7, label=r['name'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('All Experimental Oscillations')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # All envelopes
    ax = axes[1, 1]
    for r in all_results:
        ax.plot(r['t'], np.degrees(r['envelope']), linewidth=1.5, alpha=0.7, label=r['name'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (degrees)')
    ax.set_title('All Amplitude Envelopes')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experimental Analysis Summary\n' +
                 f'Average: k_θ={np.mean(k_values):.2f}±{np.std(k_values):.2f}, ' +
                 f'ζ={np.mean(zeta_values):.4f}, μ_c={np.mean(mu_c_values):.4f}, μ_q={np.mean(mu_q_values):.4f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'exp_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary plot: exp_summary.png")

    print("\n" + "=" * 70)
    print("FINAL ESTIMATED PARAMETERS")
    print("=" * 70)
    print(f"\n  Stiffness:  k_θ = {np.mean(k_values):.4f} ± {np.std(k_values):.4f}")
    print(f"  Viscous:    ζ   = {np.mean(zeta_values):.6f} ± {np.std(zeta_values):.6f}")
    print(f"  Coulomb:    μ_c = {np.mean(mu_c_values):.6f} ± {np.std(mu_c_values):.6f}")
    print(f"  Quadratic:  μ_q = {np.mean(mu_q_values):.6f} ± {np.std(mu_q_values):.6f}")
    print("\n" + "=" * 70)
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
