#!/usr/bin/env python3
"""
Refined Experimental Data Analysis for Horizontal Pendulum
Handles nonlinear effects and equilibrium offsets properly.

Key improvements:
1. Detects equilibrium offset and removes it
2. Uses small-amplitude regime for stiffness estimation
3. Accounts for amplitude-dependent frequency
4. More robust peak detection
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, hilbert
from scipy.optimize import minimize_scalar, minimize, curve_fit
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures', 'experimental')
os.makedirs(FIGURES_DIR, exist_ok=True)

EXPERIMENT_DIR = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Experiment'

DATA_FILES = {
    '60': '60.txt',
    '70': '70.txt',
    '80': '80P.txt',
    '90': '90p.txt',
    '100': '100.txt',
    '110': '110Pnedulum-120-14-1.6-2.txt',
    '120_2': '120Pnedulum-120-14-1.6-2.txt',
}


def load_experimental_data(filename):
    """Load experimental data from file."""
    filepath = os.path.join(EXPERIMENT_DIR, filename)
    times = []
    angles = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

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
    times = times - times[0]
    angles_rad = np.radians(angles)

    return times, angles, angles_rad


def resample_uniform(t, y, dt=0.002):
    """Resample data to uniform time steps."""
    t_uniform = np.arange(t[0], t[-1], dt)
    y_uniform = np.interp(t_uniform, t, y)
    return t_uniform, y_uniform


def detect_equilibrium_offset(theta, window_frac=0.2):
    """
    Detect equilibrium offset from the signal.
    Uses the last portion of the signal where amplitude is smallest.
    """
    n = len(theta)
    last_portion = theta[int(n * (1 - window_frac)):]
    offset = np.mean(last_portion)
    return offset


def extract_peaks_and_valleys(t, theta, min_distance=None):
    """
    Extract peaks and valleys from the signal.
    Returns times and values of peaks (maxima) and valleys (minima).
    """
    if min_distance is None:
        # Estimate minimum distance from data
        zero_crossings = np.where(np.diff(np.sign(theta - np.mean(theta))))[0]
        if len(zero_crossings) >= 4:
            min_distance = int(np.median(np.diff(zero_crossings)) * 0.8)
        else:
            min_distance = len(t) // 50

    min_distance = max(10, min_distance)

    # Find peaks (maxima)
    peaks, peak_props = find_peaks(theta, distance=min_distance, prominence=0.01)

    # Find valleys (minima) by finding peaks in negated signal
    valleys, valley_props = find_peaks(-theta, distance=min_distance, prominence=0.01)

    return {
        'peak_times': t[peaks],
        'peak_values': theta[peaks],
        'peak_indices': peaks,
        'valley_times': t[valleys],
        'valley_values': theta[valleys],
        'valley_indices': valleys
    }


def estimate_frequency_from_peaks(peak_times, valley_times):
    """
    Estimate frequency from peak-to-peak and valley-to-valley intervals.
    """
    periods = []

    # Peak-to-peak periods
    if len(peak_times) >= 2:
        peak_periods = np.diff(peak_times)
        periods.extend(peak_periods)

    # Valley-to-valley periods
    if len(valley_times) >= 2:
        valley_periods = np.diff(valley_times)
        periods.extend(valley_periods)

    if len(periods) == 0:
        return None, None, None

    periods = np.array(periods)

    # Remove outliers (more than 2 std from median)
    median_period = np.median(periods)
    std_period = np.std(periods)
    valid_mask = np.abs(periods - median_period) < 2 * std_period
    valid_periods = periods[valid_mask] if np.sum(valid_mask) > 2 else periods

    avg_period = np.mean(valid_periods)
    frequency = 1.0 / avg_period
    omega = 2 * np.pi * frequency

    return omega, avg_period, frequency


def estimate_frequency_amplitude_relation(peaks_data, valleys_data):
    """
    Analyze how frequency varies with amplitude (nonlinear effect).
    For large angles, period increases due to -cos(θ) term.
    """
    results = []

    peak_times = peaks_data['peak_times']
    peak_values = peaks_data['peak_values']
    valley_times = peaks_data['valley_times']
    valley_values = peaks_data['valley_values']

    # Calculate amplitude and period for each half-cycle
    # Interleave peaks and valleys
    all_times = np.concatenate([peak_times, valley_times])
    all_values = np.concatenate([peak_values, valley_values])
    all_types = np.concatenate([np.ones(len(peak_times)), -np.ones(len(valley_times))])

    sort_idx = np.argsort(all_times)
    all_times = all_times[sort_idx]
    all_values = all_values[sort_idx]
    all_types = all_types[sort_idx]

    for i in range(len(all_times) - 2):
        # Full cycle: peak-valley-peak or valley-peak-valley
        if all_types[i] == all_types[i+2]:
            period = all_times[i+2] - all_times[i]
            amplitude = np.abs(all_values[i] - all_values[i+1])
            results.append({
                'time': all_times[i],
                'period': period,
                'amplitude': amplitude,
                'frequency': 1.0 / period,
                'omega': 2 * np.pi / period
            })

    return results


def fit_amplitude_frequency_model(freq_amp_data):
    """
    Fit ω(A) = ω₀ * (1 - c*A²) model for horizontal pendulum nonlinearity.

    For horizontal pendulum: ω²(A) ≈ ω₀² * (1 - A²/4) for small-moderate A
    This means: ω(A) ≈ ω₀ * (1 - A²/8)
    """
    if len(freq_amp_data) < 5:
        return None

    amplitudes = np.array([d['amplitude'] for d in freq_amp_data])
    omegas = np.array([d['omega'] for d in freq_amp_data])

    # Filter valid data
    valid = (amplitudes > 0.01) & (amplitudes < 2.0) & (omegas > 0)
    if np.sum(valid) < 5:
        return None

    A = amplitudes[valid]
    w = omegas[valid]

    # Fit: ω = ω₀ * (1 - c * A²)
    # Linear fit: ω = ω₀ - ω₀*c*A²
    # y = a + b*x where y=ω, x=A², a=ω₀, b=-ω₀*c

    A_sq = A**2
    try:
        coeffs = np.polyfit(A_sq, w, 1)
        omega_0 = coeffs[1]  # Intercept
        c = -coeffs[0] / omega_0 if omega_0 != 0 else 0

        # Also try with weights (higher weight for smaller amplitudes)
        weights = 1.0 / (A + 0.1)
        coeffs_w = np.polyfit(A_sq, w, 1, w=weights)
        omega_0_w = coeffs_w[1]
        c_w = -coeffs_w[0] / omega_0_w if omega_0_w != 0 else 0

        # Predicted at A=0 (small amplitude limit)
        omega_small = omega_0

        return {
            'omega_0': omega_0,
            'c': c,
            'omega_0_weighted': omega_0_w,
            'c_weighted': c_w,
            'omega_small_amp': omega_small,
            'amplitudes': A,
            'omegas': w
        }
    except:
        return None


def estimate_stiffness_from_small_amplitude(omega_0):
    """
    Estimate k_θ from the small-amplitude natural frequency.

    For horizontal pendulum: θ̈ + damping + k_θ*θ - cos(θ) = 0
    Linearized: θ̈ + damping + (k_θ - 1)*θ ≈ 0
    So: ω₀² = k_θ - 1  =>  k_θ = ω₀² + 1
    """
    k_theta = omega_0**2 + 1
    return k_theta


def extract_envelope_improved(t, theta, peaks_data):
    """
    Extract envelope from peak and valley data directly.
    More robust than Hilbert transform for noisy data.
    """
    peak_times = peaks_data['peak_times']
    peak_values = np.abs(peaks_data['peak_values'])
    valley_times = peaks_data['valley_times']
    valley_values = np.abs(peaks_data['valley_values'])

    # Combine and sort by time
    all_times = np.concatenate([peak_times, valley_times])
    all_amplitudes = np.concatenate([peak_values, valley_values])
    sort_idx = np.argsort(all_times)
    all_times = all_times[sort_idx]
    all_amplitudes = all_amplitudes[sort_idx]

    # Interpolate to full time array
    envelope = np.interp(t, all_times, all_amplitudes)

    return envelope


def analyze_decay_improved(t, envelope, peaks_data):
    """
    Improved decay analysis using peak amplitudes directly.
    """
    # Use peak amplitudes for decay fitting
    peak_times = peaks_data['peak_times']
    peak_amps = np.abs(peaks_data['peak_values'])

    if len(peak_amps) < 5:
        return 'unknown', {}

    t_fit = peak_times - peak_times[0]
    A_fit = peak_amps

    results = {}

    # 1. Exponential fit (viscous): A(t) = A₀ * exp(-λt)
    try:
        log_A = np.log(A_fit[A_fit > 0.01])
        t_log = t_fit[A_fit > 0.01]

        if len(t_log) >= 3:
            coeffs = np.polyfit(t_log, log_A, 1)
            decay_rate = -coeffs[0]
            A0 = np.exp(coeffs[1])

            pred = A0 * np.exp(-decay_rate * t_fit)
            ss_res = np.sum((A_fit - pred)**2)
            ss_tot = np.sum((A_fit - np.mean(A_fit))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            results['viscous'] = {
                'decay_rate': decay_rate,
                'A0': A0,
                'R2': r2
            }
    except:
        results['viscous'] = {'R2': -1}

    # 2. Linear fit (Coulomb): A(t) = A₀ - kt
    try:
        coeffs = np.polyfit(t_fit, A_fit, 1)
        pred = coeffs[0] * t_fit + coeffs[1]
        ss_res = np.sum((A_fit - pred)**2)
        ss_tot = np.sum((A_fit - np.mean(A_fit))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results['coulomb'] = {
            'decay_rate': -coeffs[0],
            'A0': coeffs[1],
            'R2': r2
        }
    except:
        results['coulomb'] = {'R2': -1}

    # 3. Hyperbolic fit (quadratic): A(t) = A₀/(1 + bt)
    try:
        inv_A = 1.0 / A_fit[A_fit > 0.01]
        t_inv = t_fit[A_fit > 0.01]

        if len(t_inv) >= 3:
            coeffs = np.polyfit(t_inv, inv_A, 1)
            A0 = 1.0 / coeffs[1] if coeffs[1] > 0 else A_fit[0]
            b = coeffs[0] * A0

            pred = A0 / (1 + b * t_fit)
            ss_res = np.sum((A_fit - pred)**2)
            ss_tot = np.sum((A_fit - np.mean(A_fit))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            results['quadratic'] = {
                'b': b,
                'A0': A0,
                'R2': r2
            }
    except:
        results['quadratic'] = {'R2': -1}

    # Determine best fit
    r2_values = {k: v.get('R2', -1) for k, v in results.items()}
    best_type = max(r2_values, key=r2_values.get)

    return best_type, results


def pendulum_ode(t, y, k_theta, zeta, mu_c, mu_q, epsilon=0.1):
    """ODE for horizontal pendulum."""
    theta, theta_dot = y
    F_damp = 2 * zeta * theta_dot
    F_damp += mu_c * np.tanh(theta_dot / epsilon)
    F_damp += mu_q * theta_dot * np.abs(theta_dot)
    theta_ddot = -F_damp - k_theta * theta + np.cos(theta)
    return [theta_dot, theta_ddot]


def simulate_pendulum(k_theta, zeta, mu_c, mu_q, theta0, theta_dot0, t_span, dt=0.002):
    """Simulate pendulum motion."""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        lambda t, y: pendulum_ode(t, y, k_theta, zeta, mu_c, mu_q),
        t_span, [theta0, theta_dot0], t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10
    )
    return sol.t, sol.y[0], sol.y[1]


def estimate_damping_from_decay(decay_results, omega_0, best_type):
    """
    Convert decay rate to damping parameter.

    For viscous: A(t) = A₀*exp(-ζ*ω₀*t), so decay_rate = ζ*ω₀
    For Coulomb: dA/dt = -μ_c/(4A) per cycle approx
    For quadratic: more complex relationship
    """
    if best_type == 'viscous' and 'decay_rate' in decay_results.get('viscous', {}):
        decay_rate = decay_results['viscous']['decay_rate']
        zeta = decay_rate / omega_0 if omega_0 > 0 else 0
        return {'zeta': max(0, zeta), 'mu_c': 0, 'mu_q': 0}

    elif best_type == 'coulomb' and 'decay_rate' in decay_results.get('coulomb', {}):
        # For Coulomb: amplitude decreases linearly
        # dA/dn = -4*μ_c/π per cycle (for underdamped)
        decay_rate = decay_results['coulomb']['decay_rate']
        period = 2 * np.pi / omega_0 if omega_0 > 0 else 1
        mu_c = decay_rate * period * np.pi / 4
        return {'zeta': 0, 'mu_c': max(0, mu_c), 'mu_q': 0}

    elif best_type == 'quadratic' and 'b' in decay_results.get('quadratic', {}):
        b = decay_results['quadratic']['b']
        A0 = decay_results['quadratic']['A0']
        # For quadratic damping: 1/A - 1/A₀ = (2μ_q*ω₀/3π)*t approximately
        mu_q = b * 3 * np.pi / (2 * omega_0 * A0) if omega_0 > 0 and A0 > 0 else 0
        return {'zeta': 0, 'mu_c': 0, 'mu_q': max(0, mu_q)}

    return {'zeta': 0, 'mu_c': 0, 'mu_q': 0}


def estimate_damping_optimization(t, theta_obs, k_theta, theta0, theta_dot0=0):
    """
    Estimate damping by optimizing simulation match.
    """
    t_span = (t[0], t[-1])

    def objective(params):
        zeta, mu_c, mu_q = params

        # Ensure non-negative
        if zeta < 0 or mu_c < 0 or mu_q < 0:
            return 1e10

        try:
            t_sim, theta_sim, _ = simulate_pendulum(k_theta, zeta, mu_c, mu_q, theta0, theta_dot0, t_span)
            theta_interp = np.interp(t, t_sim, theta_sim)
            error = np.mean((theta_interp - theta_obs)**2)
            return error
        except:
            return 1e10

    # Start with small values
    x0 = [0.01, 0.01, 0.001]
    bounds = [(0, 0.5), (0, 1.0), (0, 0.1)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    return {
        'zeta': result.x[0],
        'mu_c': result.x[1],
        'mu_q': result.x[2],
        'mse': result.fun
    }


def analyze_single_experiment_refined(name, filename, plot=True):
    """Refined analysis of a single experimental dataset."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {name} ({filename})")
    print('='*70)

    # Load data
    t_raw, angles_deg, angles_rad = load_experimental_data(filename)
    print(f"  Raw data points: {len(t_raw)}")
    print(f"  Duration: {t_raw[-1]:.2f} s")
    print(f"  Angle range: [{angles_deg.min():.1f}°, {angles_deg.max():.1f}°]")

    # Resample to uniform time steps
    t, theta = resample_uniform(t_raw, angles_rad, dt=0.002)
    dt = t[1] - t[0]

    # Detect equilibrium offset
    offset = detect_equilibrium_offset(theta)
    theta_centered = theta - offset
    print(f"  Equilibrium offset: {np.degrees(offset):.2f}°")

    # Extract peaks and valleys
    peaks_data = extract_peaks_and_valleys(t, theta_centered)
    n_peaks = len(peaks_data['peak_times'])
    n_valleys = len(peaks_data['valley_times'])
    print(f"  Detected {n_peaks} peaks and {n_valleys} valleys")

    # Estimate frequency-amplitude relationship
    freq_amp_data = estimate_frequency_amplitude_relation(peaks_data, peaks_data)
    print(f"  Frequency-amplitude data points: {len(freq_amp_data)}")

    # Fit amplitude-frequency model
    fit_result = fit_amplitude_frequency_model(freq_amp_data)

    if fit_result is not None:
        omega_0 = fit_result['omega_0_weighted']
        print(f"\n  Frequency-Amplitude Analysis:")
        print(f"    ω₀ (small amplitude): {omega_0:.4f} rad/s")
        print(f"    Nonlinearity coeff c: {fit_result['c_weighted']:.4f}")
        print(f"    Period (small amp): {2*np.pi/omega_0:.4f} s")
        print(f"    Frequency: {omega_0/(2*np.pi):.4f} Hz")
    else:
        # Fallback: use simple peak-based estimation
        omega_0, period, freq = estimate_frequency_from_peaks(
            peaks_data['peak_times'], peaks_data['valley_times'])
        if omega_0 is None:
            omega_0 = 10.0  # Default fallback
        print(f"\n  Simple Frequency Estimation:")
        print(f"    ω₀: {omega_0:.4f} rad/s")

    # Estimate stiffness
    k_theta = estimate_stiffness_from_small_amplitude(omega_0)
    print(f"\n  Stiffness Estimation:")
    print(f"    k_θ = ω₀² + 1 = {k_theta:.4f}")

    # Extract envelope and analyze decay
    envelope = extract_envelope_improved(t, theta_centered, peaks_data)
    best_type, decay_results = analyze_decay_improved(t, envelope, peaks_data)

    print(f"\n  Decay Analysis:")
    for dtype, result in decay_results.items():
        r2 = result.get('R2', -1)
        print(f"    {dtype}: R² = {r2:.4f}")
    print(f"    Best fit: {best_type}")

    # Estimate damping from decay
    damping_from_decay = estimate_damping_from_decay(decay_results, omega_0, best_type)
    print(f"\n  Damping from Decay Analysis:")
    print(f"    ζ = {damping_from_decay['zeta']:.6f}")
    print(f"    μ_c = {damping_from_decay['mu_c']:.6f}")
    print(f"    μ_q = {damping_from_decay['mu_q']:.6f}")

    # Optimization-based refinement
    theta0 = theta_centered[0]
    # Estimate initial velocity from first few points
    theta_dot0 = (theta_centered[10] - theta_centered[0]) / (t[10] - t[0]) if len(t) > 10 else 0

    print(f"\n  Running optimization...")
    opt_result = estimate_damping_optimization(t, theta_centered, k_theta, theta0, theta_dot0)
    print(f"  Optimization Results:")
    print(f"    ζ = {opt_result['zeta']:.6f}")
    print(f"    μ_c = {opt_result['mu_c']:.6f}")
    print(f"    μ_q = {opt_result['mu_q']:.6f}")
    print(f"    MSE = {opt_result['mse']:.6e}")

    # Store results
    results = {
        'name': name,
        'filename': filename,
        'duration': t_raw[-1],
        'angle_range_deg': (angles_deg.min(), angles_deg.max()),
        'equilibrium_offset_deg': np.degrees(offset),
        'n_peaks': n_peaks,
        'omega_0': omega_0,
        'period': 2*np.pi/omega_0,
        'frequency': omega_0/(2*np.pi),
        'k_theta': k_theta,
        'best_damping_type': best_type,
        'decay_results': decay_results,
        'damping_decay': damping_from_decay,
        'damping_opt': opt_result,
        't': t,
        'theta': theta_centered,
        'envelope': envelope,
        'peaks_data': peaks_data,
        'freq_amp_data': freq_amp_data,
        'fit_result': fit_result
    }

    # Plotting
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Plot 1: Raw oscillation
        ax = axes[0, 0]
        ax.plot(t, np.degrees(theta_centered), 'b-', linewidth=0.5, alpha=0.7)
        ax.scatter(peaks_data['peak_times'], np.degrees(peaks_data['peak_values']),
                   c='red', s=20, zorder=5, label='Peaks')
        ax.scatter(peaks_data['valley_times'], np.degrees(peaks_data['valley_values']),
                   c='green', s=20, zorder=5, label='Valleys')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{name}: Centered Oscillation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Envelope decay
        ax = axes[0, 1]
        ax.plot(peaks_data['peak_times'], np.degrees(np.abs(peaks_data['peak_values'])),
                'ro-', markersize=4, label='Peak amplitudes')

        # Overlay best fit
        t_norm = peaks_data['peak_times'] - peaks_data['peak_times'][0]
        if best_type == 'viscous' and 'decay_rate' in decay_results.get('viscous', {}):
            dr = decay_results['viscous']
            fit = dr['A0'] * np.exp(-dr['decay_rate'] * t_norm)
            ax.plot(peaks_data['peak_times'], np.degrees(fit), 'b--', linewidth=2,
                    label=f'Exp fit (R²={dr["R2"]:.3f})')
        elif best_type == 'coulomb' and 'decay_rate' in decay_results.get('coulomb', {}):
            dr = decay_results['coulomb']
            fit = dr['A0'] - dr['decay_rate'] * t_norm
            ax.plot(peaks_data['peak_times'], np.degrees(fit), 'g--', linewidth=2,
                    label=f'Linear fit (R²={dr["R2"]:.3f})')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (degrees)')
        ax.set_title(f'{name}: Amplitude Decay')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Frequency vs Amplitude
        ax = axes[0, 2]
        if fit_result is not None:
            ax.scatter(np.degrees(fit_result['amplitudes']), fit_result['omegas'],
                       c='blue', s=30, alpha=0.6)

            # Plot fit
            A_plot = np.linspace(0, max(fit_result['amplitudes']), 100)
            w_plot = fit_result['omega_0_weighted'] * (1 - fit_result['c_weighted'] * A_plot**2)
            ax.plot(np.degrees(A_plot), w_plot, 'r-', linewidth=2,
                    label=f'ω = {fit_result["omega_0_weighted"]:.2f}(1 - {fit_result["c_weighted"]:.3f}A²)')

            ax.axhline(fit_result['omega_0_weighted'], color='green', linestyle='--',
                       alpha=0.5, label=f'ω₀ = {fit_result["omega_0_weighted"]:.2f}')

        ax.set_xlabel('Amplitude (degrees)')
        ax.set_ylabel('Angular frequency ω (rad/s)')
        ax.set_title(f'{name}: Frequency-Amplitude Relation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 4: Phase portrait
        ax = axes[1, 0]
        theta_dot = np.gradient(theta_centered, dt)
        theta_dot = savgol_filter(theta_dot, 51, 3)
        ax.plot(np.degrees(theta_centered), np.degrees(theta_dot), 'b-', linewidth=0.3, alpha=0.7)
        ax.scatter([np.degrees(theta_centered[0])], [np.degrees(theta_dot[0])],
                   c='green', s=100, zorder=5, label='Start')
        ax.scatter([np.degrees(theta_centered[-1])], [np.degrees(theta_dot[-1])],
                   c='red', s=100, zorder=5, label='End')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Angular velocity (deg/s)')
        ax.set_title(f'{name}: Phase Portrait')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: Simulation comparison
        ax = axes[1, 1]
        ax.plot(t, np.degrees(theta_centered), 'b-', linewidth=1, alpha=0.7, label='Experimental')

        try:
            t_sim, theta_sim, _ = simulate_pendulum(
                k_theta,
                opt_result['zeta'],
                opt_result['mu_c'],
                opt_result['mu_q'],
                theta0, theta_dot0, (t[0], t[-1])
            )
            theta_sim_interp = np.interp(t, t_sim, theta_sim)
            ax.plot(t, np.degrees(theta_sim_interp), 'r--', linewidth=1.5, label='Simulated (opt)')
        except Exception as e:
            print(f"    Simulation error: {e}")

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{name}: Experimental vs Simulated')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Period evolution
        ax = axes[1, 2]
        if len(freq_amp_data) > 0:
            times = [d['time'] for d in freq_amp_data]
            periods = [d['period'] for d in freq_amp_data]
            ax.plot(times, periods, 'bo-', markersize=4)
            ax.axhline(2*np.pi/omega_0, color='red', linestyle='--',
                       label=f'T₀ = {2*np.pi/omega_0:.3f} s')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Period (s)')
        ax.set_title(f'{name}: Period Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Refined Analysis: {name}\n' +
                     f'k_θ={k_theta:.2f}, ω₀={omega_0:.2f} rad/s, ' +
                     f'ζ={opt_result["zeta"]:.4f}, μ_c={opt_result["mu_c"]:.4f}, μ_q={opt_result["mu_q"]:.4f}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'refined_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: refined_{name}.png")

    return results


def main():
    """Main analysis function."""
    print("=" * 70)
    print("REFINED EXPERIMENTAL DATA ANALYSIS")
    print("Estimating Stiffness and Damping Parameters")
    print("=" * 70)

    all_results = []

    for name, filename in DATA_FILES.items():
        try:
            result = analyze_single_experiment_refined(name, filename, plot=True)
            all_results.append(result)
        except Exception as e:
            print(f"\nError analyzing {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF REFINED RESULTS")
    print("=" * 70)

    print(f"\n{'Name':<10} {'ω₀(rad/s)':<10} {'k_θ':<8} {'ζ':<10} {'μ_c':<10} {'μ_q':<10} {'Best':<10}")
    print("-" * 80)

    k_values = []
    omega_values = []
    zeta_values = []
    mu_c_values = []
    mu_q_values = []

    for r in all_results:
        name = r['name']
        omega = r['omega_0']
        k_theta = r['k_theta']
        zeta = r['damping_opt']['zeta']
        mu_c = r['damping_opt']['mu_c']
        mu_q = r['damping_opt']['mu_q']
        best = r['best_damping_type']

        print(f"{name:<10} {omega:<10.4f} {k_theta:<8.2f} {zeta:<10.6f} {mu_c:<10.6f} {mu_q:<10.6f} {best:<10}")

        k_values.append(k_theta)
        omega_values.append(omega)
        zeta_values.append(zeta)
        mu_c_values.append(mu_c)
        mu_q_values.append(mu_q)

    print("-" * 80)
    print(f"{'Mean':<10} {np.mean(omega_values):<10.4f} {np.mean(k_values):<8.2f} " +
          f"{np.mean(zeta_values):<10.6f} {np.mean(mu_c_values):<10.6f} {np.mean(mu_q_values):<10.6f}")
    print(f"{'Std':<10} {np.std(omega_values):<10.4f} {np.std(k_values):<8.2f} " +
          f"{np.std(zeta_values):<10.6f} {np.std(mu_c_values):<10.6f} {np.std(mu_q_values):<10.6f}")

    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r['name'] for r in all_results]
    x = np.arange(len(names))

    # Plot 1: k_θ comparison
    ax = axes[0, 0]
    ax.bar(x, k_values, color='blue', alpha=0.7)
    ax.axhline(np.mean(k_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(k_values):.2f}')
    ax.fill_between([-0.5, len(names)-0.5],
                    np.mean(k_values) - np.std(k_values),
                    np.mean(k_values) + np.std(k_values),
                    alpha=0.2, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Stiffness k_θ')
    ax.set_title('Estimated Stiffness by Experiment')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Damping parameters
    ax = axes[0, 1]
    width = 0.25
    ax.bar(x - width, zeta_values, width, label='ζ (viscous)', color='blue', alpha=0.7)
    ax.bar(x, mu_c_values, width, label='μ_c (Coulomb)', color='green', alpha=0.7)
    ax.bar(x + width, mu_q_values, width, label='μ_q (quadratic)', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Damping Parameters by Experiment')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: All oscillations
    ax = axes[1, 0]
    for r in all_results:
        ax.plot(r['t'], np.degrees(r['theta']), linewidth=0.5, alpha=0.7, label=r['name'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('All Experimental Oscillations (Centered)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: ω vs experiment (showing nonlinearity)
    ax = axes[1, 1]
    ax.bar(x, omega_values, color='purple', alpha=0.7)
    ax.axhline(np.mean(omega_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(omega_values):.2f} rad/s')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Natural Frequency ω₀ (rad/s)')
    ax.set_title('Small-Amplitude Natural Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Refined Experimental Analysis Summary\n' +
                 f'k_θ = {np.mean(k_values):.2f} ± {np.std(k_values):.2f}, ' +
                 f'ω₀ = {np.mean(omega_values):.2f} ± {np.std(omega_values):.2f} rad/s\n' +
                 f'ζ = {np.mean(zeta_values):.4f}, μ_c = {np.mean(mu_c_values):.4f}, μ_q = {np.mean(mu_q_values):.5f}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'refined_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: refined_summary.png")

    print("\n" + "=" * 70)
    print("FINAL ESTIMATED PARAMETERS (REFINED)")
    print("=" * 70)
    print(f"\n  Natural Frequency: ω₀ = {np.mean(omega_values):.4f} ± {np.std(omega_values):.4f} rad/s")
    print(f"  Period:            T  = {2*np.pi/np.mean(omega_values):.4f} s")
    print(f"  Stiffness:         k_θ = {np.mean(k_values):.4f} ± {np.std(k_values):.4f}")
    print(f"\n  Viscous:           ζ   = {np.mean(zeta_values):.6f} ± {np.std(zeta_values):.6f}")
    print(f"  Coulomb:           μ_c = {np.mean(mu_c_values):.6f} ± {np.std(mu_c_values):.6f}")
    print(f"  Quadratic:         μ_q = {np.mean(mu_q_values):.6f} ± {np.std(mu_q_values):.6f}")
    print("\n" + "=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
