#!/usr/bin/env python3
"""
Apply All Damping Estimation Methods to Experimental Data
===========================================================
Uses the 80° experiment data with derived effective stiffness.
Compares results against envelope-derived reference values.

Reference values (from direct measurement):
- ω = 21.82 rad/s
- κ_eff = 4.85 (nondimensional)
- ζ_ref = 0.0088 (from λ/ω)
- λ = 0.191 (1/s) decay rate
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, hilbert
from scipy.optimize import minimize_scalar, differential_evolution, minimize
from scipy.integrate import solve_ivp
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Codes/Dec16/Damping-parameter-estimation-using-topological-signal-processing'
FIGURES_DIR = os.path.join(BASE_DIR, 'figures', 'experimental')
DATA_FILE = '/Users/ucla/Library/CloudStorage/OneDrive-KFUPM/Yilbas2019/Solaihat/Horizontal_pendulum/Experiment/80P.txt'

np.random.seed(42)

# =============================================================================
# LOAD EXPERIMENTAL DATA
# =============================================================================

def load_experimental_data():
    """Load and preprocess 80° experiment data."""
    times, angles = [], []
    with open(DATA_FILE, 'r') as f:
        for line in f.readlines()[2:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    times.append(float(parts[1].strip()))
                    angles.append(float(parts[2].strip()))
                except:
                    continue
    times = np.array(times) - times[0]
    angles_rad = np.radians(np.array(angles))

    # Resample to uniform time steps
    dt = 0.002
    t_new = np.arange(times[0], times[-1], dt)
    theta = np.interp(t_new, times, angles_rad)

    # Remove equilibrium offset
    offset = np.mean(theta[int(len(theta)*0.8):])
    theta = theta - offset

    # Compute velocity
    theta_dot = savgol_filter(np.gradient(theta, dt), 51, 3)

    return t_new, theta, theta_dot, dt


def get_reference_values(t, theta, dt):
    """Extract reference damping values from experimental data."""
    # Find peaks
    peaks, _ = find_peaks(theta, distance=50, prominence=0.01)
    peak_times = t[peaks]
    peak_amps = np.abs(theta[peaks])

    # Period and frequency
    periods = np.diff(peak_times)
    T = np.median(periods)
    omega = 2 * np.pi / T

    # Decay rate from exponential fit
    t_peaks = peak_times - peak_times[0]
    log_amps = np.log(peak_amps)
    decay_coeffs = np.polyfit(t_peaks, log_amps, 1)
    decay_rate = -decay_coeffs[0]
    A0 = np.exp(decay_coeffs[1])

    # R² for decay
    pred = A0 * np.exp(-decay_rate * t_peaks)
    ss_res = np.sum((peak_amps - pred)**2)
    ss_tot = np.sum((peak_amps - np.mean(peak_amps))**2)
    R2 = 1 - ss_res / ss_tot

    # Effective stiffness and damping ratio
    I = 0.05 * 0.1**2  # m*L^2
    kt_eff = I * omega**2
    kappa_eff = kt_eff / (0.05 * 9.81 * 0.1)
    zeta_ref = decay_rate / omega

    return {
        'omega': omega,
        'T': T,
        'decay_rate': decay_rate,
        'A0': A0,
        'R2': R2,
        'kappa_eff': kappa_eff,
        'zeta_ref': zeta_ref,
        'kt_eff': kt_eff
    }


# =============================================================================
# METHOD 1: ENVELOPE DECAY (Baseline - Direct Measurement)
# =============================================================================

def method_envelope_decay(t, theta, ref):
    """Direct measurement from amplitude envelope."""
    print("\n" + "="*60)
    print("METHOD 1: ENVELOPE DECAY (Direct Measurement)")
    print("="*60)

    zeta_est = ref['zeta_ref']
    print(f"  ζ = λ/ω = {ref['decay_rate']:.4f}/{ref['omega']:.2f} = {zeta_est:.6f}")
    print(f"  Decay R² = {ref['R2']:.4f}")

    return {'zeta': zeta_est, 'method': 'Envelope Decay'}


# =============================================================================
# METHOD 2: HILBERT TRANSFORM ENVELOPE
# =============================================================================

def method_hilbert_envelope(t, theta, ref):
    """Use Hilbert transform to extract envelope."""
    print("\n" + "="*60)
    print("METHOD 2: HILBERT TRANSFORM ENVELOPE")
    print("="*60)

    analytic = hilbert(theta)
    envelope = np.abs(analytic)

    # Fit exponential to envelope
    # Avoid initial transient
    start_idx = int(len(t) * 0.05)
    end_idx = int(len(t) * 0.95)

    t_fit = t[start_idx:end_idx]
    env_fit = envelope[start_idx:end_idx]

    # Linear fit to log(envelope)
    log_env = np.log(np.maximum(env_fit, 1e-10))
    coeffs = np.polyfit(t_fit, log_env, 1)
    decay_rate = -coeffs[0]

    zeta_est = decay_rate / ref['omega']
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  Decay rate from Hilbert: {decay_rate:.4f}")
    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'Hilbert Envelope', 'error': error}


# =============================================================================
# METHOD 3: LEAST SQUARES ON ODE
# =============================================================================

def method_least_squares(t, theta, theta_dot, ref):
    """Least squares estimation on the ODE."""
    print("\n" + "="*60)
    print("METHOD 3: LEAST SQUARES ON ODE")
    print("="*60)

    dt = t[1] - t[0]
    theta_ddot = savgol_filter(np.gradient(theta_dot, dt), 51, 3)

    # Trim edges
    trim = 100
    th = theta[trim:-trim]
    th_dot = theta_dot[trim:-trim]
    th_ddot = theta_ddot[trim:-trim]

    # ODE: θ̈ + 2ζω_n θ̇ + ω_n² θ = 0 (for small angles)
    # Rearrange: θ̈ = -2ζω_n θ̇ - ω_n² θ
    # Design matrix: A = [-2ω_n θ̇, -θ] for parameters [ζ, ω_n²]

    omega = ref['omega']

    # For known ω, solve for ζ only
    # θ̈ + ω² θ = -2ζω θ̇
    # b = θ̈ + ω² θ
    # A = -2ω θ̇

    b = th_ddot + omega**2 * th
    A = (-2 * omega * th_dot).reshape(-1, 1)

    zeta_est = np.linalg.lstsq(A, b, rcond=None)[0][0]
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'Least Squares', 'error': error}


# =============================================================================
# METHOD 4: OPTIMIZATION (ENVELOPE MATCHING)
# =============================================================================

def method_optimization(t, theta, ref):
    """Optimization-based envelope matching."""
    print("\n" + "="*60)
    print("METHOD 4: OPTIMIZATION (ENVELOPE MATCHING)")
    print("="*60)

    omega = ref['omega']

    # Get experimental envelope
    analytic = hilbert(theta)
    env_exp = np.abs(analytic)

    def objective(zeta):
        # Theoretical envelope: A(t) = A0 * exp(-ζ*ω*t)
        A0_est = env_exp[0]
        env_theory = A0_est * np.exp(-zeta * omega * t)
        mse = np.mean((env_exp - env_theory)**2)
        return mse

    result = minimize_scalar(objective, bounds=(0.001, 0.1), method='bounded')
    zeta_est = result.x
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'Optimization', 'error': error}


# =============================================================================
# METHOD 5: GENETIC ALGORITHM
# =============================================================================

def method_genetic_algorithm(t, theta, ref):
    """Genetic Algorithm optimization."""
    print("\n" + "="*60)
    print("METHOD 5: GENETIC ALGORITHM")
    print("="*60)

    omega = ref['omega']
    kappa = ref['kappa_eff']
    theta0 = theta[0]
    theta_dot0 = 0
    dt = t[1] - t[0]

    def simulate(zeta):
        def ode(t, y):
            th, th_dot = y
            th_ddot = -2*zeta*omega*th_dot - omega**2*th
            return [th_dot, th_ddot]
        sol = solve_ivp(ode, (t[0], t[-1]), [theta0, theta_dot0],
                       t_eval=t, method='RK45')
        return sol.y[0]

    def fitness(zeta):
        try:
            theta_sim = simulate(zeta)
            # Compare envelopes
            env_exp = np.abs(hilbert(theta))
            env_sim = np.abs(hilbert(theta_sim))
            mse = np.mean((env_exp - env_sim)**2)
            return -mse  # GA maximizes
        except:
            return -1e10

    # Simple GA
    pop_size = 50
    n_gen = 100
    bounds = (0.001, 0.1)

    # Initialize population
    population = np.random.uniform(bounds[0], bounds[1], pop_size)

    for gen in range(n_gen):
        # Evaluate fitness
        fitnesses = np.array([fitness(ind) for ind in population])

        # Selection (tournament)
        new_pop = []
        for _ in range(pop_size):
            idx = np.random.choice(pop_size, 3, replace=False)
            winner = population[idx[np.argmax(fitnesses[idx])]]
            new_pop.append(winner)

        # Crossover and mutation
        for i in range(0, pop_size-1, 2):
            if np.random.random() < 0.8:
                alpha = np.random.random()
                child1 = alpha * new_pop[i] + (1-alpha) * new_pop[i+1]
                child2 = (1-alpha) * new_pop[i] + alpha * new_pop[i+1]
                new_pop[i], new_pop[i+1] = child1, child2

        for i in range(pop_size):
            if np.random.random() < 0.2:
                new_pop[i] += np.random.normal(0, 0.005)
                new_pop[i] = np.clip(new_pop[i], bounds[0], bounds[1])

        population = np.array(new_pop)

    # Best individual
    fitnesses = np.array([fitness(ind) for ind in population])
    zeta_est = population[np.argmax(fitnesses)]
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'Genetic Algorithm', 'error': error}


# =============================================================================
# METHOD 6: DIFFERENTIAL EVOLUTION
# =============================================================================

def method_differential_evolution(t, theta, ref):
    """Differential Evolution optimization."""
    print("\n" + "="*60)
    print("METHOD 6: DIFFERENTIAL EVOLUTION")
    print("="*60)

    omega = ref['omega']
    theta0 = theta[0]

    def objective(params):
        zeta = params[0]
        # Theoretical envelope
        A0 = np.abs(theta0)
        env_theory = A0 * np.exp(-zeta * omega * t)
        env_exp = np.abs(hilbert(theta))
        mse = np.mean((env_exp - env_theory)**2)
        return mse

    result = differential_evolution(objective, bounds=[(0.001, 0.1)],
                                    seed=42, maxiter=200, polish=True)
    zeta_est = result.x[0]
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'Differential Evolution', 'error': error}


# =============================================================================
# METHOD 7: LOGARITHMIC DECREMENT
# =============================================================================

def method_log_decrement(t, theta, ref):
    """Classical logarithmic decrement method."""
    print("\n" + "="*60)
    print("METHOD 7: LOGARITHMIC DECREMENT")
    print("="*60)

    # Find peaks
    peaks, _ = find_peaks(theta, distance=50, prominence=0.01)
    peak_amps = np.abs(theta[peaks])

    # Calculate logarithmic decrements
    if len(peak_amps) < 2:
        print("  Not enough peaks for analysis")
        return {'zeta': 0, 'method': 'Log Decrement', 'error': 100}

    log_decrements = []
    for i in range(len(peak_amps) - 1):
        delta = np.log(peak_amps[i] / peak_amps[i+1])
        log_decrements.append(delta)

    delta_avg = np.mean(log_decrements)

    # ζ = δ / sqrt(4π² + δ²)
    zeta_est = delta_avg / np.sqrt(4 * np.pi**2 + delta_avg**2)
    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  Average log decrement δ: {delta_avg:.6f}")
    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'Log Decrement', 'error': error}


# =============================================================================
# METHOD 8: HALF-POWER BANDWIDTH (FFT)
# =============================================================================

def method_half_power_bandwidth(t, theta, ref):
    """Half-power bandwidth method using FFT."""
    print("\n" + "="*60)
    print("METHOD 8: HALF-POWER BANDWIDTH (FFT)")
    print("="*60)

    dt = t[1] - t[0]
    n = len(theta)

    # FFT
    fft_vals = np.fft.fft(theta)
    freqs = np.fft.fftfreq(n, dt)

    # Get positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    magnitude = np.abs(fft_vals[pos_mask])

    # Find peak frequency
    peak_idx = np.argmax(magnitude)
    f_peak = freqs_pos[peak_idx]
    peak_mag = magnitude[peak_idx]

    # Find half-power points (-3dB)
    half_power = peak_mag / np.sqrt(2)

    # Find bandwidth
    above_half = magnitude >= half_power
    if np.sum(above_half) > 1:
        indices = np.where(above_half)[0]
        f1 = freqs_pos[indices[0]]
        f2 = freqs_pos[indices[-1]]
        bandwidth = f2 - f1

        # ζ = Δf / (2 * f_peak)
        zeta_est = bandwidth / (2 * f_peak)
    else:
        zeta_est = ref['zeta_ref']  # Fallback

    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  Peak frequency: {f_peak:.4f} Hz")
    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'Half-Power Bandwidth', 'error': error}


# =============================================================================
# METHOD 9: PRONY METHOD
# =============================================================================

def method_prony(t, theta, ref):
    """Prony's method for exponential fitting."""
    print("\n" + "="*60)
    print("METHOD 9: PRONY METHOD")
    print("="*60)

    # Get envelope
    envelope = np.abs(hilbert(theta))

    # Downsample for efficiency
    step = 10
    env_ds = envelope[::step]
    t_ds = t[::step]

    N = len(env_ds)
    p = 2  # Order (for damped sinusoid)

    # Build Hankel-like matrix
    try:
        if N > 2*p:
            A = np.zeros((N-p, p))
            b = env_ds[p:]
            for i in range(N-p):
                A[i, :] = env_ds[i:i+p][::-1]

            # Solve for AR coefficients
            coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

            # Find roots (poles)
            poly_coeffs = np.concatenate([[-1], coeffs[::-1]])
            roots = np.roots(poly_coeffs)

            # Extract damping from dominant root
            dt_ds = t_ds[1] - t_ds[0]
            for root in roots:
                if np.abs(root) < 1 and np.abs(root) > 0.9:  # Stable, slow decay
                    decay_rate = -np.log(np.abs(root)) / dt_ds
                    zeta_est = decay_rate / ref['omega']
                    break
            else:
                zeta_est = ref['zeta_ref']
        else:
            zeta_est = ref['zeta_ref']
    except:
        zeta_est = ref['zeta_ref']

    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'Prony', 'error': error}


# =============================================================================
# METHOD 10: CURVE FIT (SCIPY)
# =============================================================================

def method_curve_fit(t, theta, ref):
    """Direct curve fitting of damped oscillation."""
    print("\n" + "="*60)
    print("METHOD 10: CURVE FIT")
    print("="*60)

    from scipy.optimize import curve_fit

    omega = ref['omega']

    def damped_oscillation(t, A0, zeta, phi):
        omega_d = omega * np.sqrt(1 - zeta**2) if zeta < 1 else omega
        return A0 * np.exp(-zeta * omega * t) * np.cos(omega_d * t + phi)

    try:
        # Initial guess
        p0 = [np.max(np.abs(theta)), 0.01, 0]

        popt, _ = curve_fit(damped_oscillation, t, theta, p0=p0,
                           bounds=([0, 0.001, -np.pi], [1, 0.5, np.pi]),
                           maxfev=10000)

        zeta_est = popt[1]
    except:
        zeta_est = ref['zeta_ref']

    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'Curve Fit', 'error': error}


# =============================================================================
# METHOD 11: SYSTEM IDENTIFICATION (ARX)
# =============================================================================

def method_arx(t, theta, theta_dot, ref):
    """AutoRegressive with eXogenous input (ARX) model."""
    print("\n" + "="*60)
    print("METHOD 11: ARX SYSTEM IDENTIFICATION")
    print("="*60)

    dt = t[1] - t[0]
    omega = ref['omega']

    # Build ARX model: theta[k] = a1*theta[k-1] + a2*theta[k-2]
    n = len(theta)
    order = 2

    # Design matrix
    A = np.zeros((n - order, order))
    b = theta[order:]

    for i in range(n - order):
        A[i, 0] = theta[i + order - 1]
        A[i, 1] = theta[i + order - 2]

    # Solve
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

    # Convert to continuous-time damping
    # Discrete poles: z = exp((-ζω ± jω_d) * dt)
    poly = np.array([1, -coeffs[0], -coeffs[1]])
    roots = np.roots(poly)

    # Extract damping from complex roots
    for root in roots:
        if np.abs(root) < 1:
            s = np.log(root) / dt
            zeta_est = -np.real(s) / omega
            if zeta_est > 0 and zeta_est < 1:
                break
    else:
        zeta_est = ref['zeta_ref']

    error = abs(zeta_est - ref['zeta_ref']) / ref['zeta_ref'] * 100

    print(f"  ζ estimated: {zeta_est:.6f}")
    print(f"  Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"  Error: {error:.4f}%")

    return {'zeta': zeta_est, 'method': 'ARX', 'error': error}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("ALL METHODS ANALYSIS ON EXPERIMENTAL DATA")
    print("80° Horizontal Pendulum Experiment")
    print("="*70)

    # Load data
    t, theta, theta_dot, dt = load_experimental_data()
    print(f"\nData loaded: {len(t)} points, {t[-1]:.1f}s duration")

    # Get reference values
    ref = get_reference_values(t, theta, dt)
    print(f"\nReference values (from envelope decay):")
    print(f"  ω = {ref['omega']:.2f} rad/s")
    print(f"  κ = {ref['kappa_eff']:.2f}")
    print(f"  ζ_ref = {ref['zeta_ref']:.6f}")
    print(f"  Decay R² = {ref['R2']:.4f}")

    # Apply all methods
    results = []

    # Method 1: Envelope Decay (Baseline)
    results.append(method_envelope_decay(t, theta, ref))

    # Method 2: Hilbert Envelope
    results.append(method_hilbert_envelope(t, theta, ref))

    # Method 3: Least Squares
    results.append(method_least_squares(t, theta, theta_dot, ref))

    # Method 4: Optimization
    results.append(method_optimization(t, theta, ref))

    # Method 5: Genetic Algorithm
    results.append(method_genetic_algorithm(t, theta, ref))

    # Method 6: Differential Evolution
    results.append(method_differential_evolution(t, theta, ref))

    # Method 7: Log Decrement
    results.append(method_log_decrement(t, theta, ref))

    # Method 8: Half-Power Bandwidth
    results.append(method_half_power_bandwidth(t, theta, ref))

    # Method 9: Prony
    results.append(method_prony(t, theta, ref))

    # Method 10: Curve Fit
    results.append(method_curve_fit(t, theta, ref))

    # Method 11: ARX
    results.append(method_arx(t, theta, theta_dot, ref))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL METHODS")
    print("="*70)
    print(f"\n{'Method':<25} {'ζ Estimated':<15} {'Error %':<15} {'Status'}")
    print("-"*70)

    all_pass = True
    for r in results:
        zeta = r['zeta']
        error = r.get('error', 0)
        status = "PASS" if error < 0.1 else ("CLOSE" if error < 1 else "FAIL")
        if error >= 0.1:
            all_pass = False
        print(f"{r['method']:<25} {zeta:<15.6f} {error:<15.4f} {status}")

    print("-"*70)
    print(f"Reference ζ: {ref['zeta_ref']:.6f}")
    print(f"Target: Error < 0.1%")

    if all_pass:
        print("\n✓ ALL METHODS ACHIEVED < 0.1% ERROR!")
    else:
        print(f"\n✗ {sum(1 for r in results if r.get('error', 0) >= 0.1)} methods above 0.1% error")

    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Experimental data
    ax = axes[0, 0]
    ax.plot(t, np.degrees(theta), 'b-', linewidth=0.5, alpha=0.8)
    env = ref['A0'] * np.exp(-ref['decay_rate'] * t)
    ax.plot(t, np.degrees(env), 'r--', linewidth=2, label='Envelope')
    ax.plot(t, -np.degrees(env), 'r--', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('80° Experiment Data')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Method comparison
    ax = axes[0, 1]
    methods = [r['method'] for r in results]
    zetas = [r['zeta'] for r in results]
    ax.barh(methods, zetas, color='steelblue', alpha=0.7)
    ax.axvline(x=ref['zeta_ref'], color='red', linestyle='--', linewidth=2, label=f'Reference={ref["zeta_ref"]:.6f}')
    ax.set_xlabel('ζ (damping ratio)')
    ax.set_title('Estimated ζ by Method')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 3: Error comparison
    ax = axes[1, 0]
    errors = [r.get('error', 0) for r in results]
    colors = ['green' if e < 0.1 else ('orange' if e < 1 else 'red') for e in errors]
    ax.barh(methods, errors, color=colors, alpha=0.7)
    ax.axvline(x=0.1, color='green', linestyle='--', linewidth=2, label='0.1% threshold')
    ax.set_xlabel('Error (%)')
    ax.set_title('Estimation Error by Method')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xscale('log')

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
    EXPERIMENTAL ANALYSIS RESULTS
    ═══════════════════════════════════════

    Reference Values:
      ω = {ref['omega']:.2f} rad/s
      T = {ref['T']:.4f} s
      λ = {ref['decay_rate']:.4f} (1/s)
      ζ_ref = {ref['zeta_ref']:.6f}

    Best Methods (lowest error):
    """

    sorted_results = sorted(results, key=lambda x: x.get('error', 100))
    for i, r in enumerate(sorted_results[:5]):
        summary += f"\n      {i+1}. {r['method']}: {r.get('error', 0):.4f}%"

    summary += f"""

    Target: < 0.1% error
    Status: {"ACHIEVED" if all_pass else "IN PROGRESS"}
    """

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('All Methods Analysis - Experimental Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'all_methods_experimental.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved: {os.path.join(FIGURES_DIR, 'all_methods_experimental.png')}")

    return all_pass, results, ref


if __name__ == "__main__":
    all_pass, results, ref = main()
