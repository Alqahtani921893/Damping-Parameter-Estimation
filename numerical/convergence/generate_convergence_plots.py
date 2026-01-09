#!/usr/bin/env python3
"""
Convergence Plots for Iterative Damping Parameter Estimation Methods

Generates parameter value vs epoch/iteration plots for all iterative methods:
1. Physics-Informed Neural Networks (PINNs)
2. Neural ODEs
3. Genetic Algorithm

Uses the existing implementations from their respective modules which have
proven excellent accuracy.
"""

import sys
import os

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)

FIGURES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# System parameters (consistent across all methods)
K_TH = 20.0
THETA0_DEG = 30
THETA0_RAD = np.radians(THETA0_DEG)
T_FINAL = 60
DT = 0.002
EPSILON = 0.1

# True parameters
TRUE_PARAMS = {
    'viscous': 0.05,
    'coulomb': 0.03,
    'quadratic': 0.05
}

PARAM_NAMES = {
    'viscous': r'$\zeta$',
    'coulomb': r'$\mu_c$',
    'quadratic': r'$\mu_q$'
}


def run_pinn_estimation(damping_type, true_param):
    """Run PINN estimation and return convergence history."""
    print(f"\n  Running PINN for {damping_type} damping...")

    # Import from existing module
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'pinns_estimation'))
    from pinn_damping_estimation import estimate_damping_pinn

    # Run estimation
    if damping_type == 'viscous':
        results = estimate_damping_pinn(
            damping_type='viscous', true_zeta=true_param, k_th=K_TH,
            theta0_deg=THETA0_DEG, t_final=T_FINAL, dt=DT,
            noise_std=0.001, epochs=3000, plotting=False
        )
    elif damping_type == 'coulomb':
        results = estimate_damping_pinn(
            damping_type='coulomb', true_mu_c=true_param, k_th=K_TH,
            theta0_deg=THETA0_DEG, t_final=T_FINAL, dt=DT,
            noise_std=0.001, epochs=3000, plotting=False
        )
    else:  # quadratic
        results = estimate_damping_pinn(
            damping_type='quadratic', true_mu_q=true_param, k_th=K_TH,
            theta0_deg=THETA0_DEG, t_final=T_FINAL, dt=DT,
            noise_std=0.001, epochs=3000, plotting=False
        )

    # Extract convergence history
    history = results['history']

    if damping_type == 'viscous':
        param_history = history['zeta']
        final_param = results['direct']['zeta']
    elif damping_type == 'coulomb':
        param_history = history['mu_c']
        final_param = results['direct']['mu_c']
    else:
        param_history = history['mu_q']
        final_param = results['direct']['mu_q']

    error = abs(final_param - true_param) / true_param * 100

    return {
        'epochs': list(range(len(param_history))),
        'params': param_history,
        'loss': history['loss'],
        'final_param': final_param,
        'error': error
    }


def run_neural_ode_estimation(damping_type, true_param):
    """Run Neural ODE estimation and return convergence history."""
    print(f"\n  Running Neural ODE for {damping_type} damping...")

    # Import from existing module
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'neural_odes'))
    from neural_ode_estimation import hybrid_neural_ode_estimation, simulate_pendulum

    # Generate data
    if damping_type == 'viscous':
        t, theta, theta_dot = simulate_pendulum(K_TH, zeta=true_param, mu_c=0, mu_q=0,
                                                 theta0_deg=THETA0_DEG, t_final=T_FINAL, dt=DT)
    elif damping_type == 'coulomb':
        t, theta, theta_dot = simulate_pendulum(K_TH, zeta=0, mu_c=true_param, mu_q=0,
                                                 theta0_deg=THETA0_DEG, t_final=T_FINAL, dt=DT)
    else:
        t, theta, theta_dot = simulate_pendulum(K_TH, zeta=0, mu_c=0, mu_q=true_param,
                                                 theta0_deg=THETA0_DEG, t_final=T_FINAL, dt=DT)

    # Add small noise
    theta_noisy = theta + np.random.normal(0, 0.001, len(theta))
    theta_dot_noisy = theta_dot + np.random.normal(0, 0.001, len(theta_dot))

    # Run estimation
    result = hybrid_neural_ode_estimation(
        t, theta_noisy, theta_dot_noisy, K_TH,
        damping_type, true_param, EPSILON, refine_epochs=300
    )

    return {
        'epochs': list(range(len(result['training_history']['param']))),
        'params': result['training_history']['param'],
        'loss': result['training_history']['loss'],
        'final_param': result['final_estimate'],
        'error': result['final_error']
    }


def run_ga_estimation(damping_type, true_param):
    """Run GA estimation and return convergence history."""
    print(f"\n  Running Genetic Algorithm for {damping_type} damping...")

    # Import from existing module
    sys.path.insert(0, os.path.join(SCRIPT_DIR, 'genetic_algorithm'))
    from genetic_algorithm_estimation import run_hybrid_ga_estimation

    # Run estimation
    final_param, error, ga, data = run_hybrid_ga_estimation(damping_type, true_param, K_TH, THETA0_RAD)

    return {
        'generations': list(range(len(ga.best_param_history))),
        'params': ga.best_param_history,
        'fitness': ga.best_fitness_history,
        'final_param': final_param,
        'error': error
    }


def generate_convergence_plots():
    """Generate convergence plots for all iterative methods."""
    print("=" * 70)
    print("GENERATING CONVERGENCE PLOTS")
    print("Using existing implementations with proven accuracy")
    print("=" * 70)

    damping_types = ['viscous', 'coulomb', 'quadratic']
    methods = ['PINN', 'Neural ODE', 'Genetic Algorithm']

    all_results = {method: {} for method in methods}

    for damping_type in damping_types:
        true_param = TRUE_PARAMS[damping_type]

        print(f"\n{'='*60}")
        print(f"Processing {damping_type.upper()} damping (true = {true_param})")
        print('='*60)

        # 1. PINN
        try:
            all_results['PINN'][damping_type] = run_pinn_estimation(damping_type, true_param)
            print(f"    PINN: {all_results['PINN'][damping_type]['final_param']:.6f}, "
                  f"Error: {all_results['PINN'][damping_type]['error']:.2f}%")
        except Exception as e:
            print(f"    PINN failed: {e}")
            all_results['PINN'][damping_type] = None

        # 2. Neural ODE
        try:
            all_results['Neural ODE'][damping_type] = run_neural_ode_estimation(damping_type, true_param)
            print(f"    Neural ODE: {all_results['Neural ODE'][damping_type]['final_param']:.6f}, "
                  f"Error: {all_results['Neural ODE'][damping_type]['error']:.4f}%")
        except Exception as e:
            print(f"    Neural ODE failed: {e}")
            all_results['Neural ODE'][damping_type] = None

        # 3. Genetic Algorithm
        try:
            all_results['Genetic Algorithm'][damping_type] = run_ga_estimation(damping_type, true_param)
            print(f"    GA: {all_results['Genetic Algorithm'][damping_type]['final_param']:.6f}, "
                  f"Error: {all_results['Genetic Algorithm'][damping_type]['error']:.4f}%")
        except Exception as e:
            print(f"    GA failed: {e}")
            all_results['Genetic Algorithm'][damping_type] = None

    # =========================================================================
    # Create Plots
    # =========================================================================
    print("\n" + "=" * 70)
    print("Creating convergence plots...")
    print("=" * 70)

    colors = {
        'PINN': 'blue',
        'Neural ODE': 'green',
        'Genetic Algorithm': 'red'
    }

    # Plot 1: Combined convergence plot (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    for i, damping_type in enumerate(damping_types):
        true_param = TRUE_PARAMS[damping_type]

        for j, method in enumerate(methods):
            ax = axes[i, j]
            result = all_results[method].get(damping_type)

            if result is None:
                ax.text(0.5, 0.5, 'Not available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{method}\n{damping_type.capitalize()}')
                continue

            x_data = result.get('epochs', result.get('generations', []))
            params = result['params']

            ax.plot(x_data, params, color=colors[method], linewidth=1.5, label='Estimated')
            ax.axhline(y=true_param, color='black', linestyle='--', linewidth=2, label=f'True = {true_param}')

            ax.set_xlabel('Epoch' if 'epochs' in result else 'Generation')
            ax.set_ylabel(PARAM_NAMES[damping_type])
            ax.set_title(f'{method}\n{damping_type.capitalize()}: Error = {result["error"]:.2f}%')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.suptitle('Parameter Convergence During Training\nAll Iterative Methods',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_convergence_all_methods.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_convergence_all_methods.png")

    # Plot 2: Per-damping type comparison
    for damping_type in damping_types:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        true_param = TRUE_PARAMS[damping_type]

        # Left: Parameter convergence
        ax = axes[0]
        for method in methods:
            result = all_results[method].get(damping_type)
            if result is None:
                continue
            x_data = result.get('epochs', result.get('generations', []))
            ax.plot(x_data, result['params'], color=colors[method], linewidth=1.5, label=method, alpha=0.8)

        ax.axhline(y=true_param, color='black', linestyle='--', linewidth=2,
                   label=f'True {PARAM_NAMES[damping_type]} = {true_param}')
        ax.set_xlabel('Epoch / Generation')
        ax.set_ylabel(f'Parameter Value ({PARAM_NAMES[damping_type]})')
        ax.set_title(f'{damping_type.capitalize()} Damping: Parameter Convergence')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Right: Final estimates bar chart
        ax = axes[1]
        method_names = []
        estimates = []
        errors = []
        bar_colors = []

        for method in methods:
            result = all_results[method].get(damping_type)
            if result is not None:
                method_names.append(method)
                estimates.append(result['final_param'])
                errors.append(result['error'])
                bar_colors.append(colors[method])

        x = np.arange(len(method_names))
        bars = ax.bar(x, estimates, color=bar_colors, alpha=0.8)
        ax.axhline(y=true_param, color='black', linestyle='--', linewidth=2, label=f'True = {true_param}')

        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_ylabel(f'{PARAM_NAMES[damping_type]} Value')
        ax.set_title(f'{damping_type.capitalize()} Damping: Final Estimates')

        # Add error labels on bars
        for bar, err in zip(bars, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{err:.2f}%', ha='center', va='bottom', fontsize=9)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Convergence Comparison: {damping_type.capitalize()} Damping',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'fig_convergence_{damping_type}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: fig_convergence_{damping_type}.png")

    # Plot 3: Final comparison bar chart (all damping types)
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(damping_types))
    width = 0.22
    offsets = [-width, 0, width]

    for j, method in enumerate(methods):
        final_values = []
        for dt in damping_types:
            result = all_results[method].get(dt)
            if result is not None:
                final_values.append(result['final_param'])
            else:
                final_values.append(0)
        ax.bar(x + offsets[j], final_values, width, label=method, color=colors[method], alpha=0.8)

    # True values as stars
    for i, (dt, tv) in enumerate(zip(damping_types, [TRUE_PARAMS[dt] for dt in damping_types])):
        ax.scatter(i, tv, s=200, marker='*', color='gold', edgecolors='black', zorder=5,
                   label='True' if i == 0 else '')

    ax.set_xlabel('Damping Type', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Final Parameter Estimates by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{dt.capitalize()}\n({PARAM_NAMES[dt]})' for dt in damping_types])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_convergence_final_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_convergence_final_comparison.png")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONVERGENCE RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<20} {'Damping':<12} {'True':<10} {'Final Est':<12} {'Error %':<10}")
    print("-" * 65)

    for method in methods:
        for damping_type in damping_types:
            true_param = TRUE_PARAMS[damping_type]
            result = all_results[method].get(damping_type)

            if result is not None:
                print(f"{method:<20} {damping_type:<12} {true_param:<10.4f} "
                      f"{result['final_param']:<12.6f} {result['error']:<10.4f}")
            else:
                print(f"{method:<20} {damping_type:<12} {true_param:<10.4f} {'N/A':<12} {'N/A':<10}")

    print("\n" + "=" * 70)
    print(f"Plots saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_convergence_plots()
