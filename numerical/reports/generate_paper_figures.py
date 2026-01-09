#!/usr/bin/env python3
"""
Generate figures for the research paper on damping parameter estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import os

# Set style for publication-quality figures
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Results data for all methods
methods_data = {
    'Topological': {'viscous': 77.6, 'coulomb': 31.0, 'quadratic': 20.0, 'category': 'Classical'},
    'Optimization': {'viscous': 0.03, 'coulomb': 0.04, 'quadratic': 0.03, 'category': 'Optimization'},
    'SINDy': {'viscous': 0.15, 'coulomb': 2.2, 'quadratic': 0.24, 'category': 'ML'},
    'PINNs': {'viscous': 0.15, 'coulomb': 0.41, 'quadratic': 0.06, 'category': 'ML'},
    'Neural ODEs': {'viscous': 0.11, 'coulomb': 0.04, 'quadratic': 0.04, 'category': 'ML'},
    'RNN/LSTM': {'viscous': 0.01, 'coulomb': 0.07, 'quadratic': 0.01, 'category': 'ML'},
    'Symbolic Reg.': {'viscous': 0.15, 'coulomb': 0.39, 'quadratic': 0.07, 'category': 'ML'},
    'Weak SINDy': {'viscous': 0.15, 'coulomb': 0.39, 'quadratic': 0.07, 'category': 'ML'},
    'Least Squares': {'viscous': 0.0004, 'coulomb': 0.006, 'quadratic': 0.001, 'category': 'Classical'},
    'Genetic Alg.': {'viscous': 0.0001, 'coulomb': 0.0001, 'quadratic': 0.0001, 'category': 'Optimization'},
    'Koopman/EDMD': {'viscous': 0.0001, 'coulomb': 0.0001, 'quadratic': 0.0001, 'category': 'Classical'},
}

def fig1_error_comparison_bar():
    """Bar chart comparing errors across all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(methods_data.keys())
    x = np.arange(len(methods))
    width = 0.25

    viscous = [methods_data[m]['viscous'] for m in methods]
    coulomb = [methods_data[m]['coulomb'] for m in methods]
    quadratic = [methods_data[m]['quadratic'] for m in methods]

    bars1 = ax.bar(x - width, viscous, width, label='Viscous', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, coulomb, width, label='Coulomb', color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, quadratic, width, label='Quadratic', color='#3498db', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Estimation Error (%)')
    ax.set_xlabel('Method')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 100)
    ax.legend(loc='upper right')
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, label='0.1% threshold')

    plt.tight_layout()
    plt.savefig('fig_error_comparison_bar.pdf')
    plt.savefig('fig_error_comparison_bar.png')
    plt.close()
    print("Generated: fig_error_comparison_bar.pdf")

def fig2_error_heatmap():
    """Heatmap of errors for all methods and damping types."""
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = list(methods_data.keys())
    damping_types = ['viscous', 'coulomb', 'quadratic']

    # Create error matrix (log scale for visualization)
    errors = np.array([[methods_data[m][d] for d in damping_types] for m in methods])
    log_errors = np.log10(errors + 1e-6)  # Add small value to avoid log(0)

    im = ax.imshow(log_errors, cmap='RdYlGn_r', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(Error %)')

    # Set ticks
    ax.set_xticks(np.arange(len(damping_types)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(['Viscous', 'Coulomb', 'Quadratic'])
    ax.set_yticklabels(methods)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(damping_types)):
            val = errors[i, j]
            if val < 0.01:
                text = f'{val:.4f}'
            elif val < 1:
                text = f'{val:.2f}'
            else:
                text = f'{val:.1f}'
            color = 'white' if log_errors[i, j] > 0 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

    ax.set_xlabel('Damping Type')
    ax.set_ylabel('Method')

    plt.tight_layout()
    plt.savefig('fig_error_heatmap.pdf')
    plt.savefig('fig_error_heatmap.png')
    plt.close()
    print("Generated: fig_error_heatmap.pdf")

def fig3_category_comparison():
    """Compare methods by category."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    categories = {'Classical': '#3498db', 'ML': '#e74c3c', 'Optimization': '#2ecc71'}
    damping_types = ['viscous', 'coulomb', 'quadratic']
    titles = ['Viscous Damping', 'Coulomb Friction', 'Quadratic Damping']

    for idx, (dtype, title) in enumerate(zip(damping_types, titles)):
        ax = axes[idx]

        for method, data in methods_data.items():
            cat = data['category']
            error = data[dtype]
            color = categories[cat]
            ax.scatter(method, error, c=color, s=100, edgecolors='black', linewidth=0.5, zorder=3)

        ax.set_yscale('log')
        ax.set_ylim(1e-5, 100)
        ax.set_ylabel('Error (%)' if idx == 0 else '')
        ax.set_title(title)
        ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(ax.get_xticklabels(), ha='right')

    # Add legend
    legend_elements = [mpatches.Patch(facecolor=c, edgecolor='black', label=cat)
                       for cat, c in categories.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.savefig('fig_category_comparison.pdf')
    plt.savefig('fig_category_comparison.png')
    plt.close()
    print("Generated: fig_category_comparison.pdf")

def fig4_ranking():
    """Ranking of methods by average error."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate average error for each method
    avg_errors = {}
    for method, data in methods_data.items():
        avg = (data['viscous'] + data['coulomb'] + data['quadratic']) / 3
        avg_errors[method] = avg

    # Sort by error
    sorted_methods = sorted(avg_errors.items(), key=lambda x: x[1])
    methods = [m[0] for m in sorted_methods]
    errors = [m[1] for m in sorted_methods]

    colors = ['#2ecc71' if e < 0.1 else '#f39c12' if e < 1 else '#e74c3c' for e in errors]

    bars = ax.barh(methods, errors, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Average Error (%)')
    ax.set_xlim(1e-5, 50)
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=1.5, label='0.1% threshold')

    # Add value labels
    for bar, error in zip(bars, errors):
        if error < 0.01:
            label = f'{error:.4f}%'
        elif error < 1:
            label = f'{error:.2f}%'
        else:
            label = f'{error:.1f}%'
        ax.text(bar.get_width() * 1.5, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8)

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('fig_ranking.pdf')
    plt.savefig('fig_ranking.png')
    plt.close()
    print("Generated: fig_ranking.pdf")

def fig5_pendulum_simulation():
    """Show example pendulum oscillation with damping."""
    from scipy.integrate import solve_ivp

    def pendulum_ode(t, y, k_th, zeta):
        theta, theta_dot = y
        theta_ddot = -k_th * theta + np.cos(theta) - 2 * zeta * theta_dot
        return [theta_dot, theta_ddot]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    k_th = 20.0
    theta0 = 0.3
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)

    # Plot 1: Time series for different damping values
    ax = axes[0, 0]
    for zeta, color, label in [(0.02, '#3498db', 'ζ=0.02'),
                                (0.05, '#2ecc71', 'ζ=0.05'),
                                (0.1, '#e74c3c', 'ζ=0.10')]:
        sol = solve_ivp(pendulum_ode, t_span, [theta0, 0], args=(k_th, zeta), t_eval=t_eval)
        ax.plot(sol.t, sol.y[0], color=color, label=label, linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (rad)')
    ax.set_title('(a) Viscous Damping - Time Response')
    ax.legend()

    # Plot 2: Phase portrait
    ax = axes[0, 1]
    for zeta, color in [(0.02, '#3498db'), (0.05, '#2ecc71'), (0.1, '#e74c3c')]:
        sol = solve_ivp(pendulum_ode, t_span, [theta0, 0], args=(k_th, zeta), t_eval=t_eval)
        ax.plot(sol.y[0], sol.y[1], color=color, linewidth=1.5)
    ax.set_xlabel('θ (rad)')
    ax.set_ylabel('θ̇ (rad/s)')
    ax.set_title('(b) Phase Portrait')

    # Plot 3: Envelope decay
    ax = axes[1, 0]
    from scipy.signal import hilbert
    for zeta, color in [(0.02, '#3498db'), (0.05, '#2ecc71'), (0.1, '#e74c3c')]:
        sol = solve_ivp(pendulum_ode, t_span, [theta0, 0], args=(k_th, zeta), t_eval=t_eval)
        envelope = np.abs(hilbert(sol.y[0]))
        ax.plot(sol.t, envelope, color=color, linewidth=1.5, label=f'ζ={zeta}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Envelope Amplitude')
    ax.set_title('(c) Envelope Decay')
    ax.legend()

    # Plot 4: Log envelope
    ax = axes[1, 1]
    for zeta, color in [(0.02, '#3498db'), (0.05, '#2ecc71'), (0.1, '#e74c3c')]:
        sol = solve_ivp(pendulum_ode, t_span, [theta0, 0], args=(k_th, zeta), t_eval=t_eval)
        envelope = np.abs(hilbert(sol.y[0]))
        ax.plot(sol.t, np.log(envelope + 1e-10), color=color, linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ln(Envelope)')
    ax.set_title('(d) Log-Envelope (Linear for Viscous)')

    plt.tight_layout()
    plt.savefig('fig_pendulum_simulation.pdf')
    plt.savefig('fig_pendulum_simulation.png')
    plt.close()
    print("Generated: fig_pendulum_simulation.pdf")

def fig6_damping_types():
    """Compare the three damping types."""
    from scipy.integrate import solve_ivp

    def pendulum_viscous(t, y, k_th, param):
        theta, theta_dot = y
        F_damp = 2 * param * theta_dot
        return [theta_dot, -k_th * theta + np.cos(theta) - F_damp]

    def pendulum_coulomb(t, y, k_th, param):
        theta, theta_dot = y
        F_damp = param * np.tanh(theta_dot / 0.1)
        return [theta_dot, -k_th * theta + np.cos(theta) - F_damp]

    def pendulum_quadratic(t, y, k_th, param):
        theta, theta_dot = y
        F_damp = param * theta_dot * np.abs(theta_dot)
        return [theta_dot, -k_th * theta + np.cos(theta) - F_damp]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    k_th = 20.0
    theta0 = 0.3
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)

    configs = [
        (pendulum_viscous, 0.05, 'Viscous (ζ=0.05)', '#2ecc71'),
        (pendulum_coulomb, 0.03, 'Coulomb (μc=0.03)', '#e74c3c'),
        (pendulum_quadratic, 0.05, 'Quadratic (μq=0.05)', '#3498db'),
    ]

    for idx, (ode_func, param, title, color) in enumerate(configs):
        ax = axes[idx]
        sol = solve_ivp(ode_func, t_span, [theta0, 0], args=(k_th, param), t_eval=t_eval)
        ax.plot(sol.t, sol.y[0], color=color, linewidth=1.5)
        ax.fill_between(sol.t, sol.y[0], alpha=0.3, color=color)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('θ (rad)' if idx == 0 else '')
        ax.set_title(title)
        ax.set_ylim(-0.35, 0.35)

    plt.tight_layout()
    plt.savefig('fig_damping_types.pdf')
    plt.savefig('fig_damping_types.png')
    plt.close()
    print("Generated: fig_damping_types.pdf")

def fig7_ml_vs_classical():
    """Compare ML methods vs classical methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ml_methods = ['SINDy', 'PINNs', 'Neural ODEs', 'RNN/LSTM', 'Symbolic Reg.', 'Weak SINDy']
    classical_methods = ['Topological', 'Least Squares', 'Koopman/EDMD']
    opt_methods = ['Optimization', 'Genetic Alg.']

    def get_avg_error(method):
        d = methods_data[method]
        return (d['viscous'] + d['coulomb'] + d['quadratic']) / 3

    ml_errors = [get_avg_error(m) for m in ml_methods]
    classical_errors = [get_avg_error(m) for m in classical_methods]
    opt_errors = [get_avg_error(m) for m in opt_methods]

    x_ml = np.arange(len(ml_methods))
    x_classical = np.arange(len(classical_methods)) + len(ml_methods) + 0.5
    x_opt = np.arange(len(opt_methods)) + len(ml_methods) + len(classical_methods) + 1

    bars1 = ax.bar(x_ml, ml_errors, color='#e74c3c', edgecolor='black', linewidth=0.5, label='Machine Learning')
    bars2 = ax.bar(x_classical, classical_errors, color='#3498db', edgecolor='black', linewidth=0.5, label='Classical')
    bars3 = ax.bar(x_opt, opt_errors, color='#2ecc71', edgecolor='black', linewidth=0.5, label='Optimization-based')

    ax.set_yscale('log')
    ax.set_ylabel('Average Error (%)')
    ax.set_ylim(1e-5, 50)
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5)

    all_x = list(x_ml) + list(x_classical) + list(x_opt)
    all_labels = ml_methods + classical_methods + opt_methods
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('fig_ml_vs_classical.pdf')
    plt.savefig('fig_ml_vs_classical.png')
    plt.close()
    print("Generated: fig_ml_vs_classical.pdf")

def main():
    """Generate all figures."""
    print("Generating research paper figures...")
    print("=" * 50)

    fig1_error_comparison_bar()
    fig2_error_heatmap()
    fig3_category_comparison()
    fig4_ranking()
    fig5_pendulum_simulation()
    fig6_damping_types()
    fig7_ml_vs_classical()

    print("=" * 50)
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
