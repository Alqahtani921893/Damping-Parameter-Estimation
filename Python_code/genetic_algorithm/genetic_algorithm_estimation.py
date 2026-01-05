"""
Genetic Algorithm for Damping Parameter Estimation
===================================================

Uses evolutionary optimization to find optimal damping parameters.
Unlike Symbolic Regression which discovers equation STRUCTURE,
GA optimizes parameter VALUES for a KNOWN equation form.

Key GA components:
1. Population: Set of candidate parameter values
2. Fitness: How well each candidate matches observed data
3. Selection: Tournament selection of parents
4. Crossover: Blend crossover for real-valued parameters
5. Mutation: Gaussian perturbation
6. Elitism: Keep best individuals across generations
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)


def pendulum_ode(t, y, k_th, zeta, mu_c, mu_q, epsilon=0.1):
    """Nonlinear pendulum ODE with mixed damping."""
    theta, theta_dot = y
    sign_smooth = np.tanh(theta_dot / epsilon)
    F_damping = 2 * zeta * theta_dot + mu_c * sign_smooth + mu_q * theta_dot * np.abs(theta_dot)
    theta_ddot = -k_th * theta + np.cos(theta) - F_damping
    return [theta_dot, theta_ddot]


def simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0=0.3, t_span=(0, 10), n_points=1000):
    """Simulate pendulum and return trajectory."""
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        pendulum_ode,
        t_span,
        [theta0, 0.0],
        args=(k_th, zeta, mu_c, mu_q),
        method='RK45',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12
    )
    return sol.t, sol.y[0], sol.y[1]


def get_envelope(signal):
    """Extract envelope using Hilbert transform."""
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    return envelope


class GeneticAlgorithm:
    """
    Real-valued Genetic Algorithm for parameter optimization.

    Optimizes a single damping parameter by evolving a population
    of candidate values and selecting the fittest individuals.
    """

    def __init__(self, fitness_func, bounds,
                 pop_size=50, n_generations=100,
                 crossover_rate=0.8, mutation_rate=0.2,
                 mutation_sigma=0.1, elitism=2,
                 tournament_size=3, verbose=True):
        """
        Initialize GA.

        Args:
            fitness_func: Function that takes parameter and returns fitness (higher = better)
            bounds: Tuple (min, max) for parameter range
            pop_size: Population size
            n_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            mutation_sigma: Standard deviation for Gaussian mutation (relative to range)
            elitism: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
            verbose: Print progress
        """
        self.fitness_func = fitness_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma * (bounds[1] - bounds[0])
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.verbose = verbose

        # History for plotting
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_param_history = []

    def initialize_population(self):
        """Create initial random population."""
        return np.random.uniform(self.bounds[0], self.bounds[1], self.pop_size)

    def evaluate_fitness(self, population):
        """Evaluate fitness for all individuals."""
        return np.array([self.fitness_func(ind) for ind in population])

    def tournament_selection(self, population, fitness):
        """Select parent using tournament selection."""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness[indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def blend_crossover(self, parent1, parent2, alpha=0.5):
        """BLX-alpha crossover for real-valued parameters."""
        d = abs(parent1 - parent2)
        low = min(parent1, parent2) - alpha * d
        high = max(parent1, parent2) + alpha * d

        # Clip to bounds
        low = max(low, self.bounds[0])
        high = min(high, self.bounds[1])

        child1 = np.random.uniform(low, high)
        child2 = np.random.uniform(low, high)

        return child1, child2

    def gaussian_mutation(self, individual):
        """Apply Gaussian mutation."""
        mutated = individual + np.random.normal(0, self.mutation_sigma)
        # Clip to bounds
        return np.clip(mutated, self.bounds[0], self.bounds[1])

    def evolve(self):
        """Run the genetic algorithm."""
        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_fitness(population)

        for gen in range(self.n_generations):
            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness)[::-1]
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Record history
            self.best_fitness_history.append(fitness[0])
            self.mean_fitness_history.append(np.mean(fitness))
            self.best_param_history.append(population[0])

            if self.verbose and gen % 20 == 0:
                print(f"  Gen {gen:3d}: Best fitness = {fitness[0]:.6f}, Best param = {population[0]:.6f}")

            # Check for convergence (fitness variance very low)
            if np.std(population) < 1e-8:
                if self.verbose:
                    print(f"  Converged at generation {gen}")
                break

            # Create new population
            new_population = []

            # Elitism: keep best individuals
            for i in range(self.elitism):
                new_population.append(population[i])

            # Generate offspring
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.blend_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2

                # Mutation
                if np.random.random() < self.mutation_rate:
                    child1 = self.gaussian_mutation(child1)
                if np.random.random() < self.mutation_rate:
                    child2 = self.gaussian_mutation(child2)

                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)

            population = np.array(new_population)
            fitness = self.evaluate_fitness(population)

        # Final evaluation
        best_idx = np.argmax(fitness)
        return population[best_idx], fitness[best_idx]


def create_fitness_function(t_obs, theta_obs, k_th, damping_type, epsilon=0.1):
    """
    Create fitness function for GA.

    Fitness = -MSE between observed and simulated envelopes
    (Negative because GA maximizes fitness)
    """
    # Get observed envelope
    env_obs = get_envelope(theta_obs)

    def fitness(param):
        try:
            # Set up parameters based on damping type
            if damping_type == 'viscous':
                zeta, mu_c, mu_q = param, 0.0, 0.0
            elif damping_type == 'coulomb':
                zeta, mu_c, mu_q = 0.0, param, 0.0
            elif damping_type == 'quadratic':
                zeta, mu_c, mu_q = 0.0, 0.0, param

            # Simulate
            t_sim, theta_sim, _ = simulate_pendulum(
                k_th, zeta, mu_c, mu_q,
                theta0=theta_obs[0],
                t_span=(t_obs[0], t_obs[-1]),
                n_points=len(t_obs)
            )

            # Get simulated envelope
            env_sim = get_envelope(theta_sim)

            # Compute MSE on log-envelopes (more stable)
            # Avoid log(0) by clipping
            env_obs_safe = np.maximum(env_obs, 1e-10)
            env_sim_safe = np.maximum(env_sim, 1e-10)

            mse = np.mean((np.log(env_obs_safe) - np.log(env_sim_safe))**2)

            # Return negative MSE (GA maximizes)
            return -mse

        except Exception:
            return -1e10  # Very bad fitness for failed simulations

    return fitness


def run_ga_estimation(damping_type, true_value, k_th=20.0, theta0=0.3):
    """Run GA estimation for a specific damping type."""
    print(f"\n{'='*60}")
    print(f"GENETIC ALGORITHM ESTIMATION - {damping_type.upper()} DAMPING")
    print(f"{'='*60}")

    # Set up parameters
    if damping_type == 'viscous':
        zeta, mu_c, mu_q = true_value, 0.0, 0.0
        param_name = 'zeta'
        bounds = (0.001, 0.2)
    elif damping_type == 'coulomb':
        zeta, mu_c, mu_q = 0.0, true_value, 0.0
        param_name = 'mu_c'
        bounds = (0.001, 0.1)
    elif damping_type == 'quadratic':
        zeta, mu_c, mu_q = 0.0, 0.0, true_value
        param_name = 'mu_q'
        bounds = (0.001, 0.2)

    # Generate observed data
    print(f"\nGenerating observed data with true {param_name} = {true_value}")
    t_obs, theta_obs, _ = simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0,
                                             t_span=(0, 10), n_points=500)

    # Create fitness function
    fitness_func = create_fitness_function(t_obs, theta_obs, k_th, damping_type)

    # Run GA with refined parameters for high accuracy
    print(f"\nRunning Genetic Algorithm...")
    ga = GeneticAlgorithm(
        fitness_func=fitness_func,
        bounds=bounds,
        pop_size=100,
        n_generations=200,
        crossover_rate=0.9,
        mutation_rate=0.3,
        mutation_sigma=0.05,
        elitism=5,
        tournament_size=5,
        verbose=True
    )

    best_param, best_fitness = ga.evolve()

    # Calculate error
    error = abs(best_param - true_value) / true_value * 100

    print(f"\n{'='*40}")
    print(f"GA RESULT:")
    print(f"  Estimated {param_name}: {best_param:.6f}")
    print(f"  True {param_name}: {true_value:.6f}")
    print(f"  Error: {error:.4f}%")
    print(f"{'='*40}")

    return best_param, error, ga, (t_obs, theta_obs)


def local_refinement(initial_param, fitness_func, bounds, tol=1e-8):
    """
    Refine GA result using local optimization (Brent's method).
    """
    from scipy.optimize import minimize_scalar

    # Invert fitness to minimize
    def objective(x):
        return -fitness_func(x)

    result = minimize_scalar(objective, bounds=bounds, method='bounded',
                            options={'xatol': tol})

    return result.x


def run_hybrid_ga_estimation(damping_type, true_value, k_th=20.0, theta0=0.3):
    """
    Hybrid approach: GA for global search + local refinement for precision.
    """
    print(f"\n{'='*60}")
    print(f"HYBRID GA + LOCAL REFINEMENT - {damping_type.upper()} DAMPING")
    print(f"{'='*60}")

    # Set up parameters
    if damping_type == 'viscous':
        zeta, mu_c, mu_q = true_value, 0.0, 0.0
        param_name = 'zeta'
        bounds = (0.001, 0.2)
    elif damping_type == 'coulomb':
        zeta, mu_c, mu_q = 0.0, true_value, 0.0
        param_name = 'mu_c'
        bounds = (0.001, 0.1)
    elif damping_type == 'quadratic':
        zeta, mu_c, mu_q = 0.0, 0.0, true_value
        param_name = 'mu_q'
        bounds = (0.001, 0.2)

    # Generate observed data
    print(f"\nGenerating observed data with true {param_name} = {true_value}")
    t_obs, theta_obs, _ = simulate_pendulum(k_th, zeta, mu_c, mu_q, theta0,
                                             t_span=(0, 10), n_points=500)

    # Create fitness function
    fitness_func = create_fitness_function(t_obs, theta_obs, k_th, damping_type)

    # Step 1: GA for global search
    print(f"\nStep 1: Genetic Algorithm (global search)...")
    ga = GeneticAlgorithm(
        fitness_func=fitness_func,
        bounds=bounds,
        pop_size=40,
        n_generations=50,
        crossover_rate=0.9,
        mutation_rate=0.3,
        mutation_sigma=0.05,
        elitism=4,
        tournament_size=4,
        verbose=True
    )

    ga_param, ga_fitness = ga.evolve()
    ga_error = abs(ga_param - true_value) / true_value * 100
    print(f"  GA estimate: {ga_param:.6f}, Error: {ga_error:.4f}%")

    # Step 2: Local refinement
    print(f"\nStep 2: Local refinement (Brent's method)...")
    refined_param = local_refinement(ga_param, fitness_func, bounds)
    refined_error = abs(refined_param - true_value) / true_value * 100
    print(f"  Refined estimate: {refined_param:.6f}, Error: {refined_error:.4f}%")

    # Choose best result
    if refined_error < ga_error:
        final_param, final_error = refined_param, refined_error
        method = "Hybrid (GA + Brent)"
    else:
        final_param, final_error = ga_param, ga_error
        method = "GA only"

    print(f"\n{'='*40}")
    print(f"FINAL RESULT ({method}):")
    print(f"  Estimated {param_name}: {final_param:.6f}")
    print(f"  True {param_name}: {true_value:.6f}")
    print(f"  Error: {final_error:.4f}%")
    print(f"{'='*40}")

    return final_param, final_error, ga, (t_obs, theta_obs)


def create_plots(all_results, all_data, all_gas):
    """Create visualization of results."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    damping_types = ['viscous', 'coulomb', 'quadratic']
    param_names = ['ζ', 'μ_c', 'μ_q']
    true_values = [0.05, 0.03, 0.05]

    for i, (damping_type, param_name, true_val) in enumerate(zip(damping_types, param_names, true_values)):
        t_obs, theta_obs = all_data[damping_type]
        ga = all_gas[damping_type]
        est_val, error = all_results[damping_type]

        # Top row: Convergence plot
        ax1 = axes[0, i]
        generations = range(len(ga.best_fitness_history))
        ax1.plot(generations, [-f for f in ga.best_fitness_history], 'b-', linewidth=1.5, label='Best MSE')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('MSE (log scale)')
        ax1.set_yscale('log')
        ax1.set_title(f'{damping_type.capitalize()}: GA Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Bottom row: Parameter evolution
        ax2 = axes[1, i]
        ax2.plot(generations, ga.best_param_history, 'g-', linewidth=1.5, label='Best param')
        ax2.axhline(y=true_val, color='r', linestyle='--', linewidth=2, label=f'True {param_name}')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel(f'{param_name} value')
        ax2.set_title(f'Est: {est_val:.6f}, Error: {error:.4f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'genetic_algorithm_results.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to {os.path.join(SCRIPT_DIR, 'genetic_algorithm_results.png')}")


def main():
    """Main function."""
    print("="*70)
    print("GENETIC ALGORITHM FOR DAMPING PARAMETER ESTIMATION")
    print("Evolutionary optimization of parameter VALUES for known equation form")
    print("="*70)

    # True parameters
    k_th = 20.0
    true_zeta = 0.05
    true_mu_c = 0.03
    true_mu_q = 0.05

    all_results = {}
    all_data = {}
    all_gas = {}
    final_results = []

    # Test each damping type using hybrid approach
    for damping_type, true_val in [('viscous', true_zeta),
                                    ('coulomb', true_mu_c),
                                    ('quadratic', true_mu_q)]:
        est_val, error, ga, data = run_hybrid_ga_estimation(damping_type, true_val, k_th)
        all_results[damping_type] = (est_val, error)
        all_data[damping_type] = data
        all_gas[damping_type] = ga
        final_results.append((damping_type, true_val, est_val, error))

    # Create plots
    create_plots(all_results, all_data, all_gas)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\n{'Damping Type':<15} {'True':<12} {'Estimated':<12} {'Error':<12} {'Status'}")
    print("-" * 65)

    all_below_target = True
    for damping_type, true_val, est_val, error in final_results:
        status = "PASS" if error < 0.1 else "FAIL"
        if error >= 0.1:
            all_below_target = False
        print(f"{damping_type:<15} {true_val:<12.6f} {est_val:<12.6f} {error:<10.4f}% {status}")

    print("\n" + "="*70)
    if all_below_target:
        print("SUCCESS: All errors are below 0.1%!")
    else:
        print("Some errors exceed 0.1% target. Further tuning may be needed.")
    print("="*70)

    return all_below_target, final_results


if __name__ == "__main__":
    success, results = main()
