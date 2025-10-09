"""
Main entry point for strike-slip fault earthquake simulator
"""

from config import Config
from simulator import run_simulation, save_results
from visualize import plot_all_diagnostics


def main():
    """
    Run complete simulation pipeline
    """
    print("Strike-Slip Fault Earthquake Simulator")

    # Create configuration
    config = Config()

    # Run simulation
    results = run_simulation(config)

    # Save results
    save_results(results, config)

    # Generate plots
    plot_all_diagnostics(results, config)

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
