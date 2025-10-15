"""
Main entry point for strike-slip fault earthquake simulator

This script runs the simulation and saves results to a pickle file.
To generate plots from saved results, use plot_results.py
"""

import sys
from config import Config
from simulator import run_simulation, save_results


def main(generate_plots=False):
    """
    Run earthquake simulation

    Parameters:
    -----------
    generate_plots : bool
        If True, generate plots immediately after simulation.
        If False (default), only save results. Use plot_results.py later.

    Returns:
    --------
    results : dict
        Simulation results dictionary
    """
    print("=" * 70)
    print("STRIKE-SLIP FAULT EARTHQUAKE SIMULATOR")
    print("=" * 70)

    # Create configuration
    config = Config()

    # Run simulation
    results = run_simulation(config)

    # Save results
    output_path = save_results(results, config)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {output_path}")
    print(f"\nTo generate plots, run:")
    print(f"  python plot_results.py {output_path}")

    # Optionally generate plots immediately
    if generate_plots:
        print("\nGenerating plots...")
        from visualize import plot_all_diagnostics

        plot_all_diagnostics(results, config)
        print(f"Plots saved to: {config.output_dir}/")

    return results


if __name__ == "__main__":
    # Check if --plot flag is provided
    generate_plots = "--plot" in sys.argv

    results = main(generate_plots=generate_plots)
