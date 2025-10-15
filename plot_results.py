"""
Standalone visualization script for earthquake simulator results

Loads results from simulation_results.pkl and generates all diagnostic plots.
This allows iterating on visualization without rerunning expensive simulations.

Usage:
    python plot_results.py [path_to_results.pkl]

If no path is provided, defaults to results/simulation_results.pkl
"""

import sys
import pickle
from pathlib import Path
from visualize import plot_all_diagnostics


def load_results(results_path):
    """
    Load simulation results from pickle file

    Parameters:
    -----------
    results_path : str or Path
        Path to the simulation_results.pkl file

    Returns:
    --------
    results : dict
        Dictionary containing all simulation data and config
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    print(f"Loading results from: {results_path}")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    print("Results loaded successfully!")
    print(f"  Events: {len(results['event_history'])}")
    print(f"  Duration: {results['config'].duration_years} years")
    print(
        f"  Grid: {results['config'].n_along_strike} x {results['config'].n_down_dip}"
    )

    return results


def main():
    """
    Main entry point for visualization script
    """
    # Parse command line arguments
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Default path
        results_path = "results/simulation_results.pkl"

    print("=" * 70)
    print("EARTHQUAKE SIMULATOR - VISUALIZATION")
    print("=" * 70)

    # Load results
    results = load_results(results_path)
    config = results["config"]

    # Generate all plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_all_diagnostics(results, config)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"Plots saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
