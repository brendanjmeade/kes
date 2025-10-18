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
from hdf5_io import load_lazy_results


def load_results(results_path):
    """
    Load simulation results from HDF5 or pickle file

    Automatically detects format based on file extension.
    HDF5 files use lazy loading for memory efficiency.

    Parameters:
    -----------
    results_path : str or Path
        Path to the results file (.h5 or .pkl)

    Returns:
    --------
    results : dict or HDF5Results
        Dictionary containing all simulation data and config
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    print(f"Loading results from: {results_path}")

    # Detect format
    if results_path.suffix in ['.h5', '.hdf5']:
        # Load HDF5 with lazy loading
        results = load_lazy_results(results_path)
        print("Results loaded successfully from HDF5!")
        print(f"  Events: {len(results['event_history'])}")
        print(f"  Duration: {results['config'].duration_years} years")
        print(
            f"  Grid: {results['config'].n_along_strike} x {results['config'].n_down_dip}"
        )
        print(f"  Storage: HDF5 (lazy loading)")

    else:
        # Load pickle (legacy)
        with open(results_path, "rb") as f:
            results = pickle.load(f)

        print("Results loaded successfully from pickle!")
        print(f"  Events: {len(results['event_history'])}")
        print(f"  Duration: {results['config'].duration_years} years")
        print(
            f"  Grid: {results['config'].n_along_strike} x {results['config'].n_down_dip}"
        )
        print(f"  Storage: Pickle (full in-memory)")

    return results


def main():
    """
    Main entry point for visualization script
    """
    # Parse command line arguments
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Default path (try HDF5 first, fall back to pickle)
        hdf5_path = Path("results/simulation_results.h5")
        pkl_path = Path("results/simulation_results.pkl")

        if hdf5_path.exists():
            results_path = hdf5_path
        elif pkl_path.exists():
            results_path = pkl_path
        else:
            results_path = "results/simulation_results.h5"  # Default to HDF5

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
