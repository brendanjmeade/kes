"""
Standalone visualization script for earthquake simulator results

Loads results from HDF5 file and generates all diagnostic plots.
This allows iterating on visualization without rerunning expensive simulations.

Usage:
    python plot_results.py [path_to_results.h5]

If no path is provided, defaults to results/simulation_results.h5
"""

import sys
import numpy as np
from pathlib import Path
from visualize import plot_all
from hdf5_io import load_lazy_results


def print_largest_events(results, n_events=20):
    """
    Print the n largest events sorted by magnitude

    Parameters:
    -----------
    results : dict
        Simulation results containing event_history and snapshot_times
    n_events : int
        Number of events to print (default 20)
    """
    event_history = results["event_history"]
    snapshot_times = results["snapshot_times"]

    if len(event_history) == 0:
        print("No events to display")
        return

    # Get magnitudes, times, and find snapshot indices
    events = []
    for i, e in enumerate(event_history):
        time = e["time"]
        mag = e["magnitude"]
        snapshot_idx = np.argmin(np.abs(np.array(snapshot_times) - time))
        events.append((i, time, mag, snapshot_idx))

    # Sort by magnitude (descending)
    events_sorted = sorted(events, key=lambda x: x[2], reverse=True)

    # Print header
    print(f"\nLargest {min(n_events, len(events_sorted))} events")
    print(f"{'Rank':<6} {'Event#':<8} {'Time (yr)':<12} {'Magnitude':<12} {'idx'}")

    for rank, (event_idx, time, mag, snapshot_idx) in enumerate(
        events_sorted[:n_events], 1
    ):
        print(f"{rank:<6} {event_idx:<8} {time:<12.2f} M{mag:<11.2f} {snapshot_idx}")


def load_results(results_path):
    """
    Load simulation results from HDF5 file

    Uses lazy loading for memory efficiency.

    Parameters:
    -----------
    results_path : str or Path
        Path to the results file (.h5)

    Returns:
    --------
    results : HDF5Results
        Lazy-loading wrapper for simulation data
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    # Lazy load HDF5
    print(f"Loading results from: {results_path}")
    results = load_lazy_results(results_path)
    print(f"Events: {len(results['event_history'])}")
    print(f"Duration: {results['config'].duration_years} years")
    print(f"Grid: {results['config'].n_along_strike} x {results['config'].n_down_dip}")

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
        results_path = Path("results/simulation_results.h5")

    # Load results
    results = load_results(results_path)
    config = results["config"]

    # Print largest events
    print_largest_events(results, n_events=20)

    # Generate all plots
    plot_all(results, config)


if __name__ == "__main__":
    main()
