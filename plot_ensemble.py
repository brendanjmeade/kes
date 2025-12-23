"""
Visualization script for ensemble simulation results

Generates a multi-panel figure showing earthquake magnitude time series
for each ensemble member, allowing comparison of how earthquake sequences
evolve under different loading conditions.

Usage:
    python plot_ensemble.py [--input_dir DIR] [--output OUTPUT]

Defaults:
    input_dir = results/ensemble
    output = results/ensemble/ensemble_comparison.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hdf5_io import load_lazy_results

FONTSIZE = 8


def plot_ensemble_timeseries(input_dir="results/ensemble", output=None):
    """
    Plot magnitude time series for all ensemble members

    Parameters:
    -----------
    input_dir : str
        Directory containing ensemble HDF5 files
    output : str, optional
        Output filename (default: input_dir/ensemble_comparison.png)
    """
    input_path = Path(input_dir)

    # Find all ensemble result files
    h5_files = sorted(input_path.glob("ensemble_run_*.h5"))

    if len(h5_files) == 0:
        print(f"No ensemble files found in {input_path}")
        print("Run 'python run_ensemble.py' first to generate results.")
        return

    print(f"Found {len(h5_files)} ensemble members:")
    for f in h5_files:
        print(f"  {f.name}")

    n_runs = len(h5_files)

    # Create figure with subplots
    fig, axes = plt.subplots(n_runs, 1, figsize=(8, 2 * n_runs), sharex=True)

    # Handle single subplot case
    if n_runs == 1:
        axes = [axes]

    # Plot each ensemble member
    for i, (h5_file, ax) in enumerate(zip(h5_files, axes)):
        print(f"Loading {h5_file.name}...")
        results = load_lazy_results(h5_file)
        config = results["config"]
        event_history = results["event_history"]

        # Extract event data
        times = [e["time"] for e in event_history]
        magnitudes = [e["magnitude"] for e in event_history]

        # Get pulse parameters from config
        pulse_x = config.moment_pulses[0]["center_x_km"] if config.moment_pulses else 0
        pulse_amp = config.moment_pulses[0]["amplitude_mm_yr"] if config.moment_pulses else 0

        # Plot vertical lines for M>=6 events (stem plot style)
        for j in range(len(times)):
            if magnitudes[j] >= 6.0:
                ax.plot(
                    [times[j], times[j]],
                    [5.0, magnitudes[j]],
                    "-k",
                    linewidth=0.25,
                    zorder=1,
                )

        # Scatter plot of all events
        scatter = ax.scatter(
            times,
            magnitudes,
            c=magnitudes,
            cmap="plasma",
            s=1e-8 * np.array(magnitudes) ** 12.0,
            alpha=1.0,
            edgecolors="black",
            linewidth=0.25,
            zorder=10,
            vmin=5,
            vmax=8,
        )

        # Formatting
        ax.set_ylabel("$M$", fontsize=FONTSIZE)
        ax.set_xlim(0, config.duration_years)
        ax.set_ylim([5, 8])
        ax.set_yticks([5, 6, 7, 8])
        ax.tick_params(axis="both", labelsize=FONTSIZE)

        # Label with varying parameters - detect from filename
        label_parts = []
        if "seed" in h5_file.name:
            label_parts.append(f"seed = {config.random_seed}")
        if "amp" in h5_file.name:
            label_parts.append(f"$A$ = {pulse_amp:.0f} mm/yr")
        if "_x" in h5_file.name and "seed" not in h5_file.name.split("_x")[0][-4:]:
            # Check for x variation (but not in seed number)
            label_parts.append(f"$x$ = {pulse_x:.0f} km")

        # If no variation detected, show seed
        if not label_parts:
            label_parts.append(f"seed = {config.random_seed}")

        label_text = ", ".join(label_parts)

        ax.text(
            0.02,
            0.95,
            label_text,
            transform=ax.transAxes,
            fontsize=FONTSIZE,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Event count
        n_events = len(event_history)
        n_large = sum(1 for m in magnitudes if m >= 6.0)
        ax.text(
            0.98,
            0.95,
            f"N = {n_events} ({n_large} M$\\geq$6)",
            transform=ax.transAxes,
            fontsize=FONTSIZE - 1,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # X-axis label on bottom subplot only
    axes[-1].set_xlabel("$t$ (years)", fontsize=FONTSIZE)

    # Title
    fig.suptitle(
        "Ensemble comparison: Earthquake sequences with varying pulse locations",
        fontsize=FONTSIZE + 2,
        y=1.01,
    )

    plt.tight_layout()

    # Save
    if output is None:
        output = input_path / "ensemble_comparison.png"
    else:
        output = Path(output)

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output}")

    return fig


def plot_ensemble_summary(input_dir="results/ensemble", output=None):
    """
    Plot summary statistics comparing ensemble members

    Parameters:
    -----------
    input_dir : str
        Directory containing ensemble HDF5 files
    output : str, optional
        Output filename (default: input_dir/ensemble_summary.png)
    """
    input_path = Path(input_dir)

    # Find all ensemble result files
    h5_files = sorted(input_path.glob("ensemble_run_*.h5"))

    if len(h5_files) == 0:
        print(f"No ensemble files found in {input_path}")
        return

    # Collect statistics
    pulse_locations = []
    n_events_list = []
    n_large_list = []
    max_mag_list = []
    total_moment_list = []

    for h5_file in h5_files:
        results = load_lazy_results(h5_file)
        config = results["config"]
        event_history = results["event_history"]

        pulse_x = config.moment_pulses[0]["center_x_km"] if config.moment_pulses else 0
        pulse_locations.append(pulse_x)

        magnitudes = [e["magnitude"] for e in event_history]
        moments = [e["geom_moment"] for e in event_history]

        n_events_list.append(len(event_history))
        n_large_list.append(sum(1 for m in magnitudes if m >= 6.0))
        max_mag_list.append(max(magnitudes) if magnitudes else 0)
        total_moment_list.append(sum(moments))

    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Total events vs pulse location
    axes[0, 0].bar(pulse_locations, n_events_list, width=3, color="steelblue")
    axes[0, 0].set_xlabel("Pulse location (km)", fontsize=FONTSIZE)
    axes[0, 0].set_ylabel("Total events", fontsize=FONTSIZE)
    axes[0, 0].tick_params(axis="both", labelsize=FONTSIZE)

    # Large events vs pulse location
    axes[0, 1].bar(pulse_locations, n_large_list, width=3, color="coral")
    axes[0, 1].set_xlabel("Pulse location (km)", fontsize=FONTSIZE)
    axes[0, 1].set_ylabel("Events M$\\geq$6", fontsize=FONTSIZE)
    axes[0, 1].tick_params(axis="both", labelsize=FONTSIZE)

    # Max magnitude vs pulse location
    axes[1, 0].bar(pulse_locations, max_mag_list, width=3, color="purple")
    axes[1, 0].set_xlabel("Pulse location (km)", fontsize=FONTSIZE)
    axes[1, 0].set_ylabel("Maximum magnitude", fontsize=FONTSIZE)
    axes[1, 0].set_ylim([6.5, 8])
    axes[1, 0].tick_params(axis="both", labelsize=FONTSIZE)

    # Total moment vs pulse location
    moment_scale = 1e-9
    axes[1, 1].bar(
        pulse_locations, np.array(total_moment_list) * moment_scale, width=3, color="green"
    )
    axes[1, 1].set_xlabel("Pulse location (km)", fontsize=FONTSIZE)
    axes[1, 1].set_ylabel(f"Total moment ($\\times 10^9$ m$^3$)", fontsize=FONTSIZE)
    axes[1, 1].tick_params(axis="both", labelsize=FONTSIZE)

    plt.tight_layout()

    # Save
    if output is None:
        output = input_path / "ensemble_summary.png"
    else:
        output = Path(output)

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")

    return fig


def plot_ensemble_loading_and_earthquakes(input_dir="results/ensemble", output=None):
    """
    Plot loading rate and earthquake locations for all ensemble members

    Creates a multi-panel figure with one row per ensemble member,
    showing the spatial loading rate (slip rate) as filled contours
    with earthquake hypocenters overlaid as circles.

    Parameters:
    -----------
    input_dir : str
        Directory containing ensemble HDF5 files
    output : str, optional
        Output filename (default: input_dir/ensemble_loading.png)
    """
    input_path = Path(input_dir)

    # Find all ensemble result files
    h5_files = sorted(input_path.glob("ensemble_run_*.h5"))

    if len(h5_files) == 0:
        print(f"No ensemble files found in {input_path}")
        print("Run 'python run_ensemble.py' first to generate results.")
        return

    print(f"Creating loading/earthquake plot for {len(h5_files)} ensemble members...")

    n_runs = len(h5_files)

    # Create figure with subplots - one row per ensemble member
    fig, axes = plt.subplots(n_runs, 1, figsize=(10, 2 * n_runs))

    # Handle single subplot case
    if n_runs == 1:
        axes = [axes]

    # We need consistent colorbar limits across all panels
    # First pass: find global min/max of slip rate
    all_slip_rates = []
    for h5_file in h5_files:
        results = load_lazy_results(h5_file)
        slip_rate = results["slip_rate"]
        all_slip_rates.append(slip_rate * 1000)  # Convert to mm/yr

    vmin = min(sr.min() for sr in all_slip_rates)
    vmax = max(sr.max() for sr in all_slip_rates)

    # Plot each ensemble member
    for i, (h5_file, ax) in enumerate(zip(h5_files, axes)):
        results = load_lazy_results(h5_file)
        config = results["config"]
        event_history = results["event_history"]
        mesh = results["mesh"]
        slip_rate = results["slip_rate"]  # m/year per element

        # Get pulse parameters
        pulse_x = config.moment_pulses[0]["center_x_km"] if config.moment_pulses else 0
        pulse_amp = config.moment_pulses[0]["amplitude_mm_yr"] if config.moment_pulses else 0

        # Create grids for contourf plotting
        length_vec = np.linspace(0, config.fault_length_km, config.n_along_strike)
        depth_vec = np.linspace(0, config.fault_depth_km, config.n_down_dip)
        length_grid, depth_grid = np.meshgrid(length_vec, depth_vec)

        # Reshape slip_rate to 2D grid and convert to mm/yr
        slip_rate_grid = slip_rate.reshape(mesh["n_along_strike"], mesh["n_down_dip"])
        slip_rate_mm_yr = slip_rate_grid.T * 1000

        # Plot loading rate as filled contours
        levels = np.linspace(vmin, vmax, 20)
        cf = ax.contourf(
            length_grid,
            depth_grid,
            slip_rate_mm_yr,
            cmap="YlOrRd",
            levels=levels,
            extend="both",
        )

        # Overlay earthquakes as circles
        if len(event_history) > 0:
            x_coords = np.array([e["hypocenter_x_km"] for e in event_history])
            z_coords = np.array([e["hypocenter_z_km"] for e in event_history])
            geom_moments = np.array([e["geom_moment"] for e in event_history])

            # Scale circle sizes: area proportional to moment
            moment_ref = 1e6
            size_ref = 20
            sizes = size_ref * (geom_moments / moment_ref)
            sizes *= 0.01

            ax.scatter(
                x_coords,
                z_coords,
                s=sizes,
                c="gray",
                alpha=0.4,
                edgecolors="black",
                linewidths=0.25,
                zorder=10,
            )

        # Formatting
        ax.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
        ax.invert_yaxis()
        ax.set_yticks([0, 25])
        ax.tick_params(axis="both", labelsize=FONTSIZE)
        ax.set_aspect("equal", adjustable="box")

        # Remove x-axis labels except for bottom panel
        if i < n_runs - 1:
            ax.set_xticklabels([])

        # Label with varying parameters
        label_parts = []
        if "seed" in h5_file.name:
            label_parts.append(f"seed={config.random_seed}")
        if "amp" in h5_file.name:
            label_parts.append(f"$A$={pulse_amp:.0f}")
        if "_x" in h5_file.name and "seed" not in h5_file.name.split("_x")[0][-4:]:
            label_parts.append(f"$x$={pulse_x:.0f}")
        if not label_parts:
            label_parts.append(f"seed={config.random_seed}")

        label_text = ", ".join(label_parts)

        ax.text(
            0.02,
            0.85,
            label_text,
            transform=ax.transAxes,
            fontsize=FONTSIZE,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Event count
        n_events = len(event_history)
        n_large = sum(1 for e in event_history if e["magnitude"] >= 6.0)
        ax.text(
            0.98,
            0.85,
            f"N={n_events} ({n_large} M$\\geq$6)",
            transform=ax.transAxes,
            fontsize=FONTSIZE - 1,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # X-axis label on bottom subplot only
    axes[-1].set_xlabel("$x$ (km)", fontsize=FONTSIZE)

    # Add shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label("$\\dot{s}_\\mathrm{d}$ (mm/yr)", fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE - 2)

    # Save
    if output is None:
        output = input_path / "ensemble_loading.png"
    else:
        output = Path(output)

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ensemble simulation results"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/ensemble",
        help="Directory containing ensemble HDF5 files (default: results/ensemble)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: input_dir/ensemble_comparison.png)",
    )

    args = parser.parse_args()

    # Generate time series comparison
    plot_ensemble_timeseries(input_dir=args.input_dir, output=args.output)

    # Generate summary statistics
    plot_ensemble_summary(input_dir=args.input_dir)

    # Generate loading rate and earthquake locations
    plot_ensemble_loading_and_earthquakes(input_dir=args.input_dir)


if __name__ == "__main__":
    main()
