"""
Visualization script for ensemble simulation results

Generates multi-panel figures showing earthquake magnitude time series
for each ensemble member, allowing comparison of how earthquake sequences
evolve under different loading conditions.

Outputs:
    ensemble_comparison.png   - Magnitude time series for each run
    ensemble_summary.png      - Summary statistics (event counts, max mag, total moment)
    ensemble_loading.png      - Slip rate with all earthquake hypocenters
    ensemble_loading_events.png - Slip rate with first N large events labeled
    ensemble_loading_events_evolution.png - Event location shifts vs pulse location change

Usage:
    python plot_ensemble.py [--input_dir DIR] [--output OUTPUT]
                           [--n_events N] [--m_threshold M]

Examples:
    # Default: first 5 M>=6 events
    python plot_ensemble.py

    # Label first 10 M>=7 events
    python plot_ensemble.py --n_events 10 --m_threshold 7.0

Defaults:
    input_dir = results/ensemble
    n_events = 5
    m_threshold = 6.0
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
        pulse_amp = (
            config.moment_pulses[0]["amplitude_mm_yr"] if config.moment_pulses else 0
        )

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
            bbox=dict(boxstyle="square", facecolor="white", alpha=1.0),
            zorder=30,
        )

        # Event count
        n_events = len(event_history)
        n_large = sum(1 for m in magnitudes if m >= 6.0)
        ax.text(
            0.98,
            0.95,
            f"$n$ = {n_events} ({n_large} $M\\geq$6)",
            transform=ax.transAxes,
            fontsize=FONTSIZE - 1,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="square", facecolor="white", alpha=1.0),
            zorder=30,
        )

    # X-axis label on bottom subplot only
    axes[-1].set_xlabel("$t$ (years)", fontsize=FONTSIZE)

    # Title
    # fig.suptitle(
    #     "Ensemble comparison: Earthquake sequences with varying pulse locations",
    #     fontsize=FONTSIZE + 2,
    #     y=1.01,
    # )

    plt.tight_layout()

    # Save
    if output is None:
        output = input_path / "ensemble_comparison.png"
    else:
        output = Path(output)

    plt.savefig(output, dpi=500, bbox_inches="tight")
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
        pulse_locations,
        np.array(total_moment_list) * moment_scale,
        width=3,
        color="green",
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
        pulse_amp = (
            config.moment_pulses[0]["amplitude_mm_yr"] if config.moment_pulses else 0
        )

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


def plot_ensemble_loading_events(
    input_dir="results/ensemble",
    output=None,
    n_events=5,
    m_threshold=6.0,
):
    """
    Plot loading rate and earthquake locations with labeled first N large events

    Creates a multi-panel figure with one row per ensemble member,
    showing the spatial loading rate (slip rate) as filled contours
    with earthquake hypocenters overlaid as circles. The first N events
    with M >= m_threshold are labeled with their order (1, 2, 3, ...) and
    the accumulated moment at the hypocenter location at the time of rupture.

    Parameters:
    -----------
    input_dir : str
        Directory containing ensemble HDF5 files
    output : str, optional
        Output filename (default: input_dir/ensemble_loading_events.png)
    n_events : int
        Number of large events to label (default: 5)
    m_threshold : float
        Minimum magnitude threshold for labeled events (default: 6.0)
    """
    input_path = Path(input_dir)

    # Find all ensemble result files
    h5_files = sorted(input_path.glob("ensemble_run_*.h5"))

    if len(h5_files) == 0:
        print(f"No ensemble files found in {input_path}")
        print("Run 'python run_ensemble.py' first to generate results.")
        return

    print(f"Creating loading/events plot for {len(h5_files)} ensemble members...")
    print(f"  Labeling first {n_events} events with M >= {m_threshold}")

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

        # Get snapshot data for moment lookup
        moment_snapshots = results["moment_snapshots"]
        snapshot_times = results["times"][:]

        # Get pulse parameters
        pulse_x = config.moment_pulses[0]["center_x_km"] if config.moment_pulses else 0
        pulse_amp = (
            config.moment_pulses[0]["amplitude_mm_yr"] if config.moment_pulses else 0
        )

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

        # Overlay all earthquakes as circles (gray, background)
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

        # Find first N events with M >= m_threshold
        large_events = [
            (idx, e)
            for idx, e in enumerate(event_history)
            if e["magnitude"] >= m_threshold
        ][:n_events]

        # Debug: print what loading_events is seeing
        print(f"  Panel {i} ({h5_file.name}):")
        for order, (idx, e) in enumerate(large_events, start=1):
            print(
                f"    Event {order}: x={e['hypocenter_x_km']:.1f} km, M={e['magnitude']:.1f}"
            )

        # Label each large event with order number and moment at hypocenter
        for order, (event_idx, event) in enumerate(large_events, start=1):
            event_time = event["time"]
            hypo_x = event["hypocenter_x_km"]
            hypo_z = event["hypocenter_z_km"]
            hypo_idx = event["hypocenter_idx"]
            magnitude = event["magnitude"]

            # Find snapshot closest to (but <= ) event time
            snapshot_idx = np.searchsorted(snapshot_times, event_time, side="right") - 1
            snapshot_idx = max(0, snapshot_idx)  # Clamp to valid range

            # Get moment at hypocenter location at this time
            moment_at_hypo = moment_snapshots[snapshot_idx, hypo_idx]

            # Highlight this event with a larger, colored marker
            ax.scatter(
                hypo_x,
                hypo_z,
                s=150,
                c="black",
                edgecolors="white",
                linewidths=0.5,
                zorder=20,
                marker="o",
            )

            # Add text label with event order number
            ax.text(
                hypo_x,
                hypo_z,
                str(order),
                fontsize=FONTSIZE - 1,
                # fontweight="bold",
                color="white",
                ha="center",
                va="center",
                zorder=25,
            )

            # Add annotation with moment and magnitude info
            # Position annotation slightly offset from the marker
            offset_x = 5 if hypo_x < config.fault_length_km / 2 else -5
            ha = "left" if offset_x > 0 else "right"

        #     label = f"M{magnitude:.1f}\n$m$={moment_at_hypo:.2f} m"
        # ax.annotate(
        #     label,
        #     xy=(hypo_x, hypo_z),
        #     xytext=(hypo_x + offset_x, hypo_z),
        #     fontsize=FONTSIZE - 2,
        #     ha=ha,
        #     va="center",
        #     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="blue"),
        #     arrowprops=dict(arrowstyle="-", color="blue", lw=0.5),
        #     zorder=30,
        # )

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
            label_parts.append(f"$x_\mathrm{{p}}$={pulse_x:.0f}")
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
            bbox=dict(boxstyle="square", facecolor="white", alpha=1.0),
            zorder=30,
        )

        # Event count
        n_total = len(event_history)
        n_large = sum(1 for e in event_history if e["magnitude"] >= m_threshold)
        ax.text(
            0.98,
            0.85,
            f"$n$={n_total}, {n_large} $M\\geq${m_threshold:.0f}",
            transform=ax.transAxes,
            fontsize=FONTSIZE - 1,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="square", facecolor="white", alpha=1.0),
            zorder=30,
        )

    # X-axis label on bottom subplot only
    axes[-1].set_xlabel("$x$ (km)", fontsize=FONTSIZE)

    # Add shared colorbar
    fig.subplots_adjust(right=0.88)
    # cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar_ax = fig.add_axes([0.90, 0.125, 0.01, 0.1])

    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label("$\\dot{s}_\\mathrm{d}$ (mm/yr)", fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE - 2)

    # # Add title
    # fig.suptitle(
    #     f"First {n_events} M$\\geq${m_threshold:.0f} events with accumulated moment at hypocenter",
    #     fontsize=FONTSIZE + 1,
    #     y=1.01,
    # )

    # Save
    if output is None:
        output = input_path / "ensemble_loading_events.png"
    else:
        output = Path(output)

    plt.savefig(output, dpi=500, bbox_inches="tight")
    print(f"Saved: {output}")

    return fig


def plot_ensemble_loading_events_evolution(
    input_dir="results/ensemble",
    output=None,
    n_events=5,
    m_threshold=6.0,
):
    """
    Plot evolution of large event locations as pulse location changes

    Creates a figure showing how the hypocenter positions of the first N
    large events change relative to the first simulation as the Gaussian
    pulse location varies. Shows x (along-strike), z (down-dip), magnitude,
    and moment at hypocenter.

    Parameters:
    -----------
    input_dir : str
        Directory containing ensemble HDF5 files
    output : str, optional
        Output filename (default: input_dir/ensemble_loading_events_evolution.png)
    n_events : int
        Number of large events to track (default: 5)
    m_threshold : float
        Minimum magnitude threshold for tracked events (default: 6.0)
    """
    input_path = Path(input_dir)

    # Find all ensemble result files
    h5_files = sorted(input_path.glob("ensemble_run_*.h5"))

    if len(h5_files) == 0:
        print(f"No ensemble files found in {input_path}")
        print("Run 'python run_ensemble.py' first to generate results.")
        return

    print(f"Creating event evolution plot for {len(h5_files)} ensemble members...")
    print(f"  Tracking first {n_events} events with M >= {m_threshold}")

    n_runs = len(h5_files)

    # Collect data from all runs - read DIRECTLY from HDF5 files
    import h5py
    import json

    pulse_locations = []
    event_x_locations = {
        i: [] for i in range(n_events)
    }  # event order -> list of x-locations
    event_z_locations = {
        i: [] for i in range(n_events)
    }  # event order -> list of z-locations
    event_magnitudes = {
        i: [] for i in range(n_events)
    }  # event order -> list of magnitudes
    event_moments = {
        i: [] for i in range(n_events)
    }  # event order -> list of moments at hypocenter

    for run_idx, h5_file in enumerate(h5_files):
        # Read directly from HDF5 file
        with h5py.File(h5_file, "r") as f:
            # Get pulse x-location from config
            moment_pulses_str = f["config"].attrs.get("moment_pulses", "[]")
            if isinstance(moment_pulses_str, str):
                moment_pulses = json.loads(moment_pulses_str)
            else:
                moment_pulses = []
            pulse_x = moment_pulses[0]["center_x_km"] if moment_pulses else 0
            pulse_locations.append(pulse_x)

            # Read events directly from the structured array
            events = f["events"][:]

            # Read moment snapshots and times for moment lookup
            moment_snapshots = f["moment_snapshots"][:]
            snapshot_times = f["times"][:]

            # Filter for large events
            large_mask = events["magnitude"] >= m_threshold
            large_events_raw = events[large_mask][:n_events]

            print(
                f"  Run {run_idx} ({h5_file.name}): pulse_x={pulse_x:.1f}, found {len(large_events_raw)} large events"
            )
            for i, e in enumerate(large_events_raw):
                # Find moment at hypocenter at time of event
                event_time = e["time"]
                hypo_idx = e["hypocenter_idx"]
                snapshot_idx = (
                    np.searchsorted(snapshot_times, event_time, side="right") - 1
                )
                snapshot_idx = max(0, snapshot_idx)
                moment_at_hypo = moment_snapshots[snapshot_idx, hypo_idx]

                print(
                    f"    Event {i + 1}: x={e['hypocenter_x_km']:.2f} km, z={e['hypocenter_z_km']:.2f} km, M={e['magnitude']:.2f}, m={moment_at_hypo:.2f}"
                )

            # Store data for each event order
            for i in range(n_events):
                if i < len(large_events_raw):
                    e = large_events_raw[i]
                    event_x_locations[i].append(float(e["hypocenter_x_km"]))
                    event_z_locations[i].append(float(e["hypocenter_z_km"]))
                    event_magnitudes[i].append(float(e["magnitude"]))

                    # Get moment at hypocenter
                    event_time = e["time"]
                    hypo_idx = e["hypocenter_idx"]
                    snapshot_idx = (
                        np.searchsorted(snapshot_times, event_time, side="right") - 1
                    )
                    snapshot_idx = max(0, snapshot_idx)
                    moment_at_hypo = moment_snapshots[snapshot_idx, hypo_idx]
                    event_moments[i].append(float(moment_at_hypo))
                else:
                    event_x_locations[i].append(np.nan)
                    event_z_locations[i].append(np.nan)
                    event_magnitudes[i].append(np.nan)
                    event_moments[i].append(np.nan)

    # Convert to arrays
    pulse_locations = np.array(pulse_locations)
    for i in range(n_events):
        event_x_locations[i] = np.array(event_x_locations[i])
        event_z_locations[i] = np.array(event_z_locations[i])
        event_magnitudes[i] = np.array(event_magnitudes[i])
        event_moments[i] = np.array(event_moments[i])

    # Calculate changes relative to first run
    pulse_delta = pulse_locations - pulse_locations[0]

    # Reference locations from first run (for computing deltas)
    reference_x = {i: event_x_locations[i][0] for i in range(n_events)}
    reference_z = {i: event_z_locations[i][0] for i in range(n_events)}

    # Debug: print summary
    print(f"\n  Pulse locations: {pulse_locations}")
    print(f"  Pulse deltas: {pulse_delta}")
    for i in range(min(n_events, 4)):  # Only print first 4 events
        if not np.all(np.isnan(event_x_locations[i])):
            print(f"  Event {i + 1} x-locations: {event_x_locations[i]}")
            print(f"  Event {i + 1} z-locations: {event_z_locations[i]}")
            print(f"  Event {i + 1} Δx: {event_x_locations[i] - reference_x[i]}")
            print(f"  Event {i + 1} Δz: {event_z_locations[i] - reference_z[i]}")
            print(f"  Event {i + 1} magnitudes: {event_magnitudes[i]}")
            print(f"  Event {i + 1} moments: {event_moments[i]}")

    # Create figure with four subplots in a single column
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # Color map for different events
    colors = plt.cm.tab10(np.linspace(0, 1, n_events))

    # Panel 1: x-coordinate changes
    ax_x = axes[0]
    for i in range(n_events):
        if np.isnan(reference_x[i]):
            continue
        event_delta = event_x_locations[i] - reference_x[i]
        valid = ~np.isnan(event_delta)
        if not valid.any():
            continue
        ax_x.plot(
            pulse_delta[valid],
            event_delta[valid],
            "-o",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=f"Event {i + 1}",
        )

    ax_x.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_x.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_x.set_ylabel("$\\Delta x_\\mathrm{hypocenter}$ (km)", fontsize=FONTSIZE + 2)
    ax_x.tick_params(axis="both", labelsize=FONTSIZE)
    ax_x.legend(fontsize=FONTSIZE, loc="best")
    ax_x.set_title("Along-strike position change", fontsize=FONTSIZE + 2)
    ax_x.grid(True, alpha=0.3)

    # Panel 2: z-coordinate (depth) changes
    ax_z = axes[1]
    for i in range(n_events):
        if np.isnan(reference_z[i]):
            continue
        event_delta = event_z_locations[i] - reference_z[i]
        valid = ~np.isnan(event_delta)
        if not valid.any():
            continue
        ax_z.plot(
            pulse_delta[valid],
            event_delta[valid],
            "-o",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=f"Event {i + 1}",
        )

    ax_z.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_z.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_z.set_ylabel("$\\Delta z_\\mathrm{hypocenter}$ (km)", fontsize=FONTSIZE + 2)
    ax_z.tick_params(axis="both", labelsize=FONTSIZE)
    ax_z.legend(fontsize=FONTSIZE, loc="best")
    ax_z.set_title("Down-dip position change", fontsize=FONTSIZE + 2)
    ax_z.grid(True, alpha=0.3)

    # Panel 3: Magnitude
    ax_m = axes[2]
    for i in range(n_events):
        valid = ~np.isnan(event_magnitudes[i])
        if not valid.any():
            continue
        ax_m.plot(
            pulse_delta[valid],
            event_magnitudes[i][valid],
            "-o",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=f"Event {i + 1}",
        )

    ax_m.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_m.set_ylabel("$M$", fontsize=FONTSIZE + 2)
    ax_m.tick_params(axis="both", labelsize=FONTSIZE)
    ax_m.legend(fontsize=FONTSIZE, loc="best")
    ax_m.set_title("Event magnitude", fontsize=FONTSIZE + 2)
    ax_m.grid(True, alpha=0.3)

    # Panel 4: Moment at hypocenter
    ax_mom = axes[3]
    for i in range(n_events):
        valid = ~np.isnan(event_moments[i])
        if not valid.any():
            continue
        ax_mom.plot(
            pulse_delta[valid],
            event_moments[i][valid],
            "-o",
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=f"Event {i + 1}",
        )

    ax_mom.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_mom.set_xlabel("$\\Delta x_\\mathrm{pulse}$ (km)", fontsize=FONTSIZE + 2)
    ax_mom.set_ylabel("$m$ at hypocenter (m)", fontsize=FONTSIZE + 2)
    ax_mom.tick_params(axis="both", labelsize=FONTSIZE)
    ax_mom.legend(fontsize=FONTSIZE, loc="best")
    ax_mom.set_title("Accumulated moment at hypocenter", fontsize=FONTSIZE + 2)
    ax_mom.grid(True, alpha=0.3)

    # Main title
    fig.suptitle(
        f"First {n_events} M$\\geq${m_threshold:.0f} events vs pulse location shift",
        fontsize=FONTSIZE + 3,
        y=0.995,
    )

    plt.tight_layout()

    # Save
    if output is None:
        output = input_path / "ensemble_loading_events_evolution.png"
    else:
        output = Path(output)

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")

    # Print summary statistics
    print("\nEvent location changes (km) relative to first run:")
    print(f"  Pulse location: {pulse_locations[0]:.1f} km (reference)")
    print(f"  Pulse delta range: {pulse_delta.min():.1f} → {pulse_delta.max():.1f} km")
    for i in range(n_events):
        if not np.isnan(reference_x[i]) and not np.isnan(reference_z[i]):
            delta_x = event_x_locations[i] - reference_x[i]
            delta_z = event_z_locations[i] - reference_z[i]
            valid = ~np.isnan(delta_x) & ~np.isnan(delta_z)
            if valid.any():
                print(
                    f"  Event {i + 1}: Δx = {delta_x[valid].min():.2f} → {delta_x[valid].max():.2f} km, "
                    f"Δz = {delta_z[valid].min():.2f} → {delta_z[valid].max():.2f} km "
                    f"(ref: x={reference_x[i]:.1f}, z={reference_z[i]:.1f} km)"
                )

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
    parser.add_argument(
        "--n_events",
        type=int,
        default=5,
        help="Number of large events to label in loading_events plot (default: 5)",
    )
    parser.add_argument(
        "--m_threshold",
        type=float,
        default=6.0,
        help="Minimum magnitude for labeled events in loading_events plot (default: 6.0)",
    )

    args = parser.parse_args()

    # Generate time series comparison
    plot_ensemble_timeseries(input_dir=args.input_dir, output=args.output)

    # Generate summary statistics
    plot_ensemble_summary(input_dir=args.input_dir)

    # Generate loading rate and earthquake locations
    plot_ensemble_loading_and_earthquakes(input_dir=args.input_dir)

    # Generate loading rate with labeled first N large events
    plot_ensemble_loading_events(
        input_dir=args.input_dir,
        n_events=args.n_events,
        m_threshold=args.m_threshold,
    )

    # Generate evolution plot showing how event locations change with pulse location
    plot_ensemble_loading_events_evolution(
        input_dir=args.input_dir,
        n_events=args.n_events,
        m_threshold=args.m_threshold,
    )


if __name__ == "__main__":
    main()
