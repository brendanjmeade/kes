"""
Visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def plot_magnitude_time_series(results, config):
    """
    Plot magnitude vs time
    """
    event_history = results["event_history"]

    if len(event_history) == 0:
        print("No events to plot")
        return

    times = [e["time"] for e in event_history]
    magnitudes = [e["magnitude"] for e in event_history]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot events
    ax.scatter(
        times,
        magnitudes,
        c=magnitudes,
        cmap="YlOrRd",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Time (years)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title(
        f"Earthquake Magnitude Time Series ({len(event_history)} events)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, config.duration_years)

    # Color bar
    sm = plt.cm.ScalarMappable(
        cmap="YlOrRd", norm=plt.Normalize(vmin=min(magnitudes), vmax=max(magnitudes))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Magnitude", fontsize=11)

    plt.tight_layout()

    # Save
    output_path = Path(config.output_dir) / "magnitude_time_series.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    return fig


def plot_moment_snapshots(results, config, times_to_plot=None):
    """
    Plot moment distribution at multiple time slices
    """
    moment_snapshots = results["moment_snapshots"]
    snapshot_times = results["snapshot_times"]
    mesh = results["mesh"]

    if times_to_plot is None:
        # Default: plot 6 evenly spaced times
        indices = np.linspace(0, len(snapshot_times) - 1, 6, dtype=int)
        times_to_plot = [snapshot_times[i] for i in indices]

    n_plots = len(times_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(times_to_plot):
        # Find closest snapshot
        idx = np.argmin(np.abs(np.array(snapshot_times) - t))
        m_snapshot = moment_snapshots[idx]
        actual_time = snapshot_times[idx]

        # Reshape to 2D grid
        m_grid = m_snapshot.reshape(mesh["n_along_strike"], mesh["n_down_dip"])

        # Plot
        ax = axes[i]
        im = ax.imshow(
            m_grid.T,
            origin="lower",
            aspect="auto",
            extent=[0, config.fault_length_km, 0, config.fault_depth_km],
            cmap="viridis",
        )

        ax.set_xlabel("Along-strike (km)")
        ax.set_ylabel("Depth (km)")
        ax.set_title(f"t = {actual_time:.1f} years")

        plt.colorbar(im, ax=ax, label="Moment")

    # Remove extra subplots if any
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(
        "Cumulative Moment Distribution Through Time",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    # Save
    output_path = Path(config.output_dir) / "moment_snapshots.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    return fig


def plot_cumulative_slip_map(results, config):
    """
    Plot cumulative slip from all events
    """
    event_history = results["event_history"]
    mesh = results["mesh"]

    # Sum all slip
    cumulative_slip = np.zeros(config.n_elements)
    for event in event_history:
        cumulative_slip += event["slip"]

    # Reshape to 2D
    slip_grid = cumulative_slip.reshape(mesh["n_along_strike"], mesh["n_down_dip"])

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        slip_grid.T,
        origin="lower",
        aspect="auto",
        extent=[0, config.fault_length_km, 0, config.fault_depth_km],
        cmap="hot",
    )

    ax.set_xlabel("Along-strike distance (km)", fontsize=12)
    ax.set_ylabel("Depth (km)", fontsize=12)
    ax.set_title("Cumulative Coseismic Slip", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Slip (m)", fontsize=11)

    # Mark hypocenters
    hypo_x = [e["hypocenter_x_km"] for e in event_history]
    hypo_z = [e["hypocenter_z_km"] for e in event_history]
    ax.scatter(
        hypo_x,
        hypo_z,
        c="cyan",
        s=30,
        marker="*",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.7,
        label="Hypocenters",
    )

    ax.legend()
    plt.tight_layout()

    # Save
    output_path = Path(config.output_dir) / "cumulative_slip.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    return fig


def create_moment_animation(results, config):
    """
    Create animation of moment evolution

    This will create an MP4 file showing moment evolution through time
    """
    moment_snapshots = results["moment_snapshots"]
    snapshot_times = results["snapshot_times"]
    mesh = results["mesh"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Initial plot
    m_grid = moment_snapshots[0].reshape(mesh["n_along_strike"], mesh["n_down_dip"])

    # Determine color scale from all snapshots
    all_moments = np.concatenate([m.flatten() for m in moment_snapshots])
    vmin, vmax = np.percentile(all_moments, [1, 99])

    im = ax.imshow(
        m_grid.T,
        origin="lower",
        aspect="auto",
        extent=[0, config.fault_length_km, 0, config.fault_depth_km],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("Along-strike distance (km)", fontsize=12)
    ax.set_ylabel("Depth (km)", fontsize=12)
    title = ax.set_title(
        f"Moment Distribution: t = 0.0 years", fontsize=14, fontweight="bold"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Geometric Moment", fontsize=11)

    def update(frame):
        m_grid = moment_snapshots[frame].reshape(
            mesh["n_along_strike"], mesh["n_down_dip"]
        )
        im.set_array(m_grid.T)
        title.set_text(f"Moment Distribution: t = {snapshot_times[frame]:.1f} years")
        return [im, title]

    anim = FuncAnimation(
        fig, update, frames=len(moment_snapshots), interval=100, blit=True
    )

    # Save
    output_path = Path(config.output_dir) / "moment_evolution.mp4"
    anim.save(output_path, writer="ffmpeg", fps=10, dpi=100)
    print(f"Saved animation: {output_path}")

    plt.close()

    return output_path


def plot_all_diagnostics(results, config):
    """
    Generate all diagnostic plots
    """
    print("\nGenerating plots...")

    plot_magnitude_time_series(results, config)
    plot_moment_snapshots(results, config)
    plot_cumulative_slip_map(results, config)

    # Animation (optional, can take time)
    # create_moment_animation(results, config)

    print("\nAll plots generated!")
