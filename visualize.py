"""
Visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def plot_moment_history(results, config):
    """
    Plot total accumulated moment through time

    Shows loading, earthquakes, and recovery
    """
    moment_snapshots = results["moment_snapshots"]
    snapshot_times = results["snapshot_times"]
    event_history = results["event_history"]

    # Calculate total moment at each snapshot
    total_moments = [np.sum(m) for m in moment_snapshots]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top panel: Total accumulated moment
    ax1.plot(snapshot_times, total_moments, "b-", linewidth=2, label="Total Moment")

    # Mark earthquakes as vertical lines
    for event in event_history:
        # Color by magnitude
        if event["magnitude"] >= 7.0:
            color = "red"
            alpha = 0.7
            linewidth = 2
        elif event["magnitude"] >= 6.0:
            color = "orange"
            alpha = 0.5
            linewidth = 1.5
        else:
            color = "gray"
            alpha = 0.3
            linewidth = 1

        ax1.axvline(
            event["time"], color=color, alpha=alpha, linewidth=linewidth, linestyle="--"
        )

    ax1.set_ylabel("Total Geometric Moment (m³)", fontsize=12)
    ax1.set_title(
        "Moment Accumulation and Release Through Time", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Add legend for earthquake colors
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", linewidth=2, linestyle="--", label="M ≥ 7.0"),
        Line2D(
            [0],
            [0],
            color="orange",
            linewidth=1.5,
            linestyle="--",
            label="6.0 ≤ M < 7.0",
        ),
        Line2D([0], [0], color="gray", linewidth=1, linestyle="--", label="M < 6.0"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=10)

    # Bottom panel: Rate of change (moment rate)
    # Compute derivative to show loading rate vs. release events
    dt = np.diff(snapshot_times)
    dm = np.diff(total_moments)
    moment_rate = dm / dt  # m³/year

    ax2.plot(
        snapshot_times[:-1], moment_rate, "g-", linewidth=1.5, label="Net Moment Rate"
    )

    # Add zero line
    ax2.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)

    # Expected loading rate (from config)
    expected_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )
    ax2.axhline(
        expected_rate,
        color="b",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Expected Loading Rate",
    )

    ax2.set_xlabel("Time (years)", fontsize=12)
    ax2.set_ylabel("Moment Rate (m³/year)", fontsize=12)
    ax2.set_title("Rate of Moment Change", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()

    # Save
    output_path = Path(config.output_dir) / "moment_history.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    return fig


# def plot_moment_budget_analysis(results, config):
#     """
#     Analyze moment budget: input vs. output over time

#     Shows cumulative loading vs. cumulative release
#     """
#     event_history = results["event_history"]

#     if len(event_history) == 0:
#         print("No events for budget analysis")
#         return

#     # Time array
#     max_time = event_history[-1]["time"]
#     times = np.linspace(0, max_time, 1000)

#     # Cumulative tectonic loading
#     total_loading_rate = (
#         config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
#     )
#     cumulative_loading = times * total_loading_rate  # m³

#     # Cumulative seismic release (as geometric moment)
#     cumulative_release = np.zeros_like(times)
#     for i, t in enumerate(times):
#         # Sum all geometric moment released up to time t
#         released = sum(
#             [
#                 np.sum(e["slip"] * config.element_area_m2)
#                 for e in event_history
#                 if e["time"] <= t
#             ]
#         )
#         cumulative_release[i] = released

#     # Moment deficit = loading - release
#     moment_deficit = cumulative_loading - cumulative_release

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

#     # Top panel: Cumulative loading vs. release
#     ax1.plot(
#         times,
#         cumulative_loading,
#         "b-",
#         linewidth=2,
#         label="Cumulative Tectonic Loading",
#     )
#     ax1.plot(
#         times, cumulative_release, "r-", linewidth=2, label="Cumulative Seismic Release"
#     )

#     ax1.set_ylabel("Cumulative Moment (m³)", fontsize=12)
#     ax1.set_title("Moment Budget: Loading vs. Release", fontsize=14, fontweight="bold")
#     ax1.legend(fontsize=11)
#     ax1.grid(True, alpha=0.3)

#     # Bottom panel: Moment deficit
#     ax2.plot(times, moment_deficit, "purple", linewidth=2)
#     ax2.axhline(0, color="k", linestyle="--", linewidth=0.5)

#     ax2.set_xlabel("Time (years)", fontsize=12)
#     ax2.set_ylabel("Moment Deficit (m³)", fontsize=12)
#     ax2.set_title("Accumulated Moment Deficit", fontsize=13, fontweight="bold")
#     ax2.grid(True, alpha=0.3)

#     # Compute coupling coefficient
#     if max_time > 0:
#         total_loaded = cumulative_loading[-1]
#         total_released = cumulative_release[-1]
#         coupling_coef = total_released / total_loaded

#         ax2.text(
#             0.02,
#             0.95,
#             f"Seismic Coupling: {coupling_coef:.3f}",
#             transform=ax2.transAxes,
#             fontsize=11,
#             verticalalignment="top",
#             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
#         )

#     plt.tight_layout()

#     # Save
#     output_path = Path(config.output_dir) / "moment_budget.png"
#     plt.savefig(output_path, dpi=150, bbox_inches="tight")
#     print(f"Saved: {output_path}")

#     return fig


def plot_moment_budget(results, config):
    """
    Plot moment budget showing loading vs release

    Uses actual event times to ensure accurate tracking throughout simulation
    """
    event_history = results["event_history"]

    # Extract event times and create dense time array
    if len(event_history) > 0:
        event_times = np.array([e["time"] for e in event_history])
        # Combine regular grid with actual event times
        regular_times = np.linspace(0, config.duration_years, 1000)
        times = np.sort(np.unique(np.concatenate([regular_times, event_times])))
    else:
        times = np.linspace(0, config.duration_years, 1000)

    # Cumulative tectonic loading (geometric moment)
    total_loading_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )

    # Include initial moment from spin-up
    initial_moment = results.get("cumulative_loading", 0.0) - (
        total_loading_rate * config.duration_years
    )

    # cumulative_loading = initial_moment + times * total_loading_rate  # m³

    # NEW (correct):
    cumulative_loading = times * total_loading_rate  # m³ - starts at zero

    # Cumulative seismic release (geometric moment)
    # Build efficiently using cumsum
    cumulative_release = np.zeros_like(times)

    if len(event_history) > 0:
        # Get event times and moments
        event_times_array = np.array([e["time"] for e in event_history])
        event_moments_array = np.array([e["geom_moment"] for e in event_history])

        # For each plot time, find how many events occurred before it
        for i, t in enumerate(times):
            mask = event_times_array <= t
            cumulative_release[i] = np.sum(event_moments_array[mask])

    # Convert to seismic moment for plotting
    cumulative_loading_seismic = cumulative_loading * config.shear_modulus_Pa
    cumulative_release_seismic = cumulative_release * config.shear_modulus_Pa

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top panel: Cumulative moment
    ax1.plot(
        times,
        cumulative_loading_seismic,
        "b-",
        linewidth=2,
        label="Cumulative Tectonic Loading",
    )
    ax1.plot(
        times,
        cumulative_release_seismic,
        "r-",
        linewidth=2,
        label="Cumulative Seismic Release",
    )
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Cumulative Moment (N·m)")
    ax1.set_title("Moment Budget: Loading vs. Release")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Moment deficit
    moment_deficit = cumulative_loading - cumulative_release
    ax2.plot(times, moment_deficit, "purple", linewidth=2)
    ax2.axhline(
        0,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Moment Deficit (m³)")
    ax2.set_title("Accumulated Moment Deficit")
    ax2.grid(True, alpha=0.3)

    # Compute coupling coefficient
    max_time = times[-1]
    if max_time > 0:
        total_loaded = cumulative_loading[-1]
        total_released = cumulative_release[-1]
        coupling_coef = total_released / total_loaded

        ax2.text(
            0.02,
            0.95,
            f"Seismic Coupling: {coupling_coef:.3f}",
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/moment_budget.png", dpi=150)
    plt.close()

    # Debug prints
    print("\n" + "=" * 70)
    print("MOMENT BUDGET DIAGNOSTICS")
    print("=" * 70)
    print(f"From results dict:")
    print(
        f"  results['cumulative_loading']: {results.get('cumulative_loading', 0.0):.2e} m³"
    )
    print(
        f"  results['cumulative_release']: {results.get('cumulative_release', 0.0):.2e} m³"
    )
    print(f"\nComputed for plot:")
    print(f"  initial_moment: {initial_moment:.2e} m³")
    print(f"  total_loading_rate: {total_loading_rate:.2e} m³/yr")
    print(f"  duration: {config.duration_years:.0f} years")
    print(
        f"  computed loading from rate: {total_loading_rate * config.duration_years:.2e} m³"
    )
    print(f"\nAt final time point:")
    print(f"  cumulative_loading[-1]: {cumulative_loading[-1]:.2e} m³")
    print(f"  cumulative_release[-1]: {cumulative_release[-1]:.2e} m³")
    print(f"  coupling: {cumulative_release[-1] / cumulative_loading[-1]:.3f}")
    print(f"\nFrom simulator printout:")
    if len(event_history) > 0:
        sim_coupling = results.get("cumulative_release", 0.0) / results.get(
            "cumulative_loading", 0.0
        )
        print(f"  coupling: {sim_coupling:.3f}")
    print("=" * 70 + "\n")


def plot_event_rate_evolution(results, config):
    """
    Plot how event rate evolves through time

    Shows clustering vs. quiescence
    """
    event_history = results["event_history"]

    if len(event_history) < 2:
        print("Not enough events for rate analysis")
        return

    # Extract event times
    event_times = [e["time"] for e in event_history]

    # Compute inter-event times
    inter_event_times = np.diff(event_times)
    inter_event_mid_times = [
        (event_times[i] + event_times[i + 1]) / 2 for i in range(len(event_times) - 1)
    ]

    # Compute instantaneous rate (1/inter-event time)
    instantaneous_rates = 1.0 / np.array(inter_event_times)  # events/year

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top panel: Inter-event times
    ax1.scatter(
        inter_event_mid_times,
        inter_event_times,
        c=inter_event_times,
        cmap="coolwarm_r",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Inter-Event Time (years)", fontsize=12)
    ax1.set_title("Clustering and Quiescence Patterns", fontsize=14, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    cbar = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar.set_label("Inter-Event Time (years)", fontsize=11)

    # Bottom panel: Instantaneous rate
    ax2.scatter(
        inter_event_mid_times,
        instantaneous_rates,
        c=instantaneous_rates,
        cmap="coolwarm",
        s=50,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    ax2.set_xlabel("Time (years)", fontsize=12)
    ax2.set_ylabel("Event Rate (events/year)", fontsize=12)
    ax2.set_title("Instantaneous Event Rate", fontsize=13, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label("Event Rate", fontsize=11)

    plt.tight_layout()

    # Save
    output_path = Path(config.output_dir) / "event_rate_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    return fig


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

    fig, ax = plt.subplots(figsize=(12, 2))

    # Plot events
    for i in range(len(times)):
        plt.plot(
            [times[i], times[i]], [3.5, magnitudes[i]], "-k", linewidth=0.5, zorder=1
        )

    ax.scatter(
        times,
        magnitudes,
        c=magnitudes,
        cmap="YlOrRd",
        s=1e-4 * np.array(magnitudes) ** 8.0,
        alpha=1.0,
        edgecolors="black",
        linewidth=0.5,
        zorder=10,
    )

    ax.set_xlabel("$t$ (years)", fontsize=12)
    ax.set_ylabel("$M$", fontsize=12)
    ax.set_title(
        f"{len(event_history)} events",
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
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(12, 2))

    to_plot = np.log10(slip_grid.T)
    min_val = np.nanmin(to_plot)
    to_plot[~np.isfinite(to_plot)] = min_val

    im = ax.imshow(
        to_plot,
        origin="upper",
        aspect="auto",
        extent=[0, config.fault_length_km, 0, config.fault_depth_km],
        cmap="hot_r",
    )
    ax.invert_yaxis()

    ax.set_xlabel("$x$ (km)", fontsize=12)
    ax.set_ylabel("$d$ (km)", fontsize=12)
    ax.set_title("Cumulative coseismic clip", fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("slip (m)", fontsize=11)

    # # Mark hypocenters
    # hypo_x = [e["hypocenter_x_km"] for e in event_history]
    # hypo_z = [e["hypocenter_z_km"] for e in event_history]
    # ax.scatter(
    #     hypo_x,
    #     hypo_z,
    #     c="cyan",
    #     s=5,
    #     marker=".",
    #     edgecolors="black",
    #     linewidth=0.25,
    #     alpha=0.7,
    #     label="Hypocenters",
    # )

    # ax.legend()
    plt.tight_layout()

    # Save
    output_path = Path(config.output_dir) / "cumulative_slip.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
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
    # all_moments = np.sign(all_moments) * np.abs(all_moments) ** 0.2
    vmin, vmax = np.percentile(all_moments, [1, 99])

    m_grid = np.sign(m_grid) * np.abs(m_grid) ** 0.5

    im = ax.imshow(
        m_grid.T,
        origin="lower",
        aspect="equal",
        extent=[0, config.fault_length_km, 0, config.fault_depth_km],
        cmap="RdYlBu_r",
        vmin=-5e3,
        vmax=5e3,
    )

    ax.set_xlabel("$x$ (km)", fontsize=12)
    ax.set_ylabel("$d$ (km)", fontsize=12)
    title = ax.set_title("$t$ = 0.0 years", fontsize=12)

    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label("Geometric Moment", fontsize=11)

    def update(frame):
        m_grid = moment_snapshots[frame].reshape(
            mesh["n_along_strike"], mesh["n_down_dip"]
        )
        m_grid = np.sign(m_grid) * np.abs(m_grid) ** 0.5

        im.set_array(m_grid.T)
        title.set_text(f"$t$ = {snapshot_times[frame]:.1f} years")
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

    # NEW PLOTS
    plot_moment_history(results, config)
    plot_moment_budget(results, config)
    plot_event_rate_evolution(results, config)

    # Animation (optional, can take time)
    # create_moment_animation(results, config)

    print("\nAll plots generated!")
