"""
Visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

FONTSIZE = 10


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

    # These correctly account for all moment including initial spin-up
    if "cumulative_loading" in results and "cumulative_release" in results:
        # Get final values from simulator
        final_loading = results["cumulative_loading"]

        # Create time array for plotting
        final_time = config.duration_years

        # Linear loading over time (constant rate)
        cumulative_loading = times * (final_loading / final_time)

        # Release: step function at each event time
        cumulative_release = np.zeros_like(times)
        if len(event_history) > 0:
            event_times_array = np.array([e["time"] for e in event_history])
            event_moments_array = np.array([e["geom_moment"] for e in event_history])

            for i, t in enumerate(times):
                mask = event_times_array <= t
                cumulative_release[i] = np.sum(event_moments_array[mask])
    else:
        # Fallback: reconstruct from loading rate (old method, less accurate)
        print(
            "WARNING: Using fallback loading calculation (cumulative values not in results)"
        )
        total_loading_rate = (
            config.background_slip_rate_m_yr
            * config.n_elements
            * config.element_area_m2
        )
        cumulative_loading = times * total_loading_rate

        cumulative_release = np.zeros_like(times)
        if len(event_history) > 0:
            event_times_array = np.array([e["time"] for e in event_history])
            event_moments_array = np.array([e["geom_moment"] for e in event_history])

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
        linewidth=1,
        label="cumulative moment accumulation",
    )
    ax1.plot(
        times,
        cumulative_release_seismic,
        "r-",
        linewidth=1,
        label="cumulative moment release",
    )
    ax1.set_xlabel("$t$ (years)")
    ax1.set_ylabel("Cumulative Moment (N·m)")
    ax1.set_title("Moment Budget: Loading vs. Release")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Moment deficit
    moment_deficit = cumulative_loading - cumulative_release
    ax2.plot(times, moment_deficit, "r", linewidth=1.0)
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
            fontsize=FONTSIZE,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/moment_budget.png", dpi=500)
    plt.close()

    # Debug prints - verify coupling agreement
    print("\n" + "=" * 70)
    print("MOMENT BUDGET PLOT DIAGNOSTICS")
    print("=" * 70)
    print("From simulator results dict (authoritative):")
    sim_loading = results.get("cumulative_loading", 0.0)
    sim_release = results.get("cumulative_release", 0.0)
    print(f"  cumulative_loading: {sim_loading:.2e} m³")
    print(f"  cumulative_release: {sim_release:.2e} m³")
    if sim_loading > 0:
        sim_coupling = sim_release / sim_loading
        print(f"  coupling: {sim_coupling:.4f}")

    print("\nPlot final values (should match):")
    print(f"  cumulative_loading[-1]: {cumulative_loading[-1]:.2e} m³")
    print(f"  cumulative_release[-1]: {cumulative_release[-1]:.2e} m³")
    if cumulative_loading[-1] > 0:
        plot_coupling = cumulative_release[-1] / cumulative_loading[-1]
        print(f"  coupling: {plot_coupling:.4f}")

    if sim_loading > 0 and cumulative_loading[-1] > 0:
        loading_match = abs(cumulative_loading[-1] - sim_loading) / sim_loading < 0.01
        release_match = abs(cumulative_release[-1] - sim_release) / sim_release < 0.01
        coupling_match = abs(plot_coupling - sim_coupling) < 0.01
        print("\nAgreement check:")
        print(
            f"  Loading match: {loading_match} (Δ={(cumulative_loading[-1] - sim_loading) / sim_loading * 100:.2f}%)"
        )
        print(
            f"  Release match: {release_match} (Δ={(cumulative_release[-1] - sim_release) / sim_release * 100:.2f}%)"
        )
        print(
            f"  Coupling match: {coupling_match} (Δ={(plot_coupling - sim_coupling) * 100:.2f}%)"
        )
    print("=" * 70 + "\n")


def plot_evolution_overview(results, config):
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

    fig = plt.figure(figsize=(10, 8))  # Increased height for 4 panels

    # Moment budget
    event_history = results["event_history"]

    # Extract event times and create dense time array
    if len(event_history) > 0:
        event_times = np.array([e["time"] for e in event_history])
        # Combine regular grid with actual event times
        regular_times = np.linspace(0, config.duration_years, 1000)
        times = np.sort(np.unique(np.concatenate([regular_times, event_times])))
    else:
        times = np.linspace(0, config.duration_years, 1000)

    # These correctly account for all moment including initial spin-up
    if "cumulative_loading" in results and "cumulative_release" in results:
        # Get final values from simulator
        final_loading = results["cumulative_loading"]

        # Create time array for plotting
        final_time = config.duration_years

        # Linear loading over time (constant rate)
        cumulative_loading = times * (final_loading / final_time)

        # Release: step function at each event time
        cumulative_release = np.zeros_like(times)
        if len(event_history) > 0:
            event_times_array = np.array([e["time"] for e in event_history])
            event_moments_array = np.array([e["geom_moment"] for e in event_history])

            for i, t in enumerate(times):
                mask = event_times_array <= t
                cumulative_release[i] = np.sum(event_moments_array[mask])
    else:
        # Fallback: reconstruct from loading rate (old method, less accurate)
        print(
            "WARNING: Using fallback loading calculation (cumulative values not in results)"
        )
        total_loading_rate = (
            config.background_slip_rate_m_yr
            * config.n_elements
            * config.element_area_m2
        )
        cumulative_loading = times * total_loading_rate

        cumulative_release = np.zeros_like(times)
        if len(event_history) > 0:
            event_times_array = np.array([e["time"] for e in event_history])
            event_moments_array = np.array([e["geom_moment"] for e in event_history])

            for i, t in enumerate(times):
                mask = event_times_array <= t
                cumulative_release[i] = np.sum(event_moments_array[mask])

    # Convert to seismic moment for plotting
    cumulative_loading_seismic = cumulative_loading
    cumulative_release_seismic = cumulative_release
    moment_deficit = cumulative_loading_seismic - cumulative_release_seismic
    moment_deficit_y_lim = 1.1 * np.max(np.abs(moment_deficit))

    plt.subplot(4, 1, 1)
    plt.fill_between(
        times,
        moment_deficit,
        0.0,
        where=moment_deficit >= 0,
        interpolate=True,
        color="tab:orange",
        edgecolor=None,
    )
    plt.fill_between(
        times,
        moment_deficit,
        0.0,
        where=moment_deficit <= 0,
        interpolate=True,
        color="tab:cyan",
        edgecolor=None,
    )

    plt.plot(
        times,
        moment_deficit,
        "-",
        linewidth=0.25,
        color="k",
    )
    plt.plot(
        times,
        np.zeros_like(times),
        "-",
        linewidth=0.25,
        color="k",
    )

    plt.xlabel("$t$ (years)", fontsize=FONTSIZE)
    plt.ylabel("$m_\\mathrm{a} - m_\\mathrm{r}$ (m$^3$)", fontsize=FONTSIZE)
    plt.xlim(0, config.duration_years)
    plt.ylim(-moment_deficit_y_lim, moment_deficit_y_lim)

    # Expected events (1-year rolling window from λ(t))
    plt.subplot(4, 1, 2)

    # Load λ(t) time series that was stored during simulation
    lambda_times = results["times"]
    lambda_values = results["lambda_history"]

    # Convert to NumPy arrays if HDF5 datasets
    if hasattr(lambda_times, "shape"):
        lambda_times = lambda_times[:]
    if hasattr(lambda_values, "shape"):
        lambda_values = lambda_values[:]

    # Convert λ(t) to incremental expected events per timestep
    dt_years = config.time_step_days / 365.25
    lambda_incremental = lambda_values * dt_years  # Expected events per day

    # Compute 1-year forward-looking rolling sum
    window_days = 365
    moving_annual = np.zeros(len(lambda_values))
    for i in range(len(lambda_values)):
        # Sum next 365 days (or remaining days at end)
        end_idx = min(i + window_days, len(lambda_values))
        moving_annual[i] = np.sum(lambda_incremental[i:end_idx])

    # Diagnostic: Print first few years to verify calculation
    print(f"\nPanel 2 diagnostic (expected events in next year):")
    print(f"  Year 0 (day 0): {moving_annual[0]:.2f} events")
    print(f"  Year 1 (day 365): {moving_annual[365]:.2f} events")
    print(f"  Year 2 (day 730): {moving_annual[730]:.2f} events")

    # Downsample to annual resolution for clearer visualization
    # Sample at the start of each year (days 0, 365, 730, ...)
    n_years = int(np.ceil(config.duration_years))
    annual_indices = np.arange(0, min(n_years * 365, len(moving_annual)), 365)
    annual_times = lambda_times[annual_indices]
    annual_expected = moving_annual[annual_indices]

    # Plot expected events in next year (downsampled to annual)
    plt.plot(
        annual_times,
        annual_expected,
        "-",
        linewidth=0.25,
        markersize=2,
        color="k",
        label="Expected (1-yr window)",
    )

    plt.fill_between(
        annual_times, annual_expected, 0, color="tab:pink", edgecolor=None, alpha=0.5
    )

    plt.xlabel("$t$ (years)", fontsize=FONTSIZE)
    plt.ylabel("$\\lambda(t)$ (events/yr)", fontsize=FONTSIZE)
    plt.xlim([0, config.duration_years])
    plt.ylim([1e-1, 1e2])
    plt.yscale("log")

    # Magnitude time series
    plt.subplot(4, 1, 3)
    event_history = results["event_history"]
    times = [e["time"] for e in event_history]
    magnitudes = [e["magnitude"] for e in event_history]

    # Plot events
    for i in range(len(times)):
        if magnitudes[i] >= 6.0:
            plt.plot(
                [times[i], times[i]],
                [5.0, magnitudes[i]],
                "-k",
                linewidth=0.25,
                zorder=1,
            )

    plt.scatter(
        times,
        magnitudes,
        c=magnitudes,
        cmap="plasma",
        s=1e-8 * np.array(magnitudes) ** 12.0,
        alpha=1.0,
        edgecolors="black",
        linewidth=0.25,
        zorder=10,
    )

    plt.xlabel("$t$ (years)", fontsize=FONTSIZE)
    plt.ylabel("$\\mathrm{M}_\\mathrm{W}$", fontsize=FONTSIZE)
    plt.xlim(0, config.duration_years)
    plt.ylim(bottom=5.0)

    # Save
    output_path = Path(config.output_dir) / "evolution_overview.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
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

    ax.set_xlabel("$t$ (years)", fontsize=FONTSIZE)
    ax.set_ylabel("$M$", fontsize=FONTSIZE)
    ax.set_title(
        f"{len(event_history)} events",
        fontsize=FONTSIZE,
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
    cbar.set_label("Magnitude", fontsize=FONTSIZE)

    plt.tight_layout()

    # Save
    output_path = Path(config.output_dir) / "magnitude_time_series.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")

    return fig


def plot_moment_snapshots(results, config, times_to_plot=None, plot_type="deficit"):
    """
    Plot moment distribution at multiple time slices

    Uses spatial reconstruction following moment_budget.png pattern:
    - spatial_loading = slip_rate × area × time (always increases)
    - spatial_release = cumulative_release × area (step increases at earthquakes)
    - spatial_deficit = loading - release (shows earthquake drops)

    Parameters:
    -----------
    plot_type : str
        "deficit" - plot moment deficit (reconstructed: loading - release)
        "release" - plot cumulative moment release only
    """
    snapshot_times = results["snapshot_times"]
    mesh = results["mesh"]
    slip_rate = results["slip_rate"]  # m/year per element
    release_snapshots = results["release_snapshots"]  # cumulative slip release (m)

    if times_to_plot is None:
        # Default: plot 6 evenly spaced times
        indices = np.linspace(0, len(snapshot_times) - 1, 6, dtype=int)
        times_to_plot = [snapshot_times[i] for i in indices]

    # Create grids for contourf plotting (matching cumulative_slip style)
    length_vec = np.linspace(0, config.fault_length_km, config.n_along_strike)
    depth_vec = np.linspace(0, config.fault_depth_km, config.n_down_dip)
    length_grid, depth_grid = np.meshgrid(length_vec, depth_vec)

    n_plots = len(times_to_plot)
    fig, axes = plt.subplots(6, 1, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(times_to_plot):
        # Find closest snapshot
        idx = np.argmin(np.abs(np.array(snapshot_times) - t))
        actual_time = snapshot_times[idx]

        # RECONSTRUCT spatial quantities (like moment_budget.png does for scalars)
        # Loading: always increases linearly with time
        spatial_loading = slip_rate * config.element_area_m2 * actual_time  # m³

        # Release: step function at earthquakes
        spatial_release = release_snapshots[idx] * config.element_area_m2  # m³

        # Select data based on plot type
        if plot_type == "release":
            moment_data = spatial_release
            title_text = "Cumulative Moment Release Through Time"
        else:  # deficit
            # Deficit shows drops when earthquakes occur!
            moment_data = spatial_loading - spatial_release
            title_text = "Moment Deficit Through Time"

        # Reshape to 2D grid
        moment_grid = moment_data.reshape(mesh["n_along_strike"], mesh["n_down_dip"])

        # Apply log transform (like cumulative_slip.png)
        to_plot = np.sign(moment_grid.T) * np.abs(moment_grid.T) ** (0.5)
        # min_val = np.nanmin(to_plot)
        # to_plot[~np.isfinite(to_plot)] = min_val

        # Plot with contourf + contour (matching cumulative_slip style)
        ax = axes[i]
        levels = np.linspace(-5000, 5000, 11)
        cbar_plot = ax.contourf(
            length_grid,
            depth_grid,
            to_plot,
            cmap="PiYG",
            levels=levels,
            extend="both",
        )

        ax.contour(
            length_grid,
            depth_grid,
            to_plot,
            colors="black",
            linewidths=0.5,
            linestyles="solid",
            levels=levels,
        )

        ax.set_xlabel("$x$ (km)", fontsize=FONTSIZE)
        ax.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
        ax.set_title(f"$t$ = {actual_time:.1f} years", fontsize=FONTSIZE)
        ax.invert_yaxis()

        cbar = plt.colorbar(cbar_plot, ax=ax)
        # cbar.set_label("log$_{10}$ moment (m³)", fontsize=FONTSIZE)

    plt.tight_layout()

    # Save
    output_path = Path(config.output_dir) / "moment_snapshots.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
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

    # from IPython import embed

    # embed()

    # Create grids for contourf plotting
    length_vec = np.linspace(0, config.fault_length_km, config.n_along_strike)
    depth_vec = np.linspace(0, config.fault_depth_km, config.n_down_dip)
    length_grid, depth_grid = np.meshgrid(length_vec, depth_vec)

    fig, ax = plt.subplots(figsize=(12, 2))

    to_plot = np.log10(slip_grid.T)
    min_val = np.nanmin(to_plot)
    to_plot[~np.isfinite(to_plot)] = min_val

    cbar = ax.contourf(
        length_grid,
        depth_grid,
        to_plot,
        cmap="cool",
    )

    ax.contour(
        length_grid,
        depth_grid,
        to_plot,
        colors="black",  # solid black lines
        linewidths=0.5,  # adjust line thickness as needed
    )

    ax.set_xlabel("$x$ (km)", fontsize=FONTSIZE)
    ax.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
    # ax.set_title("Cumulative coseismic slip", fontsize=FONTSIZE)
    ax.invert_yaxis()
    cbar = plt.colorbar(cbar, ax=ax)
    cbar.set_label("log$_{10}$ slip (m)", fontsize=FONTSIZE)
    plt.tight_layout()

    # Save
    output_path = Path(config.output_dir) / "cumulative_slip.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")

    return fig


def create_moment_animation(results, config):
    """
    Create animation of moment evolution using contourf/contour style

    This will create an MP4 file showing moment evolution through time
    Uses the same visual style as plot_cumulative_slip_map
    """
    moment_snapshots = results["moment_snapshots"]
    snapshot_times = results["snapshot_times"]
    mesh = results["mesh"]

    # Create grids for contourf plotting (matching slip map style)
    length_vec = np.linspace(0, config.fault_length_km, config.n_along_strike)
    depth_vec = np.linspace(0, config.fault_depth_km, config.n_down_dip)
    length_grid, depth_grid = np.meshgrid(length_vec, depth_vec)

    fig, ax = plt.subplots(figsize=(12, 3))

    # Initial frame setup
    m_grid = moment_snapshots[0].reshape(mesh["n_along_strike"], mesh["n_down_dip"])

    # Apply log transform (like slip map)
    to_plot = np.log10(m_grid.T + 1e-10)  # Add small value to avoid log(0)
    min_val = np.nanmin(to_plot)
    to_plot[~np.isfinite(to_plot)] = min_val

    # Initial contourf and contour
    contourf_plot = ax.contourf(
        length_grid,
        depth_grid,
        to_plot,
        cmap="cool",
        levels=20,
    )

    contour_plot = ax.contour(
        length_grid,
        depth_grid,
        to_plot,
        colors="black",
        linewidths=0.5,
        levels=20,
    )

    ax.set_xlabel("$x$ (km)", fontsize=FONTSIZE)
    ax.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
    ax.invert_yaxis()  # Match slip map orientation
    title = ax.set_title("$t$ = 0.0 years", fontsize=FONTSIZE)

    # Add colorbar
    cbar = plt.colorbar(contourf_plot, ax=ax)
    cbar.set_label("log$_{10}$ moment (m³)", fontsize=FONTSIZE)

    plt.tight_layout()

    def update(frame):
        """Update function for animation - redraw contours each frame"""
        # Clear previous contours
        for c in ax.collections:
            c.remove()

        # Get current frame data
        m_grid = moment_snapshots[frame].reshape(
            mesh["n_along_strike"], mesh["n_down_dip"]
        )

        # Apply log transform
        to_plot = np.log10(m_grid.T + 1e-10)
        min_val = np.nanmin(to_plot)
        to_plot[~np.isfinite(to_plot)] = min_val

        # Redraw contourf and contour
        ax.contourf(
            length_grid,
            depth_grid,
            to_plot,
            cmap="cool",
            levels=20,
        )

        ax.contour(
            length_grid,
            depth_grid,
            to_plot,
            colors="black",
            linewidths=0.5,
            levels=20,
        )

        # Update title
        title.set_text(f"$t$ = {snapshot_times[frame]:.1f} years")

        return ax.collections + [title]

    anim = FuncAnimation(
        fig, update, frames=len(moment_snapshots), interval=100, blit=False
    )

    # Save
    output_path = Path(config.output_dir) / "moment_evolution.mp4"
    anim.save(output_path, writer="ffmpeg", fps=10, dpi=150)
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
    plot_moment_budget(results, config)
    plot_evolution_overview(results, config)

    # Animation (optional, can take time)
    # create_moment_animation(results, config)

    print("\nAll plots generated!")
