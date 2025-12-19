"""
Visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from tqdm import tqdm

FONTSIZE = 10


def plot_moment_budget(results, config):
    """
    Plot moment budget showing loading vs release (coseismic + afterslip)

    Uses actual event times to ensure accurate tracking throughout simulation.
    Now includes afterslip release for complete moment accounting.
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

    # Get afterslip snapshots if available (for total release accounting)
    afterslip_cumulative = None
    afterslip_times = None
    if "afterslip_snapshots" in results:
        afterslip_cumulative = results["afterslip_snapshots"]
        afterslip_times = results.get(
            "times", np.linspace(0, config.duration_years, len(afterslip_cumulative))
        )

    # These correctly account for all moment including initial spin-up
    if "cumulative_loading" in results and "cumulative_release" in results:
        # Get final values from simulator
        final_loading = results["cumulative_loading"]
        final_release_total = results[
            "cumulative_release"
        ]  # Includes coseismic + afterslip

        # Create time array for plotting
        final_time = config.duration_years

        # Linear loading over time (constant rate)
        cumulative_loading = times * (final_loading / final_time)

        # Coseismic release: step function at each event time
        cumulative_coseismic = np.zeros_like(times)
        if len(event_history) > 0:
            event_times_array = np.array([e["time"] for e in event_history])
            event_moments_array = np.array([e["geom_moment"] for e in event_history])

            for i, t in enumerate(times):
                mask = event_times_array <= t
                cumulative_coseismic[i] = np.sum(event_moments_array[mask])

        # Afterslip release: interpolate from snapshots
        cumulative_afterslip = np.zeros_like(times)
        if afterslip_cumulative is not None and len(afterslip_cumulative) > 0:
            # Sum over spatial elements and convert to geometric moment
            afterslip_total = (
                np.sum(afterslip_cumulative, axis=1) * config.element_area_m2
            )
            # Interpolate to dense time array
            cumulative_afterslip = np.interp(times, afterslip_times, afterslip_total)

        # Total release = coseismic + afterslip
        cumulative_release = cumulative_coseismic + cumulative_afterslip
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

        cumulative_coseismic = np.zeros_like(times)
        if len(event_history) > 0:
            event_times_array = np.array([e["time"] for e in event_history])
            event_moments_array = np.array([e["geom_moment"] for e in event_history])

            for i, t in enumerate(times):
                mask = event_times_array <= t
                cumulative_coseismic[i] = np.sum(event_moments_array[mask])

        cumulative_afterslip = np.zeros_like(times)
        cumulative_release = cumulative_coseismic

    # Convert to seismic moment for plotting
    cumulative_loading_seismic = cumulative_loading
    cumulative_release_seismic = cumulative_release
    cumulative_coseismic_seismic = cumulative_coseismic
    cumulative_afterslip_seismic = cumulative_afterslip

    total_released = cumulative_release[-1]
    coseismic_released = cumulative_coseismic[-1]
    afterslip_released = cumulative_afterslip[-1]
    afterslip_fraction = (
        afterslip_released / total_released if total_released > 0 else 0
    )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Top panel: Cumulative moment with breakdown
    ax1.plot(
        times,
        cumulative_loading_seismic,
        "b-",
        color="tab:blue",
        linewidth=1.0,
        label="accumulation",
    )
    ax1.plot(
        times,
        cumulative_release_seismic,
        "r-",
        color="tab:red",
        linewidth=1.0,
        label="coseismic + afterslip",
    )
    # Show breakdown if afterslip is significant
    if np.max(cumulative_afterslip) > 0:
        ax1.plot(
            times,
            cumulative_coseismic_seismic,
            "r-",
            color="tab:orange",
            linewidth=1,
            alpha=0.7,
            label=f"coseismic ({100 * (1 - afterslip_fraction):.0f}%)",
        )

    ax1.plot(
        times,
        cumulative_release_seismic - cumulative_coseismic_seismic,
        "r-",
        color="tab:purple",
        linewidth=1,
        label=f"afterslip ({100 * afterslip_fraction:.0f}%)",
    )

    # plt.fill_between(
    #     times,
    #     moment_deficit,
    #     0.0,
    #     where=moment_deficit >= 0,
    #     interpolate=True,
    #     color="tab:orange",
    #     edgecolor=None,
    # )

    ax1.set_xlabel("$t$ (years)")
    ax1.set_ylabel("cumulative geometric moment (m$^3$)")
    ax1.legend(loc="upper left")
    ax1.set_xlim([0, config.duration_years])
    ax1.set_ylim(bottom=0)
    ax1.grid(False)

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
    ax2.set_xlabel("$t$ (years)")
    ax2.set_ylabel("geometric moment budget (m$^3$)")
    ax2.grid(False)

    # Compute coupling coefficients
    max_time = times[-1]
    if max_time > 0:
        total_loaded = cumulative_loading[-1]
        total_released = cumulative_release[-1]
        coseismic_released = cumulative_coseismic[-1]
        afterslip_released = cumulative_afterslip[-1]

        total_coupling = total_released / total_loaded if total_loaded > 0 else 0
        coseismic_coupling = (
            coseismic_released / total_loaded if total_loaded > 0 else 0
        )
        afterslip_fraction = (
            afterslip_released / total_released if total_released > 0 else 0
        )

    ax2.set_xlim([0, config.duration_years])
    # ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/moment_budget.png", dpi=500)
    plt.close()


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

    fig = plt.figure(figsize=(6, 6))

    # Moment budget data calculation (includes afterslip)
    event_history = results["event_history"]

    # Extract event times and create dense time array
    if len(event_history) > 0:
        event_times = np.array([e["time"] for e in event_history])
        # Combine regular grid with actual event times
        regular_times = np.linspace(0, config.duration_years, 1000)
        times = np.sort(np.unique(np.concatenate([regular_times, event_times])))
    else:
        times = np.linspace(0, config.duration_years, 1000)

    # Get afterslip snapshots if available (for total release accounting)
    afterslip_cumulative = None
    afterslip_times = None
    if "afterslip_snapshots" in results:
        afterslip_cumulative = results["afterslip_snapshots"]
        afterslip_times = results.get(
            "times", np.linspace(0, config.duration_years, len(afterslip_cumulative))
        )

    # These correctly account for all moment including initial spin-up
    if "cumulative_loading" in results and "cumulative_release" in results:
        # Get final values from simulator
        final_loading = results["cumulative_loading"]

        # Create time array for plotting
        final_time = config.duration_years

        # Linear loading over time (constant rate)
        cumulative_loading = times * (final_loading / final_time)

        # Coseismic release: step function at each event time
        cumulative_coseismic = np.zeros_like(times)
        if len(event_history) > 0:
            event_times_array = np.array([e["time"] for e in event_history])
            event_moments_array = np.array([e["geom_moment"] for e in event_history])

            for i, t in enumerate(times):
                mask = event_times_array <= t
                cumulative_coseismic[i] = np.sum(event_moments_array[mask])

        # Afterslip release: interpolate from snapshots
        cumulative_afterslip = np.zeros_like(times)
        if afterslip_cumulative is not None and len(afterslip_cumulative) > 0:
            # Sum over spatial elements and convert to geometric moment
            afterslip_total = (
                np.sum(afterslip_cumulative, axis=1) * config.element_area_m2
            )
            # Interpolate to dense time array
            cumulative_afterslip = np.interp(times, afterslip_times, afterslip_total)

        # Total release = coseismic + afterslip
        cumulative_release = cumulative_coseismic + cumulative_afterslip
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

        cumulative_coseismic = np.zeros_like(times)
        if len(event_history) > 0:
            event_times_array = np.array([e["time"] for e in event_history])
            event_moments_array = np.array([e["geom_moment"] for e in event_history])

            for i, t in enumerate(times):
                mask = event_times_array <= t
                cumulative_coseismic[i] = np.sum(event_moments_array[mask])

        cumulative_afterslip = np.zeros_like(times)
        cumulative_release = cumulative_coseismic

    # Compute afterslip fraction for labels
    total_released = cumulative_release[-1]
    afterslip_released = cumulative_afterslip[-1]
    afterslip_fraction = (
        afterslip_released / total_released if total_released > 0 else 0
    )

    # Top panel: Cumulative moment (loading vs release)
    plt.subplot(3, 1, 1)
    plt.plot(
        times,
        cumulative_loading,
        "-",
        color="tab:blue",
        linewidth=0.5,
        label="accumulation",
    )
    plt.plot(
        times,
        cumulative_release,
        "-",
        color="tab:red",
        linewidth=0.5,
        label="coseismic + afterslip",
    )
    plt.plot(
        times,
        cumulative_coseismic,
        "-",
        color="tab:orange",
        linewidth=0.5,
        label=f"coseismic ({100 * (1 - afterslip_fraction):.0f}%)",
    )
    plt.plot(
        times,
        cumulative_afterslip,
        "-",
        color="tab:purple",
        linewidth=0.5,
        label=f"afterslip ({100 * afterslip_fraction:.0f}%)",
    )

    # plt.xlabel("$t$ (years)", fontsize=FONTSIZE)
    plt.ylabel("$m$ (m$^3$)", fontsize=FONTSIZE)
    plt.xlim(0, config.duration_years)
    plt.xticks([])
    plt.ylim(bottom=5.0)
    plt.legend(loc="upper left", fontsize=FONTSIZE - 2)

    # Expected events (1-year rolling window from λ(t))
    plt.subplot(3, 1, 2)

    # Load λ(t) time series that was stored during simulation
    lambda_times = results["times"]
    lambda_values = results["lambda_history"]

    # Convert to NumPy arrays if HDF5 datasets
    if hasattr(lambda_times, "shape"):
        lambda_times = lambda_times[:]
    if hasattr(lambda_values, "shape"):
        lambda_values = lambda_values[:]

    # Convert \rho(t) to incremental expected events per timestep
    dt_years = config.time_step_years
    lambda_incremental = lambda_values * dt_years  # Expected events per timestep

    # Compute N-year forward-looking rolling sum (for smoothing with yearly timesteps)
    window_timesteps = 1  # 10-year window
    moving_window = np.zeros(len(lambda_values))
    for i in range(len(lambda_values)):
        # Sum next window_timesteps (or remaining timesteps at end)
        end_idx = min(i + window_timesteps, len(lambda_values))
        moving_window[i] = np.sum(lambda_incremental[i:end_idx])

    # No downsampling needed with yearly timesteps - already at annual resolution
    annual_times = lambda_times
    annual_expected = moving_window

    # Plot expected events in next 10 years (rolling window)
    plt.plot(
        annual_times,
        annual_expected,
        "-",
        linewidth=0.25,
        markersize=2,
        color="k",
        label="Expected (10-yr window)",
    )

    plt.fill_between(
        annual_times, annual_expected, 0, color="tab:pink", edgecolor=None, alpha=0.5
    )

    # plt.xlabel("$t$ (years)", fontsize=FONTSIZE)
    plt.ylabel("$\\rho(t)$ (events / yr)", fontsize=FONTSIZE)
    plt.xlim([0, config.duration_years])
    # plt.ylim([1e-2, 1e1])
    plt.ylim([0, 3])

    plt.xticks([])
    # plt.yscale("log")

    # Magnitude time series
    plt.subplot(3, 1, 3)
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
    # plt.ylabel("$\\mathrm{M}_\\mathrm{W}$", fontsize=FONTSIZE)
    plt.ylabel("$M$", fontsize=FONTSIZE)

    plt.xlim(0, config.duration_years)
    plt.ylim([5, 8])

    # Save
    output_path = Path(config.output_dir) / "evolution_overview.png"
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
    Create animation of moment deficit evolution using contourf/contour style

    This will create an MP4 file showing moment deficit (m_a - m_r) through time
    Uses symmetric colorbar with fixed limits from extrema
    Reconstructs deficit same way as plot_moment_snapshots for consistency
    """
    print("Starting animation!!!")

    def scale(values):
        power = 0.2
        scaled_values = np.sign(values) * (np.abs(values) ** power)
        return scaled_values

    release_snapshots = results["release_snapshots"]
    afterslip_snapshots = results["afterslip_snapshots"]  # NEW: Load afterslip data
    snapshot_times = results["snapshot_times"]
    mesh = results["mesh"]
    slip_rate = results["slip_rate"]  # m/year per element
    extrema = results["moment_extrema"]

    # Calculate stride to achieve 1 frame per year in animation
    # Works dynamically with any snapshot_interval_years setting
    desired_frame_interval_years = 1.0
    stride = max(1, int(desired_frame_interval_years / config.snapshot_interval_years))
    annual_indices = np.arange(0, len(snapshot_times), stride)

    # Compute symmetric colorbar limits for DEFICIT from extrema (guaranteed centered at zero)
    max_abs_deficit = max(abs(extrema["min_deficit"]), abs(extrema["max_deficit"]))
    max_abs_deficit = scale(max_abs_deficit)
    vmin_deficit = -max_abs_deficit
    vmax_deficit = max_abs_deficit

    # Compute afterslip colorbar limits (not symmetric - cumulative only grows)
    max_afterslip_elem = np.max(afterslip_snapshots[-1])  # Maximum at final time (m)
    max_afterslip_scaled = scale(
        max_afterslip_elem * config.element_area_m2
    )  # Scale to m³
    vmin_afterslip = 0.0
    # Prevent zero range when afterslip is disabled or zero
    vmax_afterslip = max(max_afterslip_scaled, 1e-10)

    # Define contour levels (fixed for all frames)
    n_levels = 20
    levels_deficit = np.linspace(vmin_deficit, vmax_deficit, n_levels)  # Symmetric
    levels_afterslip = np.linspace(
        vmin_afterslip, vmax_afterslip, n_levels
    )  # Non-symmetric

    # Create grids for contourf plotting
    length_vec = np.linspace(0, config.fault_length_km, config.n_along_strike)
    depth_vec = np.linspace(0, config.fault_depth_km, config.n_down_dip)
    length_grid, depth_grid = np.meshgrid(length_vec, depth_vec)

    # Create figure with 2 vertically stacked panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))

    # Initial frame setup
    snapshot_idx = annual_indices[0]
    actual_time = snapshot_times[snapshot_idx]

    # ===== PANEL 1 (TOP): Moment Deficit =====
    spatial_loading = slip_rate * config.element_area_m2 * actual_time  # m³
    spatial_release = release_snapshots[snapshot_idx] * config.element_area_m2  # m³
    deficit = spatial_loading - spatial_release

    deficit_grid = deficit.reshape(mesh["n_along_strike"], mesh["n_down_dip"]).T
    deficit_grid = scale(deficit_grid)

    # Initial contourf with symmetric limits
    contourf_deficit = ax1.contourf(
        length_grid,
        depth_grid,
        deficit_grid,
        cmap="Spectral_r",
        levels=levels_deficit,
        vmin=vmin_deficit,
        vmax=vmax_deficit,
    )

    ax1.set_xlabel("$x$ (km)", fontsize=FONTSIZE)
    ax1.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
    ax1.set_aspect("equal")
    ax1.invert_yaxis()  # Match slip map orientation
    title1 = ax1.set_title(
        "$m_\\mathrm{a} - m_\\mathrm{r}$, $t$ = 0.0 years", fontsize=FONTSIZE
    )

    # Add colorbar
    cbar1 = plt.colorbar(contourf_deficit, ax=ax1)

    # ===== PANEL 2 (BOTTOM): Afterslip Cumulative Release =====
    spatial_afterslip = afterslip_snapshots[snapshot_idx] * config.element_area_m2  # m³
    afterslip_grid = spatial_afterslip.reshape(
        mesh["n_along_strike"], mesh["n_down_dip"]
    ).T
    afterslip_grid = scale(afterslip_grid)

    # Initial contourf with non-symmetric limits (0 to max)
    contourf_afterslip = ax2.contourf(
        length_grid,
        depth_grid,
        afterslip_grid,
        cmap="YlOrRd",  # Sequential colormap for cumulative
        levels=levels_afterslip,
        vmin=vmin_afterslip,
        vmax=vmax_afterslip,
    )

    ax2.set_xlabel("$x$ (km)", fontsize=FONTSIZE)
    ax2.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
    ax2.set_aspect("equal")
    ax2.invert_yaxis()

    # Add title with disabled note if applicable
    afterslip_status = "" if config.afterslip_enabled else " (DISABLED)"
    title2 = ax2.set_title(
        f"Afterslip cumulative release{afterslip_status}, $t$ = 0.0 years",
        fontsize=FONTSIZE,
    )

    # Add colorbar
    cbar2 = plt.colorbar(contourf_afterslip, ax=ax2)

    plt.tight_layout()

    # Setup progress bar for annual frames
    n_frames = len(annual_indices)
    print(
        f"\nCreating moment deficit animation ({n_frames} annual frames from {len(snapshot_times)} snapshots)..."
    )
    pbar = tqdm(total=n_frames, desc="Rendering frames")

    def update(frame):
        """Update function for animation - redraw contours for both panels"""
        # Clear previous contours from BOTH panels
        for c in ax1.collections:
            c.remove()
        for c in ax2.collections:
            c.remove()

        # Map frame to annual snapshot index
        snapshot_idx = annual_indices[frame]
        actual_time = snapshot_times[snapshot_idx]

        # ===== PANEL 1: Moment Deficit =====
        spatial_loading = slip_rate * config.element_area_m2 * actual_time  # m³
        spatial_release = release_snapshots[snapshot_idx] * config.element_area_m2  # m³
        deficit = spatial_loading - spatial_release

        deficit_grid = deficit.reshape(mesh["n_along_strike"], mesh["n_down_dip"]).T
        deficit_grid = scale(deficit_grid)

        # Redraw contourf with fixed symmetric limits
        ax1.contourf(
            length_grid,
            depth_grid,
            deficit_grid,
            cmap="Spectral_r",
            levels=levels_deficit,
            vmin=vmin_deficit,
            vmax=vmax_deficit,
        )

        # Update title
        title1.set_text(
            f"$m_\\mathrm{{a}} - m_\\mathrm{{r}}$, $t$ = {actual_time:.1f} years"
        )

        # ===== PANEL 2: Afterslip Cumulative Release =====
        spatial_afterslip = (
            afterslip_snapshots[snapshot_idx] * config.element_area_m2
        )  # m³
        afterslip_grid = spatial_afterslip.reshape(
            mesh["n_along_strike"], mesh["n_down_dip"]
        ).T
        afterslip_grid = scale(afterslip_grid)

        # Redraw contourf
        ax2.contourf(
            length_grid,
            depth_grid,
            afterslip_grid,
            cmap="YlOrRd",
            levels=levels_afterslip,
            vmin=vmin_afterslip,
            vmax=vmax_afterslip,
        )

        # Update title
        title2.set_text(
            f"Afterslip cumulative release{afterslip_status}, $t$ = {actual_time:.1f} years"
        )

        # Update progress bar
        pbar.update(1)

        return ax1.collections + ax2.collections + [title1, title2]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)

    # Save
    output_path = Path(config.output_dir) / "moment_evolution.mp4"
    anim.save(output_path, writer="ffmpeg", fps=20, dpi=300)

    # Close progress bar
    pbar.close()
    print(f"Saved animation: {output_path}")
    plt.close()

    return output_path


def plot_all(results, config):
    """
    Generate all plots
    """
    print("\nGenerating plots...")

    plot_moment_snapshots(results, config)
    plot_cumulative_slip_map(results, config)
    plot_moment_budget(results, config)
    plot_evolution_overview(results, config)

    # Animation (can take a long time, may fail if ffmpeg not working)
    # try:
    #     create_moment_animation(results, config)
    # except Exception as e:
    #     print(f"\nWARNING: Animation creation failed: {e}")
    #     print("  Try: brew reinstall ffmpeg")
    #     print("  Other plots completed successfully.")
