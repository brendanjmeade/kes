"""
Standalone test for seismic cycle with afterslip - EXTENDED 100-YEAR SIMULATION

This script extends the afterslip test to span a full 100-year seismic cycle.
Shows the transition from post-seismic afterslip to inter-seismic loading,
tracking moment accumulation toward the next earthquake.

Validates:
- Exponential velocity decay with spatially-varying rates
- Cumulative slip accumulation
- Moment budget tracking and depletion
- Continuous tectonic loading throughout cycle
- Fault coupling evolution over 100 years

Generates 6 diagnostic plots showing full cycle evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import Config
from geometry import create_fault_mesh
from afterslip import initialize_afterslip_sequence, update_afterslip_sequences

# =============================================================================
# SETUP - Mimic max_ent_afterslip.py
# =============================================================================

# Initialize config
config = Config()
config.compute_derived_parameters()

# Create mesh
mesh = create_fault_mesh(config)

# Define circular coseismic rupture (mimic max_ent_afterslip.py)
eq_center_x_km = config.fault_length_km / 2  # 100 km
eq_center_z_km = config.fault_depth_km / 2 - 10.0
eq_radius_km = 15.0  # 20 km radius
eq_slip_m = 1.0  # 1 meter slip

# Calculate distance from earthquake center for each element
centroids = mesh["centroids"]
dist_from_eq = np.sqrt(
    (centroids[:, 0] - eq_center_x_km) ** 2 + (centroids[:, 2] - eq_center_z_km) ** 2
)

# Create coseismic slip array (circular rupture)
slip_coseismic = np.zeros(config.n_elements)
slip_coseismic[dist_from_eq <= eq_radius_km] = eq_slip_m

# Find ruptured elements
ruptured_elements = np.where(dist_from_eq <= eq_radius_km)[0]

# Calculate magnitude (rough estimate)
area_co_m2 = len(ruptured_elements) * config.element_area_m2
M0_Nm = eq_slip_m * area_co_m2 * config.shear_modulus_Pa
magnitude = (2 / 3) * np.log10(M0_Nm) - 6.07

print("Test earthquake:")
print(f"  Magnitude: M {magnitude:.2f}")
print(f"  Rupture area: {area_co_m2 / 1e6:.1f} km²")
print(f"  Number of ruptured elements: {len(ruptured_elements)}")
print(f"  Rupture radius: {eq_radius_km} km")

# Create event dict (as would be passed from simulator)
event = {
    "ruptured_elements": ruptured_elements,
    "slip": slip_coseismic,
    "magnitude": magnitude,
    "time": 0.0,
}

# m_current represents residual moment after coseismic release
# CRITICAL: Drives spatial pattern of afterslip
# Simulate mid-cycle loading (e.g., 50 years at 10 mm/yr)
years_accumulated = 50.0
loading_rate_m_yr = 0.01  # 10 mm/yr
# loading_rate_m_yr = 0.00005  # 10 mm/yr

# loading_rate_m_yr = 0.03  # 30 mm/yr

m_accumulated = np.ones(config.n_elements) * loading_rate_m_yr * years_accumulated
m_current = np.maximum(m_accumulated - slip_coseismic, 0.0)

print("\nResidual moment field:")
print(
    f"  m_current inside rupture (r < {eq_radius_km / 2:.0f} km): {m_current[dist_from_eq < eq_radius_km / 2].mean():.3f} m"
)
print(
    f"  m_current at rupture edge (r ≈ {eq_radius_km} km): {m_current[np.abs(dist_from_eq - eq_radius_km) < 2].mean():.3f} m"
)
print(
    f"  m_current in halo (r > {eq_radius_km + 5:.0f} km): {m_current[dist_from_eq > eq_radius_km + 5].mean():.3f} m"
)

# =============================================================================
# INITIALIZE AFTERSLIP SEQUENCE
# =============================================================================

print("\nInitializing afterslip sequence...")
sequence = initialize_afterslip_sequence(event, m_current, mesh, config)

# Extract fields
Phi = sequence["Phi"]
v_initial = sequence["v_initial"]
m_residual_initial = sequence["m_residual_initial"]

print(f"\nAfterslip statistics:")
print(f"  Max Phi: {Phi.max():.3f}")
print(f"  Max v_initial: {v_initial.max():.4f} m/yr")
print(f"  Max m_residual: {m_residual_initial.max():.2f} m")
print(f"  Moment budget: {sequence['moment_budget']:.2e} m³")

# Find where peak v_initial occurs
peak_idx = np.argmax(v_initial)
peak_x = centroids[peak_idx, 0]
peak_z = centroids[peak_idx, 2]
peak_dist_from_center = dist_from_eq[peak_idx]

print(f"\nPeak v_initial location:")
print(f"  Position: x={peak_x:.1f} km, z={peak_z:.1f} km")
print(f"  Distance from rupture center: {peak_dist_from_center:.1f} km")
print(f"  Expected: ~{eq_radius_km} km (inner halo edge)")

# =============================================================================
# TIME EVOLUTION LOOP
# =============================================================================

print("\nRunning time evolution...")

# Time parameters - EXTENDED TO 100 YEARS FOR FULL CYCLE
max_time_years = 100.0
dt_days = 1.0
dt_years = dt_days / 365.25
snapshot_times = [0, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0]  # Years (for velocity plots)
snapshot_times_cycle = [0, 1, 10, 25, 50, 100]  # Years (for 100-year moment evolution)

# Wrap sequence in list for update function
sequences = [sequence]

# Storage for time series
history = {
    "times": [],
    "v_total": [],
    "v_max": [],
    "moment_released": [],
    "moment_fraction": [],
    "n_active_patches": [],
    "m_total_sum": [],  # NEW: Total moment integrated over fault
    "m_deficit_mean": [],  # NEW: Mean moment deficit across fault
    "snapshots": {t: {} for t in snapshot_times},
    "snapshots_cycle": {t: {} for t in snapshot_times_cycle},  # NEW: For 100-year cycle
}

# Sample locations for decay curves (indices)
sample_locations = {
    "Inside rupture (r=10km)": np.argmin(np.abs(dist_from_eq - 10.0)),
    "Rupture edge (r=20km)": np.argmin(np.abs(dist_from_eq - eq_radius_km)),
    "Halo (r=30km)": np.argmin(np.abs(dist_from_eq - 30.0)),
    "Far field (r=50km)": np.argmin(np.abs(dist_from_eq - 50.0)),
}
velocity_timeseries = {loc: [] for loc in sample_locations}

# Time evolution loop
current_time = 0.0
snapshot_idx = 0
snapshot_cycle_idx = 0

# Track tectonic loading accumulation (continuous throughout cycle)
m_tectonic_accumulated = np.zeros(config.n_elements)

while current_time <= max_time_years:
    # Compute current velocity field
    dt_since_event = current_time
    v_current = sequence["v_initial"] * np.exp(
        -sequence["decay_rates"] * dt_since_event
    )
    v_current[v_current < config.afterslip_v_min] = 0.0

    # Add continuous tectonic loading (inter-seismic strain accumulation)
    if current_time > 0:
        m_tectonic_accumulated += loading_rate_m_yr * dt_years

    # Total moment = residual from afterslip + tectonic loading
    m_total_current = sequence["m_residual_current"] + m_tectonic_accumulated

    # Store time series
    history["times"].append(current_time)
    history["v_total"].append(np.sum(v_current))
    history["v_max"].append(np.max(v_current))
    history["moment_released"].append(sequence["moment_released"])
    history["moment_fraction"].append(
        sequence["moment_released"] / sequence["moment_budget"]
    )
    history["n_active_patches"].append(np.sum(v_current > 0))
    history["m_total_sum"].append(np.sum(m_total_current))  # NEW
    history["m_deficit_mean"].append(np.mean(m_total_current))  # NEW

    # Store velocity at sample locations
    for loc, idx in sample_locations.items():
        velocity_timeseries[loc].append(v_current[idx])

    # Store spatial snapshots at specified times (for velocity plots)
    if snapshot_idx < len(snapshot_times) and np.isclose(
        current_time, snapshot_times[snapshot_idx], atol=dt_years / 2
    ):
        history["snapshots"][snapshot_times[snapshot_idx]] = {
            "v_current": v_current.copy(),
            "cumulative_slip": sequence["cumulative_afterslip"].copy(),
            "m_residual": sequence["m_residual_current"].copy(),
        }
        print(f"  Snapshot at t = {snapshot_times[snapshot_idx]:.1f} years")
        snapshot_idx += 1

    # Store spatial snapshots for 100-year cycle (moment evolution)
    if snapshot_cycle_idx < len(snapshot_times_cycle) and np.isclose(
        current_time, snapshot_times_cycle[snapshot_cycle_idx], atol=dt_years / 2
    ):
        history["snapshots_cycle"][snapshot_times_cycle[snapshot_cycle_idx]] = {
            "m_total": m_total_current.copy(),
            "m_tectonic": m_tectonic_accumulated.copy(),
            "cumulative_afterslip": sequence["cumulative_afterslip"].copy(),
            "v_current": v_current.copy(),
        }
        print(
            f"  Cycle snapshot at t = {snapshot_times_cycle[snapshot_cycle_idx]:.1f} years"
        )
        snapshot_cycle_idx += 1

    # Update afterslip sequences
    if current_time < max_time_years:
        moment_release = update_afterslip_sequences(
            sequences, current_time, dt_years, config
        )

    current_time += dt_years

# Convert to numpy arrays
for key in [
    "times",
    "v_total",
    "v_max",
    "moment_released",
    "moment_fraction",
    "n_active_patches",
    "m_total_sum",
    "m_deficit_mean",
]:
    history[key] = np.array(history[key])

for loc in velocity_timeseries:
    velocity_timeseries[loc] = np.array(velocity_timeseries[loc])

print(f"\nTime evolution complete.")
print(f"  Final moment released: {sequence['moment_released']:.2e} m³")
print(f"  Moment budget used: {history['moment_fraction'][-1] * 100:.1f}%")

# =============================================================================
# FIGURE 1: TIME SERIES
# =============================================================================

print("\nCreating Figure 1: Time series (100-year cycle)...")
fig, axes = plt.subplots(5, 1, figsize=(12, 15))

# Panel A: Total velocity vs time
axes[0].plot(history["times"], history["v_total"], "b-", linewidth=2)
axes[0].set_ylabel("Total Velocity (m/yr)", fontsize=12)
axes[0].set_title("100-Year Seismic Cycle Evolution", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)

# Panel B: Maximum velocity vs time (log scale)
axes[1].semilogy(history["times"], history["v_max"], "r-", linewidth=2)
axes[1].set_ylabel("Max Velocity (m/yr)", fontsize=12)
axes[1].grid(True, alpha=0.3, which="both")

# Panel C: Mean moment deficit (shows inter-seismic accumulation)
axes[2].plot(history["times"], history["m_deficit_mean"], "purple", linewidth=2)
axes[2].set_ylabel("Mean Moment Deficit (m)", fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].axhline(
    years_accumulated * loading_rate_m_yr,
    color="k",
    linestyle="--",
    alpha=0.5,
    label=f"Pre-earthquake level ({years_accumulated:.0f} yr)",
)
axes[2].legend()

# Panel D: Moment budget fraction (afterslip only)
axes[3].plot(history["times"], history["moment_fraction"], "g-", linewidth=2)
axes[3].set_ylabel("Afterslip Budget Fraction", fontsize=12)
axes[3].axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Full budget")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# Panel E: Number of active patches
axes[4].plot(history["times"], history["n_active_patches"], "m-", linewidth=2)
axes[4].set_ylabel("N Active Patches", fontsize=12)
axes[4].set_xlabel("Time (years)", fontsize=12)
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/test_afterslip_time_series.png", dpi=150, bbox_inches="tight")
print("  Saved: results/test_afterslip_time_series.png")
plt.close()

# =============================================================================
# FIGURE 2: SPATIAL VELOCITY EVOLUTION
# =============================================================================

print("Creating Figure 2: Spatial velocity evolution (100-year cycle)...")
fig, axes = plt.subplots(6, 1, figsize=(15, 18))
times_to_plot = snapshot_times_cycle  # Use cycle times: [0, 1, 10, 25, 50, 100]

nx = config.n_along_strike
ny = config.n_down_dip

# Create coordinate grids for contourf
length_vec = np.linspace(0, config.fault_length_km, nx)
depth_vec = np.linspace(0, config.fault_depth_km, ny)
length_grid, depth_grid = np.meshgrid(length_vec, depth_vec)

# We need to get velocity data for cycle snapshots - compute from stored m_total
# Velocity at time t is afterslip velocity (which we can reconstruct)
# For now, show coupling = (loading_rate - afterslip_velocity) / loading_rate

# Compute global color limits across all snapshots
epsilon = 1e-10  # Small value to avoid log(0)

# Get coupling fields for all cycle times
all_coupling_fields = []
for t in times_to_plot:
    # Compute afterslip velocity at this time
    dt_since_event = t
    v_afterslip = sequence["v_initial"] * np.exp(
        -sequence["decay_rates"] * dt_since_event
    )
    v_afterslip[v_afterslip < config.afterslip_v_min] = 0.0

    # Coupling = (tectonic loading rate - aseismic slip rate) / tectonic loading rate
    coupling = (loading_rate_m_yr - v_afterslip) / loading_rate_m_yr
    all_coupling_fields.append(coupling)

# Use linear scale for coupling (0 to 1)
global_vmin = -20.0
global_vmax = 1.0
global_levels = np.linspace(global_vmin, global_vmax, 11)

global_levels = np.linspace(-2, global_vmax, 51)


for idx, (ax, t, coupling_field) in enumerate(
    zip(axes, times_to_plot, all_coupling_fields)
):
    v_field = coupling_field.reshape(nx, ny).T

    # Power-law normalize
    v_field = np.sign(v_field) * (np.abs(v_field) ** 0.2)

    # Filled contours with global levels (linear scale for coupling)
    cbar_plot = ax.contourf(
        length_grid,
        depth_grid,
        v_field,
        cmap="RdYlGn",  # Red (low coupling) to Green (high coupling)
        levels=global_levels,
        extend="neither",
    )
    # Contour lines overlay
    ax.contour(
        length_grid,
        depth_grid,
        v_field,
        colors="black",
        linewidths=0.5,
        linestyles="solid",
        levels=global_levels,
    )

    ax.set_title(f"$t$ = {t} years", fontsize=12, fontweight="normal")
    ax.set_xlabel("$x$ (km)", fontsize=12)
    ax.set_ylabel("$d$ (km)", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Add rupture boundary
    circle = plt.Circle(
        (eq_center_x_km, eq_center_z_km),
        eq_radius_km,
        fill=False,
        edgecolor="blue",
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(circle)

    # Colorbar (matched height to subplot)
    cbar = plt.colorbar(cbar_plot, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$c$", fontsize=12)

plt.tight_layout()
plt.savefig(
    "results/test_afterslip_velocity_evolution.png", dpi=300, bbox_inches="tight"
)
print("  Saved: results/test_afterslip_velocity_evolution.png")
plt.close()

# =============================================================================
# FIGURE 3: CUMULATIVE SLIP EVOLUTION
# =============================================================================

print("Creating Figure 3: Cumulative afterslip evolution (100-year cycle)...")
fig, axes = plt.subplots(6, 1, figsize=(15, 18))

# Compute global color limits across all cycle snapshots
all_cum_slip = [
    history["snapshots_cycle"][t]["cumulative_afterslip"] for t in snapshot_times_cycle
]
global_vmin = min(c.min() for c in all_cum_slip)
global_vmax = max(c.max() for c in all_cum_slip)
if global_vmax - global_vmin < 1e-10:  # Handle case where all values are the same
    global_vmin, global_vmax = global_vmin - 0.01, global_vmax + 0.01
global_levels = np.linspace(global_vmin, global_vmax, 11)

for idx, (ax, t) in enumerate(zip(axes, snapshot_times_cycle)):
    cum_slip = history["snapshots_cycle"][t]["cumulative_afterslip"].reshape(nx, ny).T

    # Filled contours with global levels
    cbar_plot = ax.contourf(
        length_grid,
        depth_grid,
        cum_slip,
        cmap="viridis",
        levels=global_levels,
        extend="both",
    )
    # Contour lines overlay
    ax.contour(
        length_grid,
        depth_grid,
        cum_slip,
        colors="black",
        linewidths=0.5,
        linestyles="solid",
        levels=global_levels,
    )

    # Annotate phase of cycle
    if t == 0:
        phase = "Post-earthquake"
    elif t <= 10:
        phase = "Post-seismic (afterslip)"
    elif t <= 50:
        phase = "Inter-seismic"
    else:
        phase = "Late inter-seismic"

    ax.set_title(f"$t$ = {t} years - {phase}", fontsize=12, fontweight="normal")
    ax.set_xlabel("$x$ (km)", fontsize=12)
    ax.set_ylabel("$d$ (km)", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Add rupture boundary
    circle = plt.Circle(
        (eq_center_x_km, eq_center_z_km),
        eq_radius_km,
        fill=False,
        edgecolor="red",
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(circle)

    # Colorbar (matched height to subplot)
    cbar = plt.colorbar(cbar_plot, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative Afterslip (m)", fontsize=11)

plt.tight_layout()
plt.savefig(
    "results/test_afterslip_cumulative_evolution.png", dpi=300, bbox_inches="tight"
)
print("  Saved: results/test_afterslip_cumulative_evolution.png")
plt.close()

# =============================================================================
# FIGURE 4: RADIAL PROFILES OVER TIME
# =============================================================================

print("Creating Figure 4: Radial profiles over 100-year cycle...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Radial distances for profile
radial_distances = dist_from_eq
sort_idx = np.argsort(radial_distances)

# Panel 1: Coupling profiles
for t in snapshot_times_cycle:
    # Compute coupling field for this time
    dt_since_event = t
    v_afterslip = sequence["v_initial"] * np.exp(
        -sequence["decay_rates"] * dt_since_event
    )
    v_afterslip[v_afterslip < config.afterslip_v_min] = 0.0
    coupling = (loading_rate_m_yr - v_afterslip) / loading_rate_m_yr

    ax1.plot(
        radial_distances[sort_idx], coupling[sort_idx], linewidth=2, label=f"t = {t} yr"
    )

ax1.axvline(
    eq_radius_km,
    color="k",
    linestyle="--",
    linewidth=2,
    alpha=0.5,
    label="Rupture edge",
)
ax1.set_xlabel("Distance from center (km)", fontsize=12)
ax1.set_ylabel("Fault Coupling", fontsize=12)
ax1.set_title("Coupling vs Radius", fontsize=14, fontweight="bold")
ax1.set_ylim([0, 1.1])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Cumulative afterslip profiles
for t in snapshot_times_cycle:
    cum_field = history["snapshots_cycle"][t]["cumulative_afterslip"]
    ax2.plot(
        radial_distances[sort_idx],
        cum_field[sort_idx],
        linewidth=2,
        label=f"t = {t} yr",
    )

ax2.axvline(
    eq_radius_km,
    color="k",
    linestyle="--",
    linewidth=2,
    alpha=0.5,
    label="Rupture edge",
)
ax2.set_xlabel("Distance from center (km)", fontsize=12)
ax2.set_ylabel("Cumulative Afterslip (m)", fontsize=12)
ax2.set_title("Cumulative Afterslip vs Radius", fontsize=14, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Total moment deficit profiles
for t in snapshot_times_cycle:
    m_total_field = history["snapshots_cycle"][t]["m_total"]
    ax3.plot(
        radial_distances[sort_idx],
        m_total_field[sort_idx],
        linewidth=2,
        label=f"t = {t} yr",
    )

ax3.axvline(
    eq_radius_km,
    color="k",
    linestyle="--",
    linewidth=2,
    alpha=0.5,
    label="Rupture edge",
)
ax3.set_xlabel("Distance from center (km)", fontsize=12)
ax3.set_ylabel("Moment Deficit (m)", fontsize=12)
ax3.set_title("Moment Deficit vs Radius", fontsize=14, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/test_afterslip_radial_profiles.png", dpi=150, bbox_inches="tight")
print("  Saved: results/test_afterslip_radial_profiles.png")
plt.close()

# =============================================================================
# FIGURE 5: VELOCITY DECAY CURVES AT SAMPLE LOCATIONS
# =============================================================================

print("Creating Figure 5: Velocity decay curves...")
fig, ax = plt.subplots(figsize=(12, 7))

for loc in sample_locations:
    ax.semilogy(history["times"], velocity_timeseries[loc], linewidth=2, label=loc)

ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Velocity (m/yr)", fontsize=12)
ax.set_title("Velocity Decay at Sample Locations", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("results/test_afterslip_decay_curves.png", dpi=150, bbox_inches="tight")
print("  Saved: results/test_afterslip_decay_curves.png")
plt.close()

# =============================================================================
# FIGURE 6: 100-YEAR MOMENT EVOLUTION (SEISMIC CYCLE)
# =============================================================================

print("Creating Figure 6: 100-year moment evolution (seismic cycle)...")
fig, axes = plt.subplots(6, 1, figsize=(15, 18))

# Compute global color limits across all cycle snapshots
all_m_total = [history["snapshots_cycle"][t]["m_total"] for t in snapshot_times_cycle]
global_vmin = min(m.min() for m in all_m_total)
global_vmax = max(m.max() for m in all_m_total)
if global_vmax - global_vmin < 1e-10:  # Handle case where all values are the same
    global_vmin, global_vmax = global_vmin - 0.1, global_vmax + 0.1
global_levels = np.linspace(global_vmin, global_vmax, 11)

for idx, (ax, t) in enumerate(zip(axes, snapshot_times_cycle)):
    m_total_field = history["snapshots_cycle"][t]["m_total"].reshape(nx, ny).T

    # Filled contours with global levels
    cbar_plot = ax.contourf(
        length_grid,
        depth_grid,
        m_total_field,
        cmap="viridis",
        levels=global_levels,
        extend="both",
    )
    # Contour lines overlay
    ax.contour(
        length_grid,
        depth_grid,
        m_total_field,
        colors="black",
        linewidths=0.5,
        linestyles="solid",
        levels=global_levels,
    )

    # Annotate phase of cycle
    if t == 0:
        phase = "Post-earthquake (coseismic release)"
    elif t <= 10:
        phase = "Post-seismic (afterslip dominant)"
    elif t <= 50:
        phase = "Inter-seismic (tectonic loading)"
    else:
        phase = "Late inter-seismic (approaching next event)"

    ax.set_title(f"$t$ = {t} years - {phase}", fontsize=12, fontweight="normal")
    ax.set_xlabel("$x$ (km)", fontsize=12)
    ax.set_ylabel("$d$ (km)", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Add rupture boundary
    circle = plt.Circle(
        (eq_center_x_km, eq_center_z_km),
        eq_radius_km,
        fill=False,
        edgecolor="red",
        linewidth=2,
        linestyle="--",
        label="Coseismic rupture" if idx == 0 else None,
    )
    ax.add_patch(circle)

    # Colorbar (matched height to subplot)
    cbar = plt.colorbar(cbar_plot, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Moment Deficit (m)", fontsize=11)

plt.tight_layout()
plt.savefig(
    "results/test_afterslip_cycle_moment_evolution.png", dpi=300, bbox_inches="tight"
)
print("  Saved: results/test_afterslip_cycle_moment_evolution.png")
plt.close()

print("\n" + "=" * 70)
print("ALL FIGURES COMPLETE")
print("=" * 70)
print("Generated 6 diagnostic plots showing full 100-year cycle:")
print("  1. test_afterslip_time_series.png - Time series of key quantities")
print("  2. test_afterslip_velocity_evolution.png - Spatial velocity at 6 times")
print("  3. test_afterslip_cumulative_evolution.png - Cumulative slip at 6 times")
print("  4. test_afterslip_radial_profiles.png - Radial profiles over time")
print("  5. test_afterslip_decay_curves.png - Decay curves at sample locations")
print("  6. test_afterslip_cycle_moment_evolution.png - 100-year moment evolution")
print("\nShowing transition from post-seismic afterslip to inter-seismic loading")
print(f"  Loading rate: {loading_rate_m_yr * 1000:.1f} mm/yr")
print(f"  Final moment integrated over fault: {history['m_total_sum'][-1]:.2e} m³")
print(f"  Mean moment deficit at t=100yr: {history['m_deficit_mean'][-1]:.3f} m")
print("=" * 70)
