import numpy as np
import matplotlib.pyplot as plt

# from config import Config
from geometry import create_fault_mesh
from afterslip import initialize_afterslip_sequence, update_afterslip_sequences


class Config:
    """Simulation configuration"""

    # === GEOMETRY ===
    fault_length_km = 200.0  # Along-strike length (km)
    fault_depth_km = 25.0  # Down-dip depth (km)
    element_size_km = 1  # Grid cell size (km)
    # element_size_km = 0.1  # Grid cell size (km)

    # === MOMENT ACCUMULATION ===
    # Background slip deficit rate
    background_slip_rate_mm_yr = 10.0  # mm/year

    # Gaussian moment pulse(s) - list of dicts
    moment_pulses = [
        {
            "center_x_km": 100.0,  # Along-strike position
            "center_z_km": 12.5,  # Down-dip position
            "sigma_km": 20.0,  # Width of Gaussian
            "amplitude_mm_yr": 0.0,  # Additional slip rate at center
        }
    ]

    # === PHYSICAL PARAMETERS ===
    shear_modulus_Pa = 3e10  # Pa (30 GPa)

    # === MAGNITUDE-DEPENDENT SPATIAL PROBABILITY ===
    gamma_min = 0.5  # Small events (SOC)
    gamma_max = 1.5  # Large events (G-R)
    alpha_spatial = 0.35  # Decay rate
    M_min = 5.0  # Minimum magnitude
    M_max = 8.0  # Maximum magnitude

    # === ADAPTIVE RATE CORRECTION ===
    adaptive_correction_enabled = (
        True  # Enable adaptive correction (True = drives coupling toward 1.0)
    )
    adaptive_correction_gain = (
        5.0  # Proportional control gain for coupling correction (continuous updates)
    )
    correction_factor_min = 0.1  # Minimum allowed correction factor
    correction_factor_max = 10.0  # Maximum allowed correction factor

    # === OMORI AFTERSHOCK PARAMETERS ===
    # Standard Omori-Utsu law: λ_aftershock(t) = K / (t + c)^p
    # where K scales with mainshock magnitude: K = K_ref × 10^(alpha × (M - M_ref))
    omori_enabled = True  # Enable/disable aftershock sequences
    omori_p = 1.0  # Decay exponent (typically ~1.0)
    omori_c_years = 1.0 / 365.25  # Time offset in years (~0.00274 years = 1 day)
    omori_K_ref = 0.1  # Productivity (events/year) for M=6 mainshock
    omori_M_ref = 6.0  # Reference magnitude for productivity
    omori_alpha = 0.8  # Magnitude scaling (Reasenberg & Jones 1989)
    omori_duration_years = (
        10.0  # Only track aftershocks for this many years after mainshock
    )

    # === BACKGROUND RATE ===
    lambda_background = 0.0  # Constant background rate (events/year, default disabled)

    # === RANDOM PERTURBATIONS ===
    perturbation_type = "none"  # Options: "none", "white_noise", "ornstein_uhlenbeck"
    perturbation_sigma = (
        0.01  # White noise: std dev (events/yr); OU: diffusion coefficient
    )
    perturbation_mean = 0.0  # OU process only: mean perturbation level (events/yr)
    perturbation_theta = 1.0  # OU process only: reversion rate (1/years)

    # === AFTERSLIP PARAMETERS ===
    # MaxEnt afterslip model: aseismic creep following coseismic events
    # Spatial activation Φ(x,y) also controls aftershock localization
    afterslip_enabled = True  # Enable/disable afterslip physics
    afterslip_v_ref_m_yr = 0.5  # Reference initial velocity (m/yr) at M_ref
    afterslip_M_ref = 7.0  # Reference magnitude for velocity scaling
    afterslip_beta = 0.33  # Magnitude scaling exponent (1/3 from MaxEnt theory)
    afterslip_correlation_length_x_km = (
        5.0  # Spatial correlation length ξ_x (along-strike)
    )
    afterslip_correlation_length_z_km = 5.0  # Spatial correlation length ξ_z (down-dip)
    afterslip_kernel_type = "exponential"  # 'exponential' or 'power_law'
    afterslip_power_law_exponent = 2.5  # Exponent if using power_law kernel
    afterslip_duration_years = 10.0  # Track sequences for this many years
    afterslip_m_critical = 0.01  # Minimum residual moment for afterslip (m)
    afterslip_v_min = 1e-6  # Minimum velocity for numerical stability (m/yr)
    afterslip_M_min = 6.0  # Only trigger afterslip for M ≥ this threshold [RAISED]
    afterslip_spatial_threshold = (
        0.3  # Only allow afterslip where Phi > threshold [NEW]
    )

    # === MOMENT INITIALIZATION ===
    spinup_fraction = 0.25  # Initialize with this fraction of mid-cycle moment (0.25 = recurrence_time/4)

    # === SLIP DISTRIBUTION ===
    slip_decay_rate = 2.0  # Exponential decay rate of slip from hypocenter
    slip_heterogeneity = 0.3  # Random perturbation amplitude (±30%)

    # === GUTENBERG-RICHTER ===
    b_value = 1.0

    # === SIMULATION ===
    duration_years = 1000.0  # Full simulation duration
    time_step_years = 1.0  # Time resolution (years)

    # Random seed for reproducibility
    random_seed = 42

    # === OUTPUT ===
    output_dir = "results"
    output_hdf5 = "simulation_results.h5"
    hdf5_compression = (
        0  # gzip compression level (0=none for speed, 4=balanced, 9=max compression)
    )
    snapshot_interval_years = (
        1.0  # Save moment snapshots every N years (1.0 = every timestep)
    )

    def compute_derived_parameters(self):
        """Compute parameters that depend on others"""
        # Grid dimensions
        self.n_along_strike = int(self.fault_length_km / self.element_size_km)
        self.n_down_dip = int(self.fault_depth_km / self.element_size_km)
        self.n_elements = self.n_along_strike * self.n_down_dip

        # Element area
        self.element_area_m2 = (self.element_size_km * 1000) ** 2

        # Time steps
        self.n_time_steps = int(self.duration_years / self.time_step_years)

        # Convert slip rates to m/year
        self.background_slip_rate_m_yr = self.background_slip_rate_mm_yr / 1000.0

        # Total moment accumulation rate (N·m/year)
        total_slip_rate = self.background_slip_rate_m_yr * self.n_elements
        self.total_moment_rate = (
            self.shear_modulus_Pa * self.element_area_m2 * total_slip_rate
        )

        print(
            f"Grid: {self.n_along_strike} x {self.n_down_dip} = {self.n_elements} elements"
        )
        print(f"Total moment rate: {self.total_moment_rate:.2e} N·m/year")

    def to_dict(self):
        """
        Serialize configuration to dictionary for HDF5 storage

        Returns:
        --------
        config_dict : dict
            Dictionary of all config parameters (excludes methods and private attrs)
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            # Skip private attributes and methods
            if key.startswith("_") or callable(value):
                continue
            # Store all simple types
            config_dict[key] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        """
        Reconstruct Config object from dictionary

        Parameters:
        -----------
        config_dict : dict
            Dictionary of config parameters

        Returns:
        --------
        config : Config
            Reconstructed configuration object
        """
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config


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

print(f"Test earthquake:")
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
loading_rate_m_yr = 0.03  # 30 mm/yr

m_accumulated = np.ones(config.n_elements) * loading_rate_m_yr * years_accumulated
m_current = np.maximum(m_accumulated - slip_coseismic, 0.0)

print(f"\nResidual moment field:")
print(
    f"  m_current inside rupture (r < {eq_radius_km / 2:.0f} km): {m_current[dist_from_eq < eq_radius_km / 2].mean():.3f} m"
)
print(
    f"  m_current at rupture edge (r ≈ {eq_radius_km} km): {m_current[np.abs(dist_from_eq - eq_radius_km) < 2].mean():.3f} m"
)
print(
    f"  m_current in halo (r > {eq_radius_km + 5:.0f} km): {m_current[dist_from_eq > eq_radius_km + 5].mean():.3f} m"
)

# INITIALIZE AFTERSLIP SEQUENCE
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

# TIME EVOLUTION LOOP
print("\nRunning time evolution...")

# Time parameters
max_time_years = 10.0
max_time_years = 25.0
dt_days = 1.0
dt_years = dt_days / 365.25
snapshot_times = [0, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0]  # Years
# snapshot_times = [0, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0]  # Years

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
    "snapshots": {t: {} for t in snapshot_times},
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

while current_time <= max_time_years:
    # Compute current velocity field
    dt_since_event = current_time
    v_current = sequence["v_initial"] * np.exp(
        -sequence["decay_rates"] * dt_since_event
    )
    v_current[v_current < config.afterslip_v_min] = 0.0

    # Store time series
    history["times"].append(current_time)
    history["v_total"].append(np.sum(v_current))
    history["v_max"].append(np.max(v_current))
    history["moment_released"].append(sequence["moment_released"])
    history["moment_fraction"].append(
        sequence["moment_released"] / sequence["moment_budget"]
    )
    history["n_active_patches"].append(np.sum(v_current > 0))

    # Store velocity at sample locations
    for loc, idx in sample_locations.items():
        velocity_timeseries[loc].append(v_current[idx])

    # Store spatial snapshots at specified times
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
]:
    history[key] = np.array(history[key])

for loc in velocity_timeseries:
    velocity_timeseries[loc] = np.array(velocity_timeseries[loc])

print(f"\nTime evolution complete.")
print(f"  Final moment released: {sequence['moment_released']:.2e} m³")
print(f"  Moment budget used: {history['moment_fraction'][-1] * 100:.1f}%")


# # FIGURE 1: TIME SERIES
# print("\nCreating Figure 1: Time series...")
# fig, axes = plt.subplots(4, 1, figsize=(12, 12))

# # Panel A: Total velocity vs time
# axes[0].plot(history["times"], history["v_total"], "b-", linewidth=2)
# axes[0].set_ylabel("Total Velocity (m/yr)", fontsize=12)
# axes[0].set_title("Afterslip Time Evolution", fontsize=14, fontweight="bold")
# axes[0].grid(True, alpha=0.3)

# # Panel B: Maximum velocity vs time (log scale)
# axes[1].semilogy(history["times"], history["v_max"], "r-", linewidth=2)
# axes[1].set_ylabel("Max Velocity (m/yr)", fontsize=12)
# axes[1].grid(True, alpha=0.3, which="both")

# # Panel C: Moment budget fraction
# axes[2].plot(history["times"], history["moment_fraction"], "g-", linewidth=2)
# axes[2].set_ylabel("Moment Budget Fraction", fontsize=12)
# axes[2].axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Full budget")
# axes[2].legend()
# axes[2].grid(True, alpha=0.3)

# # Panel D: Number of active patches
# axes[3].plot(history["times"], history["n_active_patches"], "m-", linewidth=2)
# axes[3].set_ylabel("N Active Patches", fontsize=12)
# axes[3].set_xlabel("Time (years)", fontsize=12)
# axes[3].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig("results/test_afterslip_time_series.png", dpi=150, bbox_inches="tight")
# print("  Saved: results/test_afterslip_time_series.png")
# plt.close()

FONTSIZE = 10

# FIGURE 2: SPATIAL VELOCITY EVOLUTION
print("Creating Figure 2: Spatial velocity evolution...")
fig, axes = plt.subplots(6, 1, figsize=(10, 10))
times_to_plot = [0, 0.1, 1.0, 3.0, 5.0, 10.0]
# times_to_plot = [0, 0.5, 1.0, 5.0, 10.0, 25.0]

nx = config.n_along_strike
ny = config.n_down_dip

# Create coordinate grids for contourf
length_vec = np.linspace(0, config.fault_length_km, nx)
depth_vec = np.linspace(0, config.fault_depth_km, ny)
length_grid, depth_grid = np.meshgrid(length_vec, depth_vec)

# Compute global color limits across all snapshots (log10 scale)
epsilon = 1e-10  # Small value to avoid log(0)
all_v_fields = [history["snapshots"][t]["v_current"] for t in times_to_plot]
all_v_log = [np.log10(v + epsilon) for v in all_v_fields]
global_vmin_log = min(v.min() for v in all_v_log)
global_vmax_log = max(v.max() for v in all_v_log)
if (
    global_vmax_log - global_vmin_log < 1e-10
):  # Handle case where all values are the same
    global_vmin_log, global_vmax_log = global_vmin_log - 0.1, global_vmax_log + 0.1
global_levels_log = np.linspace(global_vmin_log, global_vmax_log, 51)
global_levels_log = np.linspace(-2, -0.5, 5)


for idx, (ax, t) in enumerate(zip(axes, times_to_plot)):
    v_field = history["snapshots"][t]["v_current"].reshape(nx, ny).T
    v_field_log = np.log10(v_field + epsilon)

    # Filled contours with global levels (log scale)
    cbar_plot = ax.contourf(
        length_grid,
        depth_grid,
        v_field_log,
        # cmap="YlOrRd",
        cmap="plasma",
        levels=global_levels_log,
        extend="both",
    )
    # Contour lines overlay
    ax.contour(
        length_grid,
        depth_grid,
        v_field_log,
        colors="black",
        linewidths=0.5,
        linestyles="solid",
        levels=global_levels_log,
    )

    ax.set_title(f"$t$ = {t:0.1f} years", fontsize=FONTSIZE, fontweight="normal")
    ax.set_xlabel("$x$ (km)", fontsize=FONTSIZE)
    ax.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Add rupture boundary
    circle = plt.Circle(
        (eq_center_x_km, eq_center_z_km),
        eq_radius_km,
        fill=False,
        edgecolor="lightgray",
        linewidth=5,
        linestyle="-",
        zorder=10,
    )
    ax.add_patch(circle)

    # Colorbar (matched height to subplot)
    cbar = plt.colorbar(cbar_plot, ax=ax, fraction=0.005, pad=0.03)
    cbar.set_label("log$_{10} \\; v$ (m/yr)", fontsize=FONTSIZE)

plt.tight_layout()
plt.savefig(
    "results/test_afterslip_velocity_evolution.png", dpi=500, bbox_inches="tight"
)
print("  Saved: results/test_afterslip_velocity_evolution.png")
plt.close()

# =============================================================================
# FIGURE 3: CUMULATIVE SLIP EVOLUTION
# =============================================================================

print("Creating Figure 3: Cumulative slip evolution...")
fig, axes = plt.subplots(6, 1, figsize=(10, 10))

# Compute global color limits across all snapshots
all_cum_slip = [history["snapshots"][t]["cumulative_slip"] for t in times_to_plot]
global_vmin = min(c.min() for c in all_cum_slip)
global_vmax = max(c.max() for c in all_cum_slip)
if global_vmax - global_vmin < 1e-10:  # Handle case where all values are the same
    global_vmin, global_vmax = global_vmin - 0.01, global_vmax + 0.01
global_levels = np.linspace(global_vmin, global_vmax, 11)

for idx, (ax, t) in enumerate(zip(axes, times_to_plot)):
    cum_slip = history["snapshots"][t]["cumulative_slip"].reshape(nx, ny).T

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

    ax.set_title(f"$t$ = {t:0.1f} years", fontsize=FONTSIZE, fontweight="normal")
    ax.set_xlabel("$x$ (km)", fontsize=FONTSIZE)
    ax.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Add rupture boundary
    circle = plt.Circle(
        (eq_center_x_km, eq_center_z_km),
        eq_radius_km,
        fill=False,
        edgecolor="lightgray",
        linewidth=5,
        linestyle="-",
        zorder=10,
    )
    ax.add_patch(circle)

    # Colorbar (matched height to subplot)
    cbar = plt.colorbar(cbar_plot, ax=ax, fraction=0.005, pad=0.03)
    cbar.set_label("$s$ (m)", fontsize=FONTSIZE)

plt.tight_layout()
plt.savefig(
    "results/test_afterslip_cumulative_evolution.png", dpi=500, bbox_inches="tight"
)
# plt.savefig("results/test_afterslip_cumulative_evolution.pdf", bbox_inches="tight")
print("  Saved: results/test_afterslip_cumulative_evolution.png")
plt.close()


### RADIAL PROFILES OVER TIME
FONTSIZE = 10
LINEWIDTH = 0.5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Radial distances for profile
radial_distances = dist_from_eq
sort_idx = np.argsort(radial_distances)

# Panel 1: Velocity profiles
for t in times_to_plot:
    v_field = history["snapshots"][t]["v_current"]
    ax1.plot(
        radial_distances[sort_idx],
        v_field[sort_idx],
        linewidth=LINEWIDTH,
        label=f"$t$ = {t:0.1f} yr",
    )
ax1.fill(
    [0, eq_radius_km, eq_radius_km, 0],
    [0.0, 0.0, 0.8, 0.8],
    color="lightgray",
    label="rupture zone",
)
ax1.set_xlabel("$r$ (km)", fontsize=FONTSIZE)
ax1.set_ylabel("$v$ (m/yr)", fontsize=FONTSIZE)
ax1.set_xlim([0, 100])
ax1.set_ylim([0, 0.8])
ax1.set_xticks([0, 50, 100])
ax1.set_yticks([0.0, 0.4, 0.8])
ax1.legend()
ax1.grid(False)

# Panel 2: Cumulative slip profiles
for t in times_to_plot:
    cum_field = history["snapshots"][t]["cumulative_slip"]
    ax2.plot(
        radial_distances[sort_idx],
        cum_field[sort_idx],
        linewidth=LINEWIDTH,
        label=f"$t$ = {t:0.1f} yr",
    )
ax2.fill(
    [0, eq_radius_km, eq_radius_km, 0],
    [0.0, 0.0, 1.6, 1.6],
    color="lightgray",
    label="rupture zone",
)
ax2.set_xlabel("$r$ (km)", fontsize=FONTSIZE)
ax2.set_ylabel("$s$ (m)", fontsize=FONTSIZE)
ax2.set_xlim([0, 100])
ax2.set_ylim([0, 1.6])
ax2.set_xticks([0, 50, 100])
ax2.set_yticks([0.0, 0.8, 1.6])
ax2.legend()
ax2.grid(False)

plt.tight_layout()
plt.savefig("results/test_afterslip_radial_profiles.png", dpi=500, bbox_inches="tight")
plt.savefig("results/test_afterslip_radial_profiles.pdf")
plt.close()

# # =============================================================================
# # FIGURE 5: VELOCITY DECAY CURVES AT SAMPLE LOCATIONS
# # =============================================================================

# print("Creating Figure 5: Velocity decay curves...")
# fig, ax = plt.subplots(figsize=(12, 7))

# for loc in sample_locations:
#     ax.semilogy(history["times"], velocity_timeseries[loc], linewidth=2, label=loc)

# ax.set_xlabel("Time (years)", fontsize=12)
# ax.set_ylabel("Velocity (m/yr)", fontsize=12)
# ax.set_title("Velocity Decay at Sample Locations", fontsize=14, fontweight="bold")
# ax.legend(fontsize=11)
# ax.grid(True, alpha=0.3, which="both")

# plt.tight_layout()
# plt.savefig("results/test_afterslip_decay_curves.png", dpi=150, bbox_inches="tight")
# print("  Saved: results/test_afterslip_decay_curves.png")
# plt.close()

# print("\n" + "=" * 70)
# print("ALL FIGURES COMPLETE")
# print("=" * 70)
# print("Generated 5 diagnostic plots showing temporal evolution:")
# print("  1. test_afterslip_time_series.png - Time series of key quantities")
# print("  2. test_afterslip_velocity_evolution.png - Spatial velocity at 6 times")
# print("  3. test_afterslip_cumulative_evolution.png - Cumulative slip at 6 times")
# print("  4. test_afterslip_radial_profiles.png - Radial profiles over time")
# print("  5. test_afterslip_decay_curves.png - Decay curves at sample locations")
# print("=" * 70)
