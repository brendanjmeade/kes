"""
Standalone validation test for single afterslip sequence - TEMPORAL EVOLUTION

This script tests the time evolution of afterslip for a single coseismic event.
Shows how velocity decays, cumulative slip accumulates, and moment depletes
over a 10-year afterslip period.

Validates:
- Exponential velocity decay with spatially-varying rates
- Cumulative slip accumulation
- Moment budget tracking and depletion
- Self-limiting behavior

Generates 5 diagnostic plots showing temporal evolution.
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
eq_center_z_km = config.fault_depth_km / 2   # 12.5 km
eq_radius_km = 20.0  # 20 km radius
eq_slip_m = 1.0  # 1 meter slip

# Calculate distance from earthquake center for each element
centroids = mesh['centroids']
dist_from_eq = np.sqrt(
    (centroids[:, 0] - eq_center_x_km)**2 +
    (centroids[:, 2] - eq_center_z_km)**2
)

# Create coseismic slip array (circular rupture)
slip_coseismic = np.zeros(config.n_elements)
slip_coseismic[dist_from_eq <= eq_radius_km] = eq_slip_m

# Find ruptured elements
ruptured_elements = np.where(dist_from_eq <= eq_radius_km)[0]

# Calculate magnitude (rough estimate)
area_co_m2 = len(ruptured_elements) * config.element_area_m2
M0_Nm = eq_slip_m * area_co_m2 * config.shear_modulus_Pa
magnitude = (2/3) * np.log10(M0_Nm) - 6.07

print(f"Test earthquake:")
print(f"  Magnitude: M {magnitude:.2f}")
print(f"  Rupture area: {area_co_m2/1e6:.1f} km²")
print(f"  Number of ruptured elements: {len(ruptured_elements)}")
print(f"  Rupture radius: {eq_radius_km} km")

# Create event dict (as would be passed from simulator)
event = {
    'ruptured_elements': ruptured_elements,
    'slip': slip_coseismic,
    'magnitude': magnitude,
    'time': 0.0,
}

# m_current represents residual moment after coseismic release
# CRITICAL: Drives spatial pattern of afterslip
# Simulate mid-cycle loading (e.g., 50 years at 10 mm/yr)
years_accumulated = 50.0
loading_rate_m_yr = 0.01  # 10 mm/yr
m_accumulated = np.ones(config.n_elements) * loading_rate_m_yr * years_accumulated
m_current = np.maximum(m_accumulated - slip_coseismic, 0.0)

print(f"\nResidual moment field:")
print(f"  m_current inside rupture (r < {eq_radius_km/2:.0f} km): {m_current[dist_from_eq < eq_radius_km/2].mean():.3f} m")
print(f"  m_current at rupture edge (r ≈ {eq_radius_km} km): {m_current[np.abs(dist_from_eq - eq_radius_km) < 2].mean():.3f} m")
print(f"  m_current in halo (r > {eq_radius_km+5:.0f} km): {m_current[dist_from_eq > eq_radius_km + 5].mean():.3f} m")

# =============================================================================
# INITIALIZE AFTERSLIP SEQUENCE
# =============================================================================

print("\nInitializing afterslip sequence...")
sequence = initialize_afterslip_sequence(event, m_current, mesh, config)

# Extract fields
Phi = sequence['Phi']
v_initial = sequence['v_initial']
m_residual_initial = sequence['m_residual_initial']

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

# Time parameters
max_time_years = 10.0
dt_days = 1.0
dt_years = dt_days / 365.25
snapshot_times = [0, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0]  # Years

# Wrap sequence in list for update function
sequences = [sequence]

# Storage for time series
history = {
    'times': [],
    'v_total': [],
    'v_max': [],
    'moment_released': [],
    'moment_fraction': [],
    'n_active_patches': [],
    'snapshots': {t: {} for t in snapshot_times},
}

# Sample locations for decay curves (indices)
sample_locations = {
    'Inside rupture (r=10km)': np.argmin(np.abs(dist_from_eq - 10.0)),
    'Rupture edge (r=20km)': np.argmin(np.abs(dist_from_eq - eq_radius_km)),
    'Halo (r=30km)': np.argmin(np.abs(dist_from_eq - 30.0)),
    'Far field (r=50km)': np.argmin(np.abs(dist_from_eq - 50.0)),
}
velocity_timeseries = {loc: [] for loc in sample_locations}

# Time evolution loop
current_time = 0.0
snapshot_idx = 0

while current_time <= max_time_years:
    # Compute current velocity field
    dt_since_event = current_time
    v_current = sequence['v_initial'] * np.exp(-sequence['decay_rates'] * dt_since_event)
    v_current[v_current < config.afterslip_v_min] = 0.0

    # Store time series
    history['times'].append(current_time)
    history['v_total'].append(np.sum(v_current))
    history['v_max'].append(np.max(v_current))
    history['moment_released'].append(sequence['moment_released'])
    history['moment_fraction'].append(sequence['moment_released'] / sequence['moment_budget'])
    history['n_active_patches'].append(np.sum(v_current > 0))

    # Store velocity at sample locations
    for loc, idx in sample_locations.items():
        velocity_timeseries[loc].append(v_current[idx])

    # Store spatial snapshots at specified times
    if snapshot_idx < len(snapshot_times) and np.isclose(current_time, snapshot_times[snapshot_idx], atol=dt_years/2):
        history['snapshots'][snapshot_times[snapshot_idx]] = {
            'v_current': v_current.copy(),
            'cumulative_slip': sequence['cumulative_afterslip'].copy(),
            'm_residual': sequence['m_residual_current'].copy(),
        }
        print(f"  Snapshot at t = {snapshot_times[snapshot_idx]:.1f} years")
        snapshot_idx += 1

    # Update afterslip sequences
    if current_time < max_time_years:
        moment_release = update_afterslip_sequences(sequences, current_time, dt_years, config)

    current_time += dt_years

# Convert to numpy arrays
for key in ['times', 'v_total', 'v_max', 'moment_released', 'moment_fraction', 'n_active_patches']:
    history[key] = np.array(history[key])

for loc in velocity_timeseries:
    velocity_timeseries[loc] = np.array(velocity_timeseries[loc])

print(f"\nTime evolution complete.")
print(f"  Final moment released: {sequence['moment_released']:.2e} m³")
print(f"  Moment budget used: {history['moment_fraction'][-1]*100:.1f}%")

# =============================================================================
# FIGURE 1: TIME SERIES
# =============================================================================

print("\nCreating Figure 1: Time series...")
fig, axes = plt.subplots(4, 1, figsize=(12, 12))

# Panel A: Total velocity vs time
axes[0].plot(history['times'], history['v_total'], 'b-', linewidth=2)
axes[0].set_ylabel('Total Velocity (m/yr)', fontsize=12)
axes[0].set_title('Afterslip Time Evolution', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Panel B: Maximum velocity vs time (log scale)
axes[1].semilogy(history['times'], history['v_max'], 'r-', linewidth=2)
axes[1].set_ylabel('Max Velocity (m/yr)', fontsize=12)
axes[1].grid(True, alpha=0.3, which='both')

# Panel C: Moment budget fraction
axes[2].plot(history['times'], history['moment_fraction'], 'g-', linewidth=2)
axes[2].set_ylabel('Moment Budget Fraction', fontsize=12)
axes[2].axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Full budget')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Panel D: Number of active patches
axes[3].plot(history['times'], history['n_active_patches'], 'm-', linewidth=2)
axes[3].set_ylabel('N Active Patches', fontsize=12)
axes[3].set_xlabel('Time (years)', fontsize=12)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/test_afterslip_time_series.png', dpi=150, bbox_inches='tight')
print("  Saved: results/test_afterslip_time_series.png")
plt.close()

# =============================================================================
# FIGURE 2: SPATIAL VELOCITY EVOLUTION
# =============================================================================

print("Creating Figure 2: Spatial velocity evolution...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
times_to_plot = [0, 0.1, 1.0, 3.0, 5.0, 10.0]

nx = config.n_along_strike
ny = config.n_down_dip
extent = [0, config.fault_length_km, 0, config.fault_depth_km]

for idx, (ax, t) in enumerate(zip(axes.flat, times_to_plot)):
    v_field = history['snapshots'][t]['v_current'].reshape(nx, ny).T
    im = ax.imshow(v_field, aspect='equal', origin='lower', extent=extent, cmap='YlOrRd')
    ax.set_title(f't = {t} years', fontsize=14, fontweight='bold')
    ax.set_xlabel('Along-strike (km)')
    ax.set_ylabel('Down-dip (km)')
    # Add rupture boundary
    circle = plt.Circle((eq_center_x_km, eq_center_z_km), eq_radius_km,
                        fill=False, edgecolor='blue', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    plt.colorbar(im, ax=ax, label='Velocity (m/yr)')

plt.tight_layout()
plt.savefig('results/test_afterslip_velocity_evolution.png', dpi=150, bbox_inches='tight')
print("  Saved: results/test_afterslip_velocity_evolution.png")
plt.close()

# =============================================================================
# FIGURE 3: CUMULATIVE SLIP EVOLUTION
# =============================================================================

print("Creating Figure 3: Cumulative slip evolution...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, (ax, t) in enumerate(zip(axes.flat, times_to_plot)):
    cum_slip = history['snapshots'][t]['cumulative_slip'].reshape(nx, ny).T
    im = ax.imshow(cum_slip, aspect='equal', origin='lower', extent=extent, cmap='viridis')
    ax.set_title(f't = {t} years', fontsize=14, fontweight='bold')
    ax.set_xlabel('Along-strike (km)')
    ax.set_ylabel('Down-dip (km)')
    # Add rupture boundary
    circle = plt.Circle((eq_center_x_km, eq_center_z_km), eq_radius_km,
                        fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    plt.colorbar(im, ax=ax, label='Cumulative Slip (m)')

plt.tight_layout()
plt.savefig('results/test_afterslip_cumulative_evolution.png', dpi=150, bbox_inches='tight')
print("  Saved: results/test_afterslip_cumulative_evolution.png")
plt.close()

# =============================================================================
# FIGURE 4: RADIAL PROFILES OVER TIME
# =============================================================================

print("Creating Figure 4: Radial profiles over time...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Radial distances for profile
radial_distances = dist_from_eq
sort_idx = np.argsort(radial_distances)

# Panel 1: Velocity profiles
for t in times_to_plot:
    v_field = history['snapshots'][t]['v_current']
    ax1.plot(radial_distances[sort_idx], v_field[sort_idx], linewidth=2, label=f't = {t} yr')

ax1.axvline(eq_radius_km, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Rupture edge')
ax1.set_xlabel('Distance from center (km)', fontsize=12)
ax1.set_ylabel('Velocity (m/yr)', fontsize=12)
ax1.set_title('Velocity vs Radius', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Cumulative slip profiles
for t in times_to_plot:
    cum_field = history['snapshots'][t]['cumulative_slip']
    ax2.plot(radial_distances[sort_idx], cum_field[sort_idx], linewidth=2, label=f't = {t} yr')

ax2.axvline(eq_radius_km, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Rupture edge')
ax2.set_xlabel('Distance from center (km)', fontsize=12)
ax2.set_ylabel('Cumulative Slip (m)', fontsize=12)
ax2.set_title('Cumulative Slip vs Radius', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/test_afterslip_radial_profiles.png', dpi=150, bbox_inches='tight')
print("  Saved: results/test_afterslip_radial_profiles.png")
plt.close()

# =============================================================================
# FIGURE 5: VELOCITY DECAY CURVES AT SAMPLE LOCATIONS
# =============================================================================

print("Creating Figure 5: Velocity decay curves...")
fig, ax = plt.subplots(figsize=(12, 7))

for loc in sample_locations:
    ax.semilogy(history['times'], velocity_timeseries[loc], linewidth=2, label=loc)

ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Velocity (m/yr)', fontsize=12)
ax.set_title('Velocity Decay at Sample Locations', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('results/test_afterslip_decay_curves.png', dpi=150, bbox_inches='tight')
print("  Saved: results/test_afterslip_decay_curves.png")
plt.close()

print("\n" + "="*70)
print("ALL FIGURES COMPLETE")
print("="*70)
print("Generated 5 diagnostic plots showing temporal evolution:")
print("  1. test_afterslip_time_series.png - Time series of key quantities")
print("  2. test_afterslip_velocity_evolution.png - Spatial velocity at 6 times")
print("  3. test_afterslip_cumulative_evolution.png - Cumulative slip at 6 times")
print("  4. test_afterslip_radial_profiles.png - Radial profiles over time")
print("  5. test_afterslip_decay_curves.png - Decay curves at sample locations")
print("="*70)
