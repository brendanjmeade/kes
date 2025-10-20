"""
Standalone validation test for single afterslip sequence

This script tests initialize_afterslip_sequence() in isolation to verify
it produces the correct spatial pattern (peak at inner halo edge).

Mimics the setup from max_ent_afterslip.py for direct comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import Config
from geometry import create_fault_mesh
from afterslip import initialize_afterslip_sequence

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

# m_current is not actually used in the new implementation,
# but we need to pass something
m_current = np.zeros(config.n_elements)

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

# Find where peak v_initial occurs
peak_idx = np.argmax(v_initial)
peak_x = centroids[peak_idx, 0]
peak_z = centroids[peak_idx, 2]
peak_dist_from_center = dist_from_eq[peak_idx]

print(f"\nPeak v_initial location:")
print(f"  Position: x={peak_x:.1f} km, z={peak_z:.1f} km")
print(f"  Distance from rupture center: {peak_dist_from_center:.1f} km")
print(f"  Expected: ~{eq_radius_km} km (inner halo edge)")
print(f"  Phi at peak: {Phi[peak_idx]:.3f}")
print(f"  m_residual at peak: {m_residual_initial[peak_idx]:.2f} m")

# =============================================================================
# VISUALIZATION - Compare with max_ent_afterslip.py
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Reshape to 2D grids
nx = config.n_along_strike
ny = config.n_down_dip

slip_grid = slip_coseismic.reshape(nx, ny).T
Phi_grid = Phi.reshape(nx, ny).T
m_residual_grid = m_residual_initial.reshape(nx, ny).T
v_initial_grid = v_initial.reshape(nx, ny).T

# Row 1: Individual components
ax = axes[0, 0]
im = ax.imshow(slip_grid, aspect='equal', origin='lower', extent=[0, 200, 0, 25], cmap='Reds')
ax.set_title('Coseismic slip (m)', fontsize=14, fontweight='bold')
ax.set_xlabel('Along-strike (km)')
ax.set_ylabel('Down-dip (km)')
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
im = ax.imshow(Phi_grid, aspect='equal', origin='lower', extent=[0, 200, 0, 25], cmap='viridis')
ax.set_title('Spatial activation Φ', fontsize=14, fontweight='bold')
ax.set_xlabel('Along-strike (km)')
ax.set_ylabel('Down-dip (km)')
plt.colorbar(im, ax=ax)

ax = axes[0, 2]
im = ax.imshow(m_residual_grid, aspect='equal', origin='lower', extent=[0, 200, 0, 25], cmap='plasma')
ax.set_title('Residual moment m_residual (m)', fontsize=14, fontweight='bold')
ax.set_xlabel('Along-strike (km)')
ax.set_ylabel('Down-dip (km)')
plt.colorbar(im, ax=ax)

# Row 2: Product and radial profiles
ax = axes[1, 0]
im = ax.imshow(v_initial_grid, aspect='equal', origin='lower', extent=[0, 200, 0, 25], cmap='YlOrRd')
ax.set_title('Initial velocity v_initial (m/yr)', fontsize=14, fontweight='bold')
ax.set_xlabel('Along-strike (km)')
ax.set_ylabel('Down-dip (km)')
ax.scatter([eq_center_x_km], [eq_center_z_km], c='blue', s=200, marker='*', label='Rupture center', edgecolors='white', linewidths=2)
ax.scatter([peak_x], [peak_z], c='lime', s=100, marker='o', label='Peak v', edgecolors='black', linewidths=2)
ax.legend(loc='upper right')
plt.colorbar(im, ax=ax)

# Radial profile through rupture center
ax = axes[1, 1]
radial_distances = dist_from_eq
sort_idx = np.argsort(radial_distances)
ax.plot(radial_distances[sort_idx], Phi[sort_idx], 'b-', linewidth=2, label='Φ', alpha=0.7)
ax.axvline(eq_radius_km, color='red', linestyle='--', linewidth=2, label='Rupture edge')
ax.set_xlabel('Distance from rupture center (km)', fontsize=12)
ax.set_ylabel('Φ', fontsize=12, color='b')
ax.tick_params(axis='y', labelcolor='b')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
ax.set_title('Radial profile: Φ', fontsize=14, fontweight='bold')

ax2 = axes[1, 2]
ax2.plot(radial_distances[sort_idx], v_initial[sort_idx], 'r-', linewidth=2, label='v_initial')
ax2.axvline(eq_radius_km, color='red', linestyle='--', linewidth=2, label='Rupture edge')
ax2.set_xlabel('Distance from rupture center (km)', fontsize=12)
ax2.set_ylabel('v_initial (m/yr)', fontsize=12, color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.set_title('Radial profile: v_initial', fontsize=14, fontweight='bold')

# Add text box with expected vs actual
textstr = f"""VALIDATION CHECK:
Expected peak location: ~{eq_radius_km} km (inner halo edge)
Actual peak location: {peak_dist_from_center:.1f} km
Status: {'✓ PASS' if abs(peak_dist_from_center - eq_radius_km) < 5 else '✗ FAIL'}
"""
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
fig.text(0.5, 0.02, textstr, fontsize=12, bbox=props, ha='center', family='monospace')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('results/test_single_afterslip.png', dpi=150, bbox_inches='tight')
print(f"\nSaved diagnostic plot: results/test_single_afterslip.png")
plt.close()

# =============================================================================
# DIAGNOSTIC: Check if pattern matches max_ent_afterslip.py expectations
# =============================================================================

# For a proper MaxEnt afterslip pattern:
# - Peak should be at inner halo edge (distance ≈ rupture_radius)
# - Velocity should be lower inside rupture (low m_residual)
# - Velocity should decay smoothly away from rupture (decreasing Phi)

# Sample velocities at key locations
inside_rupture_idx = np.where(dist_from_eq < eq_radius_km / 2)[0]
at_edge_idx = np.where(np.abs(dist_from_eq - eq_radius_km) < 2.0)[0]  # Within 2km of edge
far_away_idx = np.where((dist_from_eq > eq_radius_km + 20) & (dist_from_eq < eq_radius_km + 30))[0]

v_inside = np.mean(v_initial[inside_rupture_idx]) if len(inside_rupture_idx) > 0 else 0
v_edge = np.mean(v_initial[at_edge_idx]) if len(at_edge_idx) > 0 else 0
v_far = np.mean(v_initial[far_away_idx]) if len(far_away_idx) > 0 else 0

print(f"\n" + "="*60)
print("SPATIAL PATTERN VALIDATION")
print("="*60)
print(f"Average v_initial inside rupture (r < {eq_radius_km/2:.0f} km): {v_inside:.4f} m/yr")
print(f"Average v_initial at halo edge (r ≈ {eq_radius_km} km):  {v_edge:.4f} m/yr")
print(f"Average v_initial far from rupture (r ≈ {eq_radius_km+25:.0f} km): {v_far:.4f} m/yr")
print()

# Expected: v_edge > v_inside and v_edge > v_far
if v_edge > v_inside and v_edge > v_far:
    print("✓ PASS: Peak velocity at inner halo edge (expected MaxEnt pattern)")
elif v_far > v_edge:
    print("✗ FAIL: Velocity increases with distance (WRONG - this is what we've been seeing)")
elif v_inside > v_edge:
    print("✗ FAIL: Peak velocity inside rupture (incorrect)")
else:
    print("? UNCLEAR: Pattern doesn't match expected behavior")

print("="*60)
