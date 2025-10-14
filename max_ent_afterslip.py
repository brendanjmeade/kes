import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist
from IPython.display import HTML

# =============================================================================
# PARAMETERS (all adjustable at top of script)
# =============================================================================

# Fault geometry
fault_length_km = 200.0  # Along-strike length (km)
fault_depth_km = 25.0  # Down-dip depth (km)
element_size_km = 1.0  # Grid cell size (km)

# Coseismic earthquake parameters
eq_center_x_km = fault_length_km / 2  # Center along-strike
eq_center_y_km = fault_depth_km / 2  # Center down-dip
eq_radius_km = 20.0  # Radius of circular rupture (km)
eq_slip_m = 1.0  # Coseismic slip within circle (m)

# Interseismic loading (simplified uniform)
v_load_mm_yr = 30.0  # Loading rate (mm/yr)
t_since_last_eq_yr = 100.0  # Time since last earthquake (yr)

# MaxEnt afterslip model parameters
v_ref_m_yr = 0.1  # Reference initial velocity (m/yr)
M_ref = 7.0  # Reference magnitude for v_ref
beta = 0.33  # Magnitude scaling exponent (1/3 from theory)
correlation_length_km = 30.0  # Spatial correlation length ξ
kernel_type = "exponential"  # 'exponential' or 'power_law'
power_law_exponent = 2.5  # If using power law kernel

# Temporal parameters
dt_years = 0.1  # Time step (years)
t_max_years = 10.0  # Maximum time (years)
n_frames_animation = 50  # Number of frames for animation

# Threshold parameters
m_critical = 0.01  # Minimum residual moment for afterslip (m)
v_min = 1e-6  # Minimum velocity for numerical stability (m/yr)

# =============================================================================
# SETUP FAULT MESH
# =============================================================================

nx = int(fault_length_km / element_size_km)
ny = int(fault_depth_km / element_size_km)
x_coords = np.linspace(element_size_km / 2, fault_length_km - element_size_km / 2, nx)
y_coords = np.linspace(element_size_km / 2, fault_depth_km - element_size_km / 2, ny)
X, Y = np.meshgrid(x_coords, y_coords)

# Patch areas (all equal in this uniform mesh)
A_i = element_size_km**2  # km²

print(f"Fault mesh: {nx} x {ny} = {nx * ny} elements")

# =============================================================================
# CALCULATE ACCUMULATED GEOMETRIC MOMENT
# =============================================================================

# Simplified: uniform loading everywhere
m_acc = v_load_mm_yr * 1e-3 * t_since_last_eq_yr  # Convert mm to m
m_accumulated = np.ones_like(X) * m_acc

print(f"Accumulated slip deficit: {m_acc:.2f} m")

# =============================================================================
# DEFINE COSEISMIC SLIP
# =============================================================================

# Calculate distance from earthquake center
dist_from_eq = np.sqrt((X - eq_center_x_km) ** 2 + (Y - eq_center_y_km) ** 2)

# Circular coseismic slip patch
m_coseismic = np.zeros_like(X)
m_coseismic[dist_from_eq <= eq_radius_km] = eq_slip_m

# Calculate moment magnitude
area_co_km2 = np.sum(m_coseismic > 0) * A_i
M0_Nm = np.sum(m_coseismic) * A_i * 1e6 * 3e10  # Rough conversion
Mw = (2 / 3) * np.log10(M0_Nm) - 6.07

print(f"Coseismic: Mw = {Mw:.2f}, Rupture area = {area_co_km2:.0f} km²")

# =============================================================================
# CALCULATE RESIDUAL GEOMETRIC MOMENT
# =============================================================================

m_residual_initial = m_accumulated - m_coseismic
m_residual_initial[m_residual_initial < m_critical] = 0  # Apply threshold

# =============================================================================
# SPATIAL ACTIVATION FUNCTION Φ_i
# =============================================================================


def calculate_spatial_kernel(X, Y, m_co, kernel_type="exponential"):
    """Calculate spatial activation based on distance from coseismic slip"""

    # Find patches that slipped coseismically
    co_patches = m_co > 0.1  # Significant slip threshold

    # Initialize activation
    Phi = np.zeros_like(X)

    # For each patch, calculate minimum distance to coseismic area
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if co_patches.any():
                # Distance to all coseismic patches
                dists = []
                for ii in range(X.shape[0]):
                    for jj in range(X.shape[1]):
                        if co_patches[ii, jj]:
                            d = np.sqrt(
                                (X[i, j] - X[ii, jj]) ** 2 + (Y[i, j] - Y[ii, jj]) ** 2
                            )
                            dists.append(d)

                min_dist = np.min(dists)

                if kernel_type == "exponential":
                    # Exponential decay
                    xi = correlation_length_km * (Mw / M_ref) ** beta
                    Phi[i, j] = np.exp(-min_dist / xi)

                elif kernel_type == "power_law":
                    # Power law decay
                    r0 = correlation_length_km * (Mw / M_ref) ** beta
                    Phi[i, j] = (1 + min_dist / r0) ** (-power_law_exponent)

    # Normalize to max of 1
    if Phi.max() > 0:
        Phi = Phi / Phi.max()

    return Phi


# Calculate spatial activation
Phi = calculate_spatial_kernel(X, Y, m_coseismic, kernel_type)

# =============================================================================
# INITIAL AFTERSLIP VELOCITY
# =============================================================================

# Magnitude scaling
v_mag_scale = v_ref_m_yr * (Mw / M_ref) ** beta

# Initial velocity field (MaxEnt linear form: f(m_r) = m_r)
v_initial = v_mag_scale * Phi * m_residual_initial

# Ensure numerical stability
v_initial[v_initial < v_min] = 0

print(f"Max initial afterslip velocity: {v_initial.max():.3f} m/yr")

# =============================================================================
# TIME EVOLUTION
# =============================================================================

time_steps = np.arange(0, t_max_years + dt_years, dt_years)
n_steps = len(time_steps)

# Storage arrays
m_residual_history = np.zeros((n_steps, ny, nx))
v_history = np.zeros((n_steps, ny, nx))
cumulative_slip = np.zeros((ny, nx))

# Initial conditions
m_residual = m_residual_initial.copy()
v_current = v_initial.copy()

# Time evolution loop
for it, t in enumerate(time_steps):
    # Store current state
    m_residual_history[it] = m_residual
    v_history[it] = v_current

    # Update for patches with afterslip
    active = (m_residual > 0) & (v_current > 0)

    if active.any():
        # Exponential decay: v(t) = v_0 * exp(-t * v_0 * A / m_r_0)
        decay_rate = v_initial[active] * A_i / m_residual_initial[active]
        v_current[active] = v_initial[active] * np.exp(-decay_rate * t)

        # Update residual moment
        if it > 0:
            dm = v_current[active] * A_i * dt_years
            m_residual[active] -= dm
            cumulative_slip[active] += v_current[active] * dt_years

        # Check for depleted patches
        depleted = m_residual < 0
        m_residual[depleted] = 0
        v_current[depleted] = 0

# =============================================================================
# VISUALIZATION
# =============================================================================

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. Initial conditions
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.pcolormesh(X, Y, m_accumulated, cmap="YlOrRd", shading="auto")
ax1.set_title("Accumulated Moment (pre-earthquake)")
ax1.set_xlabel("Along-strike (km)")
ax1.set_ylabel("Down-dip (km)")
plt.colorbar(im1, ax=ax1, label="Slip deficit (m)")

ax2 = plt.subplot(2, 3, 2)
im2 = ax2.pcolormesh(X, Y, m_coseismic, cmap="Blues", shading="auto")
circle = Circle(
    (eq_center_x_km, eq_center_y_km),
    eq_radius_km,
    fill=False,
    edgecolor="red",
    linewidth=2,
)
ax2.add_patch(circle)
ax2.set_title(f"Coseismic Slip (Mw {Mw:.1f})")
ax2.set_xlabel("Along-strike (km)")
ax2.set_ylabel("Down-dip (km)")
plt.colorbar(im2, ax=ax2, label="Coseismic slip (m)")

ax3 = plt.subplot(2, 3, 3)
im3 = ax3.pcolormesh(
    X, Y, m_residual_initial, cmap="RdBu_r", shading="auto", vmin=-0.5, vmax=m_acc
)
ax3.set_title("Residual Moment (post-earthquake)")
ax3.set_xlabel("Along-strike (km)")
ax3.set_ylabel("Down-dip (km)")
plt.colorbar(im3, ax=ax3, label="Residual moment (m)")

ax4 = plt.subplot(2, 3, 4)
im4 = ax4.pcolormesh(X, Y, Phi, cmap="viridis", shading="auto")
ax4.set_title("Spatial Activation Φ")
ax4.set_xlabel("Along-strike (km)")
ax4.set_ylabel("Down-dip (km)")
plt.colorbar(im4, ax=ax4, label="Activation")

ax5 = plt.subplot(2, 3, 5)
im5 = ax5.pcolormesh(X, Y, v_initial * 100, cmap="plasma", shading="auto")
ax5.set_title("Initial Afterslip Velocity")
ax5.set_xlabel("Along-strike (km)")
ax5.set_ylabel("Down-dip (km)")
plt.colorbar(im5, ax=ax5, label="Velocity (cm/yr)")

ax6 = plt.subplot(2, 3, 6)
# Plot velocity decay at center
center_idx = (ny // 2, nx // 2)
times_plot = time_steps[::5]  # Subsample for clarity
v_center = [v_history[i][center_idx] for i in range(0, n_steps, 5)]
ax6.semilogy(times_plot, v_center, "b-", label="Center patch")
ax6.set_xlabel("Time (years)")
ax6.set_ylabel("Velocity (m/yr)")
ax6.set_title("Velocity Decay (log scale)")
ax6.grid(True)
ax6.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# ANIMATION
# =============================================================================

fig_anim, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Setup plots
im1 = ax1.pcolormesh(
    X,
    Y,
    v_history[0] * 100,
    cmap="plasma",
    shading="auto",
    vmin=0,
    vmax=v_initial.max() * 100,
)
ax1.set_title("Afterslip Velocity")
ax1.set_xlabel("Along-strike (km)")
ax1.set_ylabel("Down-dip (km)")
cbar1 = plt.colorbar(im1, ax=ax1, label="Velocity (cm/yr)")

cumsum_temp = np.zeros_like(X)
im2 = ax2.pcolormesh(X, Y, cumsum_temp, cmap="Greens", shading="auto", vmin=0, vmax=0.5)
ax2.set_title("Cumulative Afterslip")
ax2.set_xlabel("Along-strike (km)")
ax2.set_ylabel("Down-dip (km)")
cbar2 = plt.colorbar(im2, ax=ax2, label="Cumulative slip (m)")


# Animation function
def animate(frame):
    idx = int(frame * n_steps / n_frames_animation)
    t = time_steps[idx]

    # Update velocity
    im1.set_array(v_history[idx].ravel() * 100)
    ax1.set_title(f"Afterslip Velocity (t = {t:.1f} yr)")

    # Calculate cumulative slip up to this time
    cumsum = np.sum(v_history[: idx + 1], axis=0) * dt_years
    im2.set_array(cumsum.ravel())
    ax2.set_title(f"Cumulative Afterslip (t = {t:.1f} yr)")

    return [im1, im2]


# Create animation
anim = animation.FuncAnimation(
    fig_anim, animate, frames=n_frames_animation, interval=100, blit=True, repeat=True
)

plt.tight_layout()

# Save animation (optional)
# anim.save('afterslip_evolution.gif', writer='pillow', fps=10)

plt.show()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

total_afterslip = cumulative_slip.sum() * A_i  # Total moment in m·km²
total_coseismic = m_coseismic.sum() * A_i
ratio = total_afterslip / total_coseismic if total_coseismic > 0 else 0

print("\n" + "=" * 50)
print("SUMMARY:")
print(f"Total coseismic moment: {total_coseismic:.1f} m·km²")
print(f"Total afterslip moment: {total_afterslip:.1f} m·km²")
print(f"Afterslip/Coseismic ratio: {ratio:.2%}")
print(f"Max cumulative afterslip: {cumulative_slip.max():.3f} m")
print(f"Afterslip area: {np.sum(cumulative_slip > 0.01) * A_i:.0f} km²")
print("=" * 50)
