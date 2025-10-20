"""
Afterslip physics from MaxEnt formulation

Implements spatially-localized afterslip following coseismic events:
- Spatial activation kernel Φ(x,y) decays from rupture zone
- Initial velocity v₀ ∝ Φ × residual_moment
- Temporal decay: v(t) = v₀ × exp(-decay_rate × t)
- Stops when residual moment depleted

The same spatial kernel Φ is used for aftershock localization.
"""

import numpy as np


def calculate_spatial_activation_kernel(mesh, ruptured_elements, magnitude, config):
    """
    Calculate spatial activation Φ(x,y) based on distance from coseismic rupture

    The kernel represents the spatial footprint where afterslip and aftershocks
    are activated. It decays with distance from the rupture zone.

    Parameters:
    -----------
    mesh : dict
        Fault mesh with 'centroids' field
    ruptured_elements : list
        Indices of elements that ruptured coseismically
    magnitude : float
        Mainshock magnitude (for scaling correlation length)
    config : Config
        Configuration with afterslip parameters

    Returns:
    --------
    Phi : (n_elements,) array
        Spatial activation field, normalized to max of 1.0
    """
    n_elements = config.n_elements
    Phi = np.zeros(n_elements)

    if len(ruptured_elements) == 0:
        return Phi

    # Get positions of ruptured elements
    ruptured_positions = mesh['centroids'][ruptured_elements]
    all_positions = mesh['centroids']

    # Magnitude-dependent correlation lengths (anisotropic)
    # ξ_x = ξ₀_x × (M / M_ref)^β (along-strike)
    # ξ_z = ξ₀_z × (M / M_ref)^β (down-dip)
    xi_x = config.afterslip_correlation_length_x_km * (
        magnitude / config.afterslip_M_ref
    ) ** config.afterslip_beta
    xi_z = config.afterslip_correlation_length_z_km * (
        magnitude / config.afterslip_M_ref
    ) ** config.afterslip_beta

    # For each element, compute minimum anisotropic distance to rupture zone
    for i in range(n_elements):
        # Component-wise distances to all ruptured patches
        dx = all_positions[i, 0] - ruptured_positions[:, 0]  # Along-strike (x)
        dz = all_positions[i, 2] - ruptured_positions[:, 2]  # Down-dip (z)

        # Anisotropic distance: d_scaled = sqrt((dx/ξ_x)² + (dz/ξ_z)²)
        distances_scaled = np.sqrt((dx / xi_x)**2 + (dz / xi_z)**2)
        min_dist_scaled = np.min(distances_scaled)

        # Apply spatial kernel
        if config.afterslip_kernel_type == "exponential":
            # Exponential decay: Φ = exp(-d_scaled)
            Phi[i] = np.exp(-min_dist_scaled)

        elif config.afterslip_kernel_type == "power_law":
            # Power law decay: Φ = (1 + d_scaled)^(-n)
            Phi[i] = (1 + min_dist_scaled) ** (-config.afterslip_power_law_exponent)

    # Normalize to max of 1.0
    if Phi.max() > 0:
        Phi = Phi / Phi.max()

    return Phi


def initialize_afterslip_sequence(event, m_current, mesh, config):
    """
    Initialize afterslip sequence following a coseismic event

    Computes:
    - Residual moment field: m_residual = m_accumulated - m_coseismic
    - Spatial activation Φ
    - Initial velocity field: v₀ = v_ref × (M/M_ref)^β × Φ × m_residual
    - Decay rates per element

    Uses uniform accumulated moment (mid-cycle value) to create spatial contrast,
    matching the MaxEnt reference implementation.

    Parameters:
    -----------
    event : dict
        Event record with 'ruptured_elements', 'slip', 'magnitude', 'time'
    m_current : (n_elements,) array
        Current moment field AFTER the coseismic release (not used for m_residual calculation)
    mesh : dict
        Fault mesh
    config : Config
        Configuration

    Returns:
    --------
    sequence : dict
        Afterslip sequence with all parameters needed for time evolution
    """
    magnitude = event['magnitude']
    time = event['time']
    ruptured_elements = event['ruptured_elements']
    slip_coseismic = event['slip']

    # Residual moment = actual moment field after earthquake
    # m_current was passed in AFTER coseismic slip was removed
    # So it already represents m_accumulated - m_coseismic
    # This is the local moment deficit that drives afterslip
    m_residual_initial = m_current.copy()

    # Apply threshold: only patches with sufficient moment participate
    m_residual_initial = np.maximum(m_residual_initial, 0.0)
    m_residual_initial[m_residual_initial < config.afterslip_m_critical] = 0.0

    # Calculate spatial activation kernel
    Phi = calculate_spatial_activation_kernel(mesh, ruptured_elements, magnitude, config)

    # Magnitude scaling for initial velocity
    # v_mag = v_ref × (M / M_ref)^β
    v_mag_scale = config.afterslip_v_ref_m_yr * (
        magnitude / config.afterslip_M_ref
    ) ** config.afterslip_beta

    # Initial velocity field (MaxEnt form: v ∝ Φ × m_residual)
    # This creates peak afterslip in halo region (high m_residual, medium Phi)
    # and lower afterslip on ruptured patch (low m_residual despite high Phi)
    v_initial = v_mag_scale * Phi * m_residual_initial

    # Apply minimum velocity threshold for numerical stability
    # No spatial threshold - let natural Phi × m_residual product determine extent
    v_initial[v_initial < config.afterslip_v_min] = 0.0

    # Decay rates: decay_rate = v₀ × A / m_residual₀
    # This ensures afterslip stops when moment is depleted
    decay_rates = np.zeros_like(v_initial)
    active_mask = (m_residual_initial > 0) & (v_initial > 0)

    if active_mask.any():
        decay_rates[active_mask] = (
            v_initial[active_mask] * config.element_area_m2 / m_residual_initial[active_mask]
        )

    # Compute total moment budget for this sequence
    # This is the maximum geometric moment that can be released
    moment_budget_total = np.sum(m_residual_initial * config.element_area_m2)

    # Create sequence record
    sequence = {
        'mainshock_time': time,
        'magnitude': magnitude,
        'ruptured_elements': ruptured_elements,
        'Phi': Phi,  # Spatial activation (used for aftershock localization too)
        'v_initial': v_initial,  # Initial velocity field (m/yr per element)
        'm_residual_initial': m_residual_initial,  # Initial residual moment (m per element)
        'm_residual_current': m_residual_initial.copy(),  # Evolving residual moment
        'decay_rates': decay_rates,  # Decay rate per element (1/yr)
        'cumulative_afterslip': np.zeros(config.n_elements),  # Track total afterslip
        'moment_budget': moment_budget_total,  # Maximum moment this sequence can release (m³)
        'moment_released': 0.0,  # Track cumulative release (m³)
        'active': True,
    }

    return sequence


def update_afterslip_sequences(sequences, current_time, dt_years, config):
    """
    Update all active afterslip sequences and compute moment release

    For each sequence:
    - Compute current velocity v(t) = v₀ × exp(-decay_rate × t)
    - Compute slip increment dm = v(t) × dt
    - Update residual moment
    - Deactivate depleted patches

    Parameters:
    -----------
    sequences : list of dict
        Active afterslip sequences
    current_time : float
        Current simulation time (years)
    dt_years : float
        Timestep (years)
    config : Config
        Configuration

    Returns:
    --------
    total_release : (n_elements,) array
        Total moment release from all sequences this timestep (m per element)
    """
    total_release = np.zeros(config.n_elements)

    for seq in sequences:
        if not seq['active']:
            continue

        # Time since mainshock
        dt_since_mainshock = current_time - seq['mainshock_time']

        # Check if sequence has expired
        if dt_since_mainshock > config.afterslip_duration_years:
            seq['active'] = False
            continue

        # Get current velocity using exponential decay
        # v(t) = v₀ × exp(-decay_rate × t)
        v_initial = seq['v_initial']
        decay_rates = seq['decay_rates']
        v_current = v_initial * np.exp(-decay_rates * dt_since_mainshock)

        # Patches with active afterslip
        active_mask = (seq['m_residual_current'] > 0) & (v_current > config.afterslip_v_min)

        if not active_mask.any():
            seq['active'] = False
            continue

        # Compute slip increment for this timestep
        # dm = v × dt (slip in meters)
        dm = np.zeros(config.n_elements)
        dm[active_mask] = v_current[active_mask] * dt_years

        # Don't exceed available residual moment
        dm = np.minimum(dm, seq['m_residual_current'])

        # Compute geometric moment for this release
        dm_moment = np.sum(dm) * config.element_area_m2

        # Check against sequence budget
        if seq['moment_released'] + dm_moment > seq['moment_budget']:
            # Scale back to not exceed budget
            scale_factor = (seq['moment_budget'] - seq['moment_released']) / dm_moment
            scale_factor = max(0.0, min(1.0, scale_factor))
            dm *= scale_factor
            dm_moment = np.sum(dm) * config.element_area_m2

        # Update residual moment
        seq['m_residual_current'] -= dm

        # Track cumulative afterslip for this sequence
        seq['cumulative_afterslip'] += dm
        seq['moment_released'] += dm_moment

        # Add to total release
        total_release += dm

        # Check for depleted patches or budget exhausted
        depleted_mask = seq['m_residual_current'] < config.afterslip_m_critical
        seq['m_residual_current'][depleted_mask] = 0.0

        # Deactivate if budget exhausted
        if seq['moment_released'] >= seq['moment_budget'] * 0.999:
            seq['active'] = False

    return total_release


def get_active_afterslip_sequences(sequences, current_time, duration_cutoff):
    """
    Filter sequences to find those within temporal window

    Parameters:
    -----------
    sequences : list of dict
        All afterslip sequences
    current_time : float
        Current simulation time (years)
    duration_cutoff : float
        Maximum age of sequence to consider active (years)

    Returns:
    --------
    active_sequences : list of dict
        Filtered list of active sequences
    """
    active_sequences = []

    for seq in sequences:
        dt = current_time - seq['mainshock_time']
        if seq['active'] and 0 < dt <= duration_cutoff:
            active_sequences.append(seq)

    return active_sequences


def compute_aftershock_spatial_weights(event_history, current_time, config):
    """
    Compute spatial weighting for aftershock nucleation

    Combines contributions from all active mainshock sequences.
    Uses stored spatial activation Φ and Omori temporal decay.

    Parameters:
    -----------
    event_history : list of dict
        All events (must include 'spatial_activation' field)
    current_time : float
        Current time (years)
    config : Config
        Configuration with Omori parameters

    Returns:
    --------
    weights : (n_elements,) array
        Spatial weighting for aftershock probability (≥ 1.0 everywhere)
    n_active_sequences : int
        Number of sequences contributing
    """
    weights = np.ones(config.n_elements)  # Base weight = 1.0
    n_active_sequences = 0

    if not config.omori_enabled:
        return weights, n_active_sequences

    # Omori parameters
    omori_c_years = config.omori_c_days / 365.25

    for event in event_history:
        dt_years = current_time - event['time']

        # Only consider events within aftershock duration window
        if not (0 < dt_years <= config.omori_duration_years):
            continue

        # Get spatial activation for this mainshock
        Phi = event.get('spatial_activation', None)
        if Phi is None:
            continue  # No spatial info, skip

        n_active_sequences += 1

        # Omori temporal weight: K / (t + c)^p
        M_mainshock = event['magnitude']
        K = config.omori_K_ref * 10 ** (
            config.omori_alpha * (M_mainshock - config.omori_M_ref)
        )
        temporal_weight = K / (dt_years + omori_c_years) ** config.omori_p

        # Add spatial contribution from this mainshock
        # Φ ranges [0, 1], so this adds more weight near rupture zones
        weights += temporal_weight * Phi

    return weights, n_active_sequences
