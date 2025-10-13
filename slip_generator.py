"""
Stochastic slip distribution generation

Try to use SKIES approach if possible, otherwise use simple geometric shapes
"""

import numpy as np
from scipy.spatial.distance import cdist


def generate_slip_distribution(hypocenter_idx, magnitude, m_current, mesh, config):
    """
    Generate spatially heterogeneous slip distribution

    Scales entire event uniformly if total requested moment exceeds available moment
    This preserves slip distribution shape while ensuring moment balance

    Returns:
    --------
    slip : (n_elements,) array of coseismic slip (m)
    ruptured_elements : list of element indices that slipped
    M0_actual : actual seismic moment released (N·m)
    """
    from moment import magnitude_to_seismic_moment

    # Target seismic moment from magnitude
    M0_target = magnitude_to_seismic_moment(magnitude)

    # Expected rupture area (Allen & Hayes 2017 for strike-slip)
    # log10(A) = M - 3.99  (A in km²)
    log10_area = magnitude - 3.99
    area_km2 = 10**log10_area
    area_m2 = area_km2 * 1e6

    # Find elements within rupture area
    ruptured_elements = select_rupture_elements(hypocenter_idx, area_m2, mesh, config)

    if len(ruptured_elements) == 0:
        return np.zeros(config.n_elements), [], 0.0

    # Generate heterogeneous slip pattern
    slip_pattern = generate_heterogeneous_slip_pattern(
        hypocenter_idx, ruptured_elements, mesh, config
    )

    # Scale slip pattern to match target moment
    current_moment = config.shear_modulus_Pa * np.sum(
        slip_pattern[ruptured_elements] * config.element_area_m2
    )
    scale_factor = M0_target / current_moment if current_moment > 0 else 1.0

    slip = np.zeros(config.n_elements)
    slip[ruptured_elements] = slip_pattern[ruptured_elements] * scale_factor

    # CRITICAL: Check total available moment in rupture area
    # Available geometric moment on each ruptured element
    available_geom_moment_per_element = m_current[
        ruptured_elements
    ]  # m (slip deficit per element)
    total_available_geom_moment = np.sum(
        available_geom_moment_per_element * config.element_area_m2
    )  # m³ (total geometric moment available)

    # Requested geometric moment for this event
    requested_geom_moment = M0_target / config.shear_modulus_Pa  # m³

    # If requesting more than available, scale DOWN the entire event uniformly
    if requested_geom_moment > total_available_geom_moment:
        # Uniform scaling preserves slip distribution shape
        constraint_factor = total_available_geom_moment / requested_geom_moment
        slip[ruptured_elements] *= constraint_factor

        # Actual moment is limited by available moment
        M0_actual = config.shear_modulus_Pa * total_available_geom_moment
    else:
        # Full moment can be released
        M0_actual = M0_target

    # Verify slip doesn't exceed available moment on any element (safety check)
    # This should be automatically satisfied by uniform scaling, but check anyway
    slip[ruptured_elements] = np.minimum(
        slip[ruptured_elements], m_current[ruptured_elements]
    )

    # Recompute actual moment after safety check
    M0_actual = config.shear_modulus_Pa * np.sum(
        slip[ruptured_elements] * config.element_area_m2
    )

    return slip, ruptured_elements, M0_actual


def select_rupture_elements(hypocenter_idx, target_area_m2, mesh, config):
    """
    Select elements that will rupture, starting from hypocenter

    Strategy:
    - Grow roughly circular/elliptical from hypocenter
    - Respect fault boundaries
    - Try to reach target area
    """
    # Get hypocenter position
    hypo_pos = mesh["centroids"][hypocenter_idx]

    # Compute distances from all elements to hypocenter
    distances = np.linalg.norm(mesh["centroids"] - hypo_pos, axis=1)

    # Sort by distance
    sorted_indices = np.argsort(distances)

    # Select elements until target area reached
    cumulative_area = 0.0
    ruptured_elements = []

    for idx in sorted_indices:
        ruptured_elements.append(idx)
        cumulative_area += config.element_area_m2

        if cumulative_area >= target_area_m2:
            break

    return ruptured_elements


def generate_heterogeneous_slip_pattern(
    hypocenter_idx, ruptured_elements, mesh, config
):
    """
    Generate spatially correlated slip pattern

    Use simple approach:
    - Maximum slip at hypocenter
    - Decay with distance (with some randomness)
    """
    slip = np.zeros(config.n_elements)

    if len(ruptured_elements) == 0:
        return slip

    # Get positions of ruptured elements
    positions = mesh["centroids"][ruptured_elements]
    hypo_pos = mesh["centroids"][hypocenter_idx]

    # Distances from hypocenter
    distances = np.linalg.norm(positions - hypo_pos, axis=1)

    # Maximum distance (for normalization)
    max_dist = np.max(distances) if len(distances) > 1 else 1.0

    # Slip tapers from hypocenter with some stochasticity
    # Use exponential decay × random factor
    normalized_dist = distances / max_dist

    # Base pattern: exponential taper
    base_slip = np.exp(-2 * normalized_dist)

    # Add random perturbations (±30%)
    random_factor = 1.0 + 0.3 * (2 * np.random.random(len(ruptured_elements)) - 1)

    slip_values = base_slip * random_factor

    # Ensure non-negative
    slip_values = np.maximum(slip_values, 0.0)

    # Assign to slip array
    slip[ruptured_elements] = slip_values

    return slip


# Alternative: Try to adapt SKIES stochastic slip generator
def generate_slip_skies_style(hypocenter_idx, ruptured_elements, mesh, config):
    """
    SKIES-style slip generation using Karhunen-Loève expansion

    This is more sophisticated - use if time permits
    """
    # This would require implementing the correlation matrix approach
    # from the SKIES repo. For now, use simpler approach above.
    pass
