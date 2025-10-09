"""
Stochastic slip distribution generation

Try to use SKIES approach if possible, otherwise use simple geometric shapes
"""

import numpy as np
from scipy.spatial.distance import cdist


def generate_slip_distribution(hypocenter_idx, magnitude, m_current, mesh, config):
    """
    Generate spatially heterogeneous slip distribution

    Approach:
    1. Determine rupture area from magnitude
    2. Find elements within this area
    3. Generate stochastic slip using correlation
    4. Scale to match target moment

    Returns:
    --------
    slip : (n_elements,) array of coseismic slip (m)
    ruptured_elements : list of element indices that slipped
    """
    from moment import magnitude_to_seismic_moment

    # Target seismic moment
    M0_target = magnitude_to_seismic_moment(magnitude)

    # Expected rupture area (Allen & Hayes 2017 for strike-slip)
    # log10(A) = M - 3.99  (A in km²)
    log10_area = magnitude - 3.99
    area_km2 = 10**log10_area
    area_m2 = area_km2 * 1e6

    # Find elements within rupture area
    ruptured_elements = select_rupture_elements(hypocenter_idx, area_m2, mesh, config)

    if len(ruptured_elements) == 0:
        return np.zeros(config.n_elements), []

    # Generate heterogeneous slip
    slip_pattern = generate_heterogeneous_slip_pattern(
        hypocenter_idx, ruptured_elements, mesh, config
    )

    # Scale to match target moment
    # M0 = μ × Σ(slip_i × area_i)
    # Current: M0_current = μ × Σ(slip_pattern_i × area_i)
    current_moment = config.shear_modulus_Pa * np.sum(
        slip_pattern[ruptured_elements] * config.element_area_m2
    )

    scale_factor = M0_target / current_moment if current_moment > 0 else 1.0

    slip = np.zeros(config.n_elements)
    slip[ruptured_elements] = slip_pattern[ruptured_elements] * scale_factor

    return slip, ruptured_elements


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
