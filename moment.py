"""
Moment accumulation and tracking
"""

import numpy as np


def initialize_moment(config, mesh):
    """
    Initialize moment distribution with background + Gaussian pulses

    Returns:
    --------
    m_current : (n_elements,) array of geometric moment (m)
    slip_rate : (n_elements,) array of slip deficit rate (m/year)
    """
    # Background slip rate (uniform)
    slip_rate = np.full(config.n_elements, config.background_slip_rate_m_yr)

    # Add Gaussian pulse(s)
    for pulse in config.moment_pulses:
        x_center = pulse["center_x_km"]
        z_center = pulse["center_z_km"]
        sigma = pulse["sigma_km"]
        amplitude = pulse["amplitude_mm_yr"] / 1000.0  # Convert to m/year

        # Compute distance from pulse center
        dx = mesh["centroids"][:, 0] - x_center
        dz = mesh["centroids"][:, 2] - z_center
        r = np.sqrt(dx**2 + dz**2)

        # Gaussian distribution
        gaussian = amplitude * np.exp(-(r**2) / (2 * sigma**2))

        slip_rate += gaussian

    # Start with zero accumulated moment
    m_current = np.zeros(config.n_elements)

    return m_current, slip_rate


def accumulate_moment(m_current, slip_rate, element_areas, dt_years):
    """
    Accumulate geometric moment over time step

    Parameters:
    -----------
    m_current : current geometric moment (m)
    slip_rate : slip deficit rate (m/year)
    element_areas : element areas (m²)
    dt_years : time step (years)

    Returns:
    --------
    m_new : updated geometric moment
    """
    # Geometric moment = slip × area
    delta_slip = slip_rate * dt_years
    delta_moment = delta_slip * element_areas

    m_new = m_current + delta_moment

    return m_new


def release_moment(m_current, slip_distribution, element_areas):
    """
    Release moment from earthquake

    Parameters:
    -----------
    m_current : current geometric moment
    slip_distribution : coseismic slip on each element (m)
    element_areas : element areas (m²)

    Returns:
    --------
    m_new : updated geometric moment after release
    """
    released_moment = slip_distribution * element_areas
    m_new = m_current - released_moment

    return m_new


def geometric_moment_to_seismic_moment(m_geom, element_areas, shear_modulus):
    """
    Convert geometric moment to seismic moment

    M0 = μ x slip x area = μ x (m_geom / area) x area = μ x m_geom

    Geometric moment m = slip x area (dimensions: length^3)
    Seismic moment M0 = μ x slip x area (dimensions: force x length)

    So: M0 = μ x m_geom / area x area = μ x m_geom

    Actually, if m_geom is already slip x area:
    M0 = μ x (m_geom / area) x area = μ x m_geom

    Hmm, let me reconsider the definition...
    """
    # In SKIES, geometric moment is slip × area
    # So m has dimensions of length³
    # To get seismic moment: M0 = μ × slip × area = μ × m
    # But m = slip × area, so this doesn't quite work...

    # Let me check: if m_geom represents total "slip volume"
    # Then to get M0, we need to know the average slip
    # Average slip = m_geom / total_area
    # M0 = μ × average_slip × total_area = μ × m_geom

    # Actually, for point-wise: m_i = slip_i × area_i
    # Total M0 = μ × Σ(slip_i × area_i) = μ × Σ m_i

    M0 = shear_modulus * np.sum(m_geom)
    return M0


def seismic_moment_to_magnitude(M0):
    """
    Convert seismic moment (N·m) to moment magnitude

    M_W = (2/3) × (log10(M0) - 9.05)
    """
    M_W = (2.0 / 3.0) * (np.log10(M0) - 9.05)
    return M_W


def magnitude_to_seismic_moment(M_W):
    """
    Convert moment magnitude to seismic moment (N·m)

    M0 = 10^(1.5 x M_W + 9.05)
    """
    M0 = 10 ** (1.5 * M_W + 9.05)
    return M0
