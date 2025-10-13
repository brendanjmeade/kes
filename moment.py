"""
Moment accumulation and tracking
"""

import numpy as np


def initialize_moment(config, mesh):
    """
    Initialize geometric moment distribution on fault with initial spin-up

    Start partway through the earthquake cycle to avoid long initial transient

    Parameters:
    -----------
    config : Config object
    mesh : dict with fault geometry

    Returns:
    --------
    m_current : array of geometric moment (m³) for each element
    slip_rate : array of slip rate (m/year) for each element
    """
    from moment import magnitude_to_seismic_moment, seismic_moment_to_magnitude

    print("\nInitializing moment distribution...")

    # === Create spatially heterogeneous slip rate ===

    # Base slip rate (uniform)
    slip_rate = np.ones(config.n_elements) * config.background_slip_rate_m_yr

    # Add Gaussian moment-rate pulses if specified
    if (
        hasattr(config, "moment_pulses")
        and config.moment_pulses is not None
        and len(config.moment_pulses) > 0
    ):
        centroids = mesh["centroids"]

        for pulse in config.moment_pulses:
            # Get pulse parameters from the pulse dict/tuple
            if isinstance(pulse, dict):
                # FIX: Use correct key names with _km and _mm_yr suffixes
                center_x = pulse.get("center_x_km", pulse.get(
                    "center_x", np.random.uniform(0, config.fault_length_km)
                ))
                center_z = pulse.get("center_z_km", pulse.get(
                    "center_z", config.fault_depth_km / 2
                ))
                # Convert amplitude from mm/yr to m/yr
                amplitude_mm_yr = pulse.get("amplitude_mm_yr", pulse.get(
                    "amplitude", config.background_slip_rate_m_yr * 1000
                ))
                amplitude = amplitude_mm_yr / 1000.0  # Convert to m/yr
                width = pulse.get("sigma_km", pulse.get("width_km", 20.0))
            else:
                # Default: random location
                center_x = np.random.uniform(0, config.fault_length_km)
                center_z = config.fault_depth_km / 2
                amplitude = config.background_slip_rate_m_yr
                width = 20.0

            # Gaussian in 2D
            dx = centroids[:, 0] - center_x
            dz = centroids[:, 2] - center_z
            r_sq = dx**2 + dz**2

            gaussian = amplitude * np.exp(-r_sq / (2 * width**2))

            slip_rate += gaussian

        print(
            f"  Background slip rate: {config.background_slip_rate_m_yr * 1000:.1f} mm/year"
        )
        print(f"  Number of moment pulses: {len(config.moment_pulses)}")
    else:
        print(
            f"  Uniform slip rate: {config.background_slip_rate_m_yr * 1000:.1f} mm/year"
        )

    # === Initialize with partial earthquake cycle ===

    # Compute lambda_required from moment balance
    geom_loading_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )
    seismic_loading_rate = config.shear_modulus_Pa * geom_loading_rate

    M0_max = magnitude_to_seismic_moment(config.M_max)
    M0_avg = M0_max / (1.5 + config.b_value)

    lambda_required = seismic_loading_rate / M0_avg
    recurrence_time = 1.0 / lambda_required

    # Start at 50% of the way to mid-cycle
    # Mid-cycle is at recurrence_time/2, so 50% of that is recurrence_time/4
    initial_time_equivalent = recurrence_time / 4  # years

    # Accumulate moment for this equivalent time
    m_current = slip_rate * config.element_area_m2 * initial_time_equivalent

    print(f"  Recurrence time: {recurrence_time:.1f} years")
    print(
        f"  Starting at equivalent time: {initial_time_equivalent:.1f} years into cycle"
    )
    print(f"  Initial total moment: {np.sum(m_current):.2e} m³")
    print(f"  Mid-cycle moment: {geom_loading_rate * recurrence_time / 2:.2e} m³")
    print(
        f"  Initial as % of mid-cycle: {100 * np.sum(m_current) / (geom_loading_rate * recurrence_time / 2):.1f}%"
    )

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

    Ensures moment never goes negative

    Parameters:
    -----------
    m_current : current geometric moment (m)
    slip_distribution : coseismic slip on each element (m)
    element_areas : element areas (m²)

    Returns:
    --------
    m_new : updated geometric moment after release (m)
    """
    # Compute released slip per unit area (geometric moment density)
    # slip has units of meters, which when divided by element area gives m³/m² = m
    # But m_current has units of meters (geometric moment per unit area)
    # So we directly subtract slip from m_current

    m_new = m_current - slip_distribution

    # Ensure non-negative (safety check, should already be satisfied by constraint)
    m_new = np.maximum(m_new, 0.0)

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
