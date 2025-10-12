"""
Rate-based event generation
Can generate 0, 1, 2, ... events per time step
"""

import numpy as np
from moment import magnitude_to_seismic_moment, seismic_moment_to_magnitude
from spatial_prob import spatial_probability
from slip_generator import generate_slip_distribution
from temporal_prob import earthquake_rate


def draw_magnitude(config):
    """
    Sample magnitude from Gutenberg-Richter distribution
    """
    b = config.b_value
    M_min = config.M_min
    M_max = config.M_max

    # Inverse transform sampling
    denom = 10 ** (-b * M_min) - 10 ** (-b * M_max)
    u = np.random.random()

    magnitude = -np.log10(10 ** (-b * M_min) - u * denom) / b

    return magnitude


def generate_events_in_timestep(
    m_current, event_history, current_time, dt_years, mesh, config
):
    """
    Generate events for a single time step using rate-based approach

    Returns:
    --------
    events : list of event dictionaries (can be empty, or have multiple events)
    m_updated : updated moment distribution
    """
    # Compute instantaneous rate
    lambda_t, components = earthquake_rate(
        m_current, event_history, current_time, config
    )

    # Expected number of events in this time interval
    expected_events = lambda_t * dt_years

    # Generate actual number of events (Poisson process)
    n_events = np.random.poisson(expected_events)

    # Generate each event
    events = []
    m_working = m_current.copy()

    for i in range(n_events):
        # Draw magnitude
        magnitude = draw_magnitude(config)

        # Compute spatial probability for this magnitude
        p_spatial, gamma_used = spatial_probability(m_working, magnitude, config)

        # Sample hypocenter location
        hypocenter_idx = np.random.choice(config.n_elements, p=p_spatial)

        # Generate slip distribution
        slip, ruptured_elements = generate_slip_distribution(
            hypocenter_idx, magnitude, m_working, mesh, config
        )

        # Compute actual moment
        M0_actual = config.shear_modulus_Pa * np.sum(slip * config.element_area_m2)
        M_actual = seismic_moment_to_magnitude(M0_actual)

        # Update moment distribution
        from moment import release_moment

        m_working = release_moment(m_working, slip, config.element_area_m2)

        # Create event record
        hypo_x = mesh["centroids"][hypocenter_idx, 0]
        hypo_z = mesh["centroids"][hypocenter_idx, 2]

        event = {
            "time": current_time,
            "magnitude": M_actual,
            "M0": M0_actual,
            "hypocenter_idx": hypocenter_idx,
            "hypocenter_x_km": hypo_x,
            "hypocenter_z_km": hypo_z,
            "ruptured_elements": ruptured_elements,
            "slip": slip,
            "gamma_used": gamma_used,
            "lambda_t": lambda_t,
            "components": components,
        }

        events.append(event)

    return events, m_working
