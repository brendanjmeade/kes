"""
Main event generation logic combining spatial and temporal
"""

import numpy as np
from moment import magnitude_to_seismic_moment, seismic_moment_to_magnitude
from spatial_prob import spatial_probability
from slip_generator import generate_slip_distribution


def draw_magnitude(config):
    """
    Sample magnitude from Gutenberg-Richter distribution
    """
    # Inverse transform sampling
    b = config.b_value
    M_min = config.M_min
    M_max = config.M_max

    denom = 10 ** (-b * M_min) - 10 ** (-b * M_max)
    u = np.random.random()

    magnitude = -np.log10(10 ** (-b * M_min) - u * denom) / b

    return magnitude


def generate_event(m_current, event_history, current_time, mesh, config):
    """
    Generate a single earthquake event

    Returns:
    --------
    event : dict with event properties, or None if no event
    m_updated : updated moment distribution
    """
    # from temporal_prob import temporal_probability

    # # Compute temporal probability
    # lambda_t, components = temporal_probability(
    #     m_current, event_history, current_time, config
    # )

    # # Bernoulli draw: does event occur?
    # dt_years = config.time_step_days / 365.25
    # p_event = lambda_t * dt_years

    # if np.random.random() > p_event:
    #     return None, m_current  # No event

    from temporal_prob import temporal_probability

    # Compute temporal probability
    lambda_t, components = temporal_probability(
        m_current, event_history, current_time, config
    )

    # Bernoulli draw: does event occur?
    dt_years = config.time_step_days / 365.25

    # IMPORTANT: lambda_t is event rate (events/year)
    # But it's already scaled by tanh to be between [lambda_min, lambda_max]
    # So probability per time step is:
    p_event = lambda_t * dt_years

    # Additional check: don't let p_event exceed reasonable value
    p_event = min(p_event, 0.1)  # Max 10% chance per time step

    # DEBUG OUTPUT
    if current_time > 0 and int(current_time * 10) % 100 == 0:  # Every 10 years
        print(
            f"t={current_time:.1f}: λ={lambda_t:.4f}/yr, p_step={p_event:.6f}, "
            f"Σm={components['total_geom_moment']:.2e}"
        )

    if np.random.random() > p_event:
        return None, m_current  # No event

    # Event occurs! Generate details

    # 1. Draw magnitude
    magnitude = draw_magnitude(config)

    # 2. Compute spatial probability
    p_spatial, gamma_used = spatial_probability(m_current, magnitude, config)

    # 3. Sample hypocenter location
    hypocenter_idx = np.random.choice(config.n_elements, p=p_spatial)

    # 4. Generate slip distribution
    slip, ruptured_elements = generate_slip_distribution(
        hypocenter_idx, magnitude, m_current, mesh, config
    )

    # 5. Compute actual moment
    M0_actual = config.shear_modulus_Pa * np.sum(slip * config.element_area_m2)
    M_actual = seismic_moment_to_magnitude(M0_actual)

    # 6. Update moment distribution
    from moment import release_moment

    m_updated = release_moment(m_current, slip, config.element_area_m2)

    # 7. Create event record
    hypo_x, hypo_z = (
        mesh["centroids"][hypocenter_idx, 0],
        mesh["centroids"][hypocenter_idx, 2],
    )

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
        "temporal_components": components,
    }

    return event, m_updated
