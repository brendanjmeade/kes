"""
Temporal probability with loading, Omori, and depletion
"""

import numpy as np
from moment import magnitude_to_seismic_moment


def compute_C_a(config):
    """
    Compute loading coefficient from moment balance

    C_a should convert geometric moment (m³) to event rate (year⁻¹)

    Dimensional analysis:
    [C_a] = year⁻¹ / m³

    Physical reasoning:
    - When total geometric moment reaches "characteristic" level, want λ ~ 1/recurrence_time
    """
    # Target: ~10 events per 100 years = 0.1 events/year
    target_rate = 0.1  # events per year

    # Characteristic geometric moment after ~10 years of loading
    # Total geometric moment rate = slip_rate × total_area
    total_slip_rate = config.background_slip_rate_m_yr  # m/year
    total_area = config.n_elements * config.element_area_m2  # m²
    geom_moment_rate = total_slip_rate * total_area  # m³/year

    # After 10 years
    typical_geom_moment = geom_moment_rate * 10.0  # m³

    # Set C_a so that this gives target rate
    C_a = target_rate / typical_geom_moment

    print(f"  Geometric moment rate: {geom_moment_rate:.2e} m³/year")
    print(f"  Typical geom moment (10 yr): {typical_geom_moment:.2e} m³")

    return C_a


def compute_C_r(config, C_a):
    """
    Compute depletion coefficient from theory

    C_r should convert seismic moment (N·m) to rate suppression (year⁻¹)

    From discussion: recovery time ~ 20 years after typical event
    """
    # Assumptions
    tau_recovery_years = 20.0
    M_char = magnitude_to_seismic_moment(7.5)  # N·m

    # Want: C_r × M_char^psi ≈ C_a × typical_geom_moment
    # So that depletion balances accumulation

    geom_moment_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )
    typical_geom_moment = geom_moment_rate * tau_recovery_years

    C_r = (C_a * typical_geom_moment) / (M_char**config.psi)

    print(f"  M_char: {M_char:.2e} N·m")

    return C_r


def temporal_probability(m_current, event_history, current_time, config):
    """
    Compute λ(t) - temporal event rate

    Returns:
    --------
    lambda_t : event rate (events per year)
    components : dict with individual components
    """
    # Component 1: Loading (from geometric moment)
    total_geom_moment = np.sum(m_current)  # m³
    r_accumulation = config.C_a * total_geom_moment  # year⁻¹

    # Component 2: Omori aftershocks
    r_omori = 0.0
    for event in event_history:
        dt_days = (current_time - event["time"]) * 365.25
        if dt_days > 0 and dt_days < 365.25 * 10:  # Only include last 10 years
            # Productivity scales with magnitude
            beta = config.omori_beta_0 * 10 ** (
                config.omori_alpha_beta * event["magnitude"]
            )

            # Omori decay
            r_omori += beta / (dt_days + config.omori_c_days) ** config.omori_p

    # Component 3: Moment depletion (from seismic moment)
    if len(event_history) > 0:
        # Only include recent events (last 50 years for memory)
        recent_events = [e for e in event_history if (current_time - e["time"]) < 50.0]
        if len(recent_events) > 0:
            M_cumulative = sum([e["M0"] for e in recent_events])  # N·m
            r_depletion = -config.C_r * M_cumulative**config.psi
        else:
            r_depletion = 0.0
    else:
        r_depletion = 0.0

    # Combine
    r_total = r_accumulation + r_omori + r_depletion

    # Wrapping function (tanh with bounds) - but scale r_total first
    # The issue is that r_total needs to be O(1) for tanh to work properly
    scaled_r = config.gamma_temporal * r_total

    if scaled_r > 0:
        lambda_t = np.tanh(scaled_r)
    else:
        lambda_t = config.lambda_min

    lambda_t = np.clip(lambda_t, config.lambda_min, config.lambda_max)

    # DEBUG: Print occasionally
    if (
        current_time > 0
        and int(current_time) % 10 == 0
        and abs(current_time - int(current_time)) < 0.01
    ):
        print(
            f"\nt={current_time:.1f}yr: Σm={total_geom_moment:.2e} m³, "
            f"r_acc={r_accumulation:.2e}, r_omori={r_omori:.2e}, r_depl={r_depletion:.2e}, "
            f"λ={lambda_t:.4f}/yr"
        )

    components = {
        "r_accumulation": r_accumulation,
        "r_omori": r_omori,
        "r_depletion": r_depletion,
        "r_total": r_total,
        "total_geom_moment": total_geom_moment,
    }

    return lambda_t, components
