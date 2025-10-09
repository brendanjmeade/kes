"""
Temporal probability with loading, Omori, and depletion
"""

import numpy as np
from moment import magnitude_to_seismic_moment


def compute_C_a(config):
    """
    Compute loading coefficient from moment balance

    Strategy: Target a realistic average event rate, then work backwards
    """
    # Target: ~20 events per 1000 years = 0.02 events/year
    # (This is more realistic for a 200km fault)
    # target_rate = 0.02  # events per year

    # # Expected geometric moment after full earthquake cycle
    # # Assume cycle time ~ 50 years
    # cycle_time = 50.0  # years

    # total_slip_rate = config.background_slip_rate_m_yr  # m/year
    # total_area = config.n_elements * config.element_area_m2  # m²
    # geom_moment_rate = total_slip_rate * total_area  # m³/year

    # # Geometric moment at end of cycle
    # geom_moment_at_cycle_end = geom_moment_rate * cycle_time  # m³

    # # We want: C_a × geom_moment_at_cycle_end ~ target_rate
    # # But need to account for tanh saturation
    # # If tanh(gamma × C_a × m) ~ 1, then need gamma × C_a × m ~ 2-3
    # # So: C_a = 2.5 / (gamma × geom_moment_at_cycle_end)

    # C_a = 2.5 / (config.gamma_temporal * geom_moment_at_cycle_end)

    # print(f"  Geometric moment rate: {geom_moment_rate:.2e} m³/year")
    # print(f"  Cycle time: {cycle_time} years")
    # print(f"  Geom moment at cycle end: {geom_moment_at_cycle_end:.2e} m³")

    # Expected geometric moment after long time (1000 years)
    target_rate_baseline = 0.001
    long_time = 1000.0  # years

    total_slip_rate = config.background_slip_rate_m_yr
    total_area = config.n_elements * config.element_area_m2
    geom_moment_rate = total_slip_rate * total_area

    geom_moment_long_time = geom_moment_rate * long_time

    # We want: tanh(gamma × C_a × m) ~ target_rate_baseline when m is at long_time value
    # For small arguments: tanh(x) ≈ x
    # So: gamma × C_a × m ~ target_rate_baseline
    # C_a = target_rate_baseline / (gamma × m)

    C_a = target_rate_baseline / (config.gamma_temporal * geom_moment_long_time)

    print(f"  Geometric moment rate: {geom_moment_rate:.2e} m³/year")
    print(f"  Long-time geom moment: {geom_moment_long_time:.2e} m³")
    print(f"  Target baseline rate: {target_rate_baseline} events/year")

    return C_a


def compute_C_r(config, C_a):
    """
    Compute depletion coefficient

    Strategy: A large event cluster should suppress activity for ~20-50 years
    """
    # Recovery time after major event
    tau_recovery_years = 30.0

    # Characteristic event magnitude
    M_char_magnitude = 7.0
    M_char = magnitude_to_seismic_moment(M_char_magnitude)  # N·m

    # After this event, want depletion to balance accumulation for tau_recovery years
    # Accumulation during recovery: C_a × (geom_moment_rate × tau_recovery)
    geom_moment_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )
    accumulated_during_recovery = geom_moment_rate * tau_recovery_years

    # Depletion term should roughly equal accumulation during recovery
    # C_r × M_char^psi ~ C_a × accumulated_during_recovery

    C_r = (C_a * accumulated_during_recovery) / (M_char**config.psi)

    # Scale up by factor to make depletion more effective
    C_r *= 5.0  # Empirical adjustment

    print(f"  M_char: M{M_char_magnitude} = {M_char:.2e} N·m")
    print(f"  Recovery time: {tau_recovery_years} years")

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
