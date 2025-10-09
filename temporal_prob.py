"""
Temporal probability with loading, Omori, and depletion
"""

import numpy as np
from moment import magnitude_to_seismic_moment  # Move import to top


def compute_C_a(config):
    """
    Compute loading coefficient from moment balance

    This is a placeholder - should be calibrated
    """
    # Rough estimate: want average rate ~ 1/100 years = 0.01/year
    # C_a × <m> ~ 0.01
    # Typical <m> ~ total_moment_rate × 100 years / n_elements

    typical_m = (config.total_moment_rate * 100) / config.n_elements
    C_a = 0.01 / typical_m

    return C_a


def compute_C_r(config, C_a):
    """
    Compute depletion coefficient from theory

    From discussion: C_r ~ (C_a × tau_recovery × M_dot) / M_char^psi
    """
    # Assumptions
    tau_recovery_years = 20.0
    M_char = magnitude_to_seismic_moment(7.5)  # Characteristic large event

    C_r = (C_a * tau_recovery_years * config.total_moment_rate) / (M_char**config.psi)

    return C_r


def temporal_probability(m_current, event_history, current_time, config):
    """
    Compute λ(t) - temporal event rate

    λ(t) = f[C_a × Σm_i + Σ(Omori) - C_r × M_cum^ψ]

    Returns:
    --------
    lambda_t : event rate
    components : dict with individual components
    """
    # Component 1: Loading
    total_moment = np.sum(m_current)
    r_accumulation = config.C_a * total_moment

    # Component 2: Omori aftershocks
    r_omori = 0.0
    for event in event_history:
        dt_days = (current_time - event["time"]) * 365.25
        if dt_days > 0:
            # Productivity scales with magnitude
            beta = config.omori_beta_0 * 10 ** (
                config.omori_alpha_beta * event["magnitude"]
            )

            # Omori decay
            r_omori += beta / (dt_days + config.omori_c_days) ** config.omori_p

    # Component 3: Moment depletion
    if len(event_history) > 0:
        M_cumulative = sum([event["M0"] for event in event_history])
        r_depletion = -config.C_r * M_cumulative**config.psi
    else:
        r_depletion = 0.0

    # Combine
    r_total = r_accumulation + r_omori + r_depletion

    # Wrapping function (tanh with bounds)
    if r_total > 0:
        lambda_t = np.tanh(config.gamma_temporal * r_total)
    else:
        lambda_t = 0.0

    lambda_t = np.clip(lambda_t, config.lambda_min, config.lambda_max)

    components = {
        "r_accumulation": r_accumulation,
        "r_omori": r_omori,
        "r_depletion": r_depletion,
        "r_total": r_total,
    }

    return lambda_t, components
