"""
Magnitude-dependent spatial probability from MaxEnt
"""

import numpy as np


def gamma_magnitude_dependent(magnitude, gamma_min, gamma_max, alpha, M_min):
    """
    Selectivity increases with magnitude

    gamma(M) = gamma_max - (gamma_max - gamma_min) × exp(-alpha × (M - M_min))
    """
    delta_M = magnitude - M_min
    gamma = gamma_max - (gamma_max - gamma_min) * np.exp(-alpha * delta_M)
    return gamma


def spatial_probability(m_current, magnitude, config, aftershock_weights=None):
    """
    Compute p(i|M) - spatial nucleation probability given magnitude

    Combines moment-based nucleation with aftershock spatial localization:
    p(i|M) ∝ m_i^γ(M) × w_aftershock(i)

    where w_aftershock comes from active mainshock sequences

    Parameters:
    -----------
    m_current : (n_elements,) array
        Current moment at each element
    magnitude : float
        Magnitude of event to generate
    config : Config
        Configuration
    aftershock_weights : (n_elements,) array, optional
        Spatial weighting from aftershock sequences (≥ 1.0)
        If None, uses uniform weights (no spatial bias)

    Returns:
    --------
    p : (n_elements,) probability distribution
    gamma : selectivity parameter used
    """
    # Magnitude-dependent gamma
    gamma = gamma_magnitude_dependent(
        magnitude,
        config.gamma_min,
        config.gamma_max,
        config.alpha_spatial,
        config.M_min,
    )

    # Ensure positive moments
    m_safe = np.maximum(m_current, 1e-10)

    # Base weight from moment deficit
    weights_moment = m_safe**gamma

    # Apply aftershock spatial weighting if provided
    if aftershock_weights is not None:
        weights_total = weights_moment * aftershock_weights
    else:
        weights_total = weights_moment

    # Normalize to probability distribution
    p = weights_total / np.sum(weights_total)

    return p, gamma
