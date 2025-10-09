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


def spatial_probability(m_current, magnitude, config):
    """
    Compute p(i|M) - spatial nucleation probability given magnitude

    p(i|M) ∝ m_i^γ(M)

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

    # Power law
    weights = m_safe**gamma

    # Normalize
    p = weights / np.sum(weights)

    return p, gamma
