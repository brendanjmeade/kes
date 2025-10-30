"""
Temporal probability/rate functions for earthquake generation

Adaptive rate formulation: Rate self-corrects to achieve moment balance
"""

import numpy as np
from moment import magnitude_to_seismic_moment


def compute_expected_moment_per_event(config):
    """
    Compute expected geometric moment per event from G-R distribution

    Integrates moment × probability over the magnitude range to get
    the average moment released per event, accounting for:
    - G-R distribution (b-value)
    - Magnitude bounds [M_min, M_max]

    This allows setting C analytically instead of using ad-hoc target rates.

    Parameters:
    -----------
    config : Config object

    Returns:
    --------
    expected_geom_moment : float
        Expected geometric moment per event (m³)
    """
    # Sample magnitude range finely
    M_array = np.linspace(config.M_min, config.M_max, 1000)
    dM = M_array[1] - M_array[0]

    # Gutenberg-Richter probability density
    # P(M) ∝ 10^(-b×M)
    # Normalize over [M_min, M_max]
    b = config.b_value
    P_unnormalized = 10 ** (-b * M_array)
    P_normalized = P_unnormalized / (np.sum(P_unnormalized) * dM)

    # Convert magnitudes to geometric moments
    M0_array = magnitude_to_seismic_moment(M_array)  # N·m (seismic)
    geom_moment_array = M0_array / config.shear_modulus_Pa  # m³ (geometric)

    # Expected value: E[M] = ∫ M × P(M) dM
    expected_geom_moment = np.sum(geom_moment_array * P_normalized * dM)

    return expected_geom_moment


def compute_rate_parameters(config):
    """
    Compute initial rate parameters based on moment balance

    Uses analytical integration of G-R distribution to estimate average
    moment per event, allowing C to be set without ad-hoc rate guesses.

    If adaptive correction is enabled, C will be further refined during
    simulation to ensure perfect moment balance.

    Parameters:
    -----------
    config : Config object

    Returns:
    --------
    C : float
        Base rate coefficient (events/year per m³ of accumulated moment)
    """

    # Estimate equilibrium accumulated moment
    geom_loading_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )

    # Method 1: Analytical from G-R distribution (NEW - better!)
    expected_geom_moment_per_event = compute_expected_moment_per_event(config)

    # Estimate equilibrium: balance requires loading_rate = lambda × <M_event>
    # At equilibrium with deficit D: lambda = C × D
    # So: loading_rate = C × D × <M_event>
    # Assuming D ≈ half-cycle worth of moment
    M0_char = magnitude_to_seismic_moment(config.M_max)
    geom_moment_char = M0_char / config.shear_modulus_Pa
    recurrence_time_char = geom_moment_char / geom_loading_rate
    geom_moment_equilibrium = geom_loading_rate * (recurrence_time_char / 2)

    # Choose target rate at equilibrium based on magnitude range
    if config.M_min < 5.0:
        lambda_target = 5.0  # events/year (many small events)
    elif config.M_min < 6.0:
        lambda_target = 2.0
    elif config.M_min < 7.0:
        lambda_target = 0.5
    else:
        lambda_target = 0.2  # few large events

    # Compute C using analytical expected moment
    # At equilibrium: geom_loading_rate = lambda_target × expected_moment_per_event
    # And lambda_target = C × geom_moment_equilibrium
    # So: C = geom_loading_rate / (geom_moment_equilibrium × expected_moment_per_event)
    C_analytical = geom_loading_rate / (geom_moment_equilibrium * expected_geom_moment_per_event)

    # Method 2: Old approach using M_max (kept for comparison)
    C_old = lambda_target / geom_moment_equilibrium

    # Use analytical method
    C = C_analytical

    # Store for diagnostics
    config.C_rate_base = C
    config.C_rate_old = C_old  # Old method for comparison
    config.expected_geom_moment_per_event = expected_geom_moment_per_event
    config.geom_loading_rate = geom_loading_rate
    config.geom_moment_equilibrium = geom_moment_equilibrium
    config.lambda_target = lambda_target
    config.recurrence_time_char = recurrence_time_char

    # Initialize adaptive correction factor
    config.rate_correction_factor = 1.0
    config.coupling_history = []  # Store periodically (every 100 years) for diagnostics

    print("\n" + "=" * 70)
    print("MOMENT-BASED RATE MODEL")
    print("=" * 70)
    print(f"  Geometric loading rate: {geom_loading_rate:.2e} m³/yr")
    print(
        f"  Seismic loading rate: {config.shear_modulus_Pa * geom_loading_rate:.2e} N·m/yr"
    )
    print(f"  Characteristic M_max event: M {config.M_max:.1f} ({M0_char:.2e} N·m)")
    print(f"  Estimated recurrence time: {recurrence_time_char:.1f} years")
    print(f"\n  Equilibrium accumulated moment: {geom_moment_equilibrium:.2e} m³")
    print(f"  Expected moment per event (G-R analytical): {expected_geom_moment_per_event:.2e} m³")
    print(f"  Target rate at equilibrium: {lambda_target:.3f} events/year")
    print(f"\n  Base rate coefficient C (analytical): {C:.3e} (events/yr)/(m³)")
    print(f"  Base rate coefficient C (old method): {C_old:.3e} (events/yr)/(m³)")
    print(f"  Improvement ratio: {C_old/C:.2f}x")
    print(f"\n  λ(t) = C × correction_factor(t) × moment_deficit(t) + λ_aftershock(t)")

    # Print adaptive correction status
    if hasattr(config, "adaptive_correction_enabled") and config.adaptive_correction_enabled:
        print(f"  ADAPTIVE CORRECTION: ENABLED (continuous updates every timestep)")
        print(f"    Gain: {config.adaptive_correction_gain}")
        print(f"    Will drive coupling → 1.0")
    else:
        print(f"  ADAPTIVE CORRECTION: DISABLED (fixed C, natural coupling)")
        print(f"    Coupling will depend on G-R distribution and slip heterogeneity")

    # Print Omori aftershock parameters if enabled
    if hasattr(config, "omori_enabled") and config.omori_enabled:
        print(f"\n  OMORI AFTERSHOCKS ENABLED:")
        print(f"    Law: λ_aftershock = K / (t + c)^p")
        print(f"    p = {config.omori_p:.2f}")
        print(f"    c = {config.omori_c_years:.6f} years")
        print(f"    K_ref = {config.omori_K_ref:.3f} events/yr (at M={config.omori_M_ref:.1f})")
        print(f"    α = {config.omori_alpha:.2f} (magnitude scaling)")
        print(f"    Duration: {config.omori_duration_years:.1f} years per sequence")
        K_M7 = config.omori_K_ref * 10 ** (config.omori_alpha * (7.0 - config.omori_M_ref))
        print(f"    Example: M7.0 → K = {K_M7:.3f} events/yr")
    else:
        print(f"\n  OMORI AFTERSHOCKS DISABLED")
    print("=" * 70)

    return C


def update_rate_correction(
    config, cumulative_loading, cumulative_release, current_time, dt_years
):
    """
    Update adaptive rate correction factor based on observed coupling

    Uses continuous proportional control to drive coupling toward 1.0
    Updates every timestep (no artificial 100-year interval)

    Parameters:
    -----------
    config : Config object
    cumulative_loading : float
        Total geometric moment loaded (m³)
    cumulative_release : float
        Total geometric moment released (m³)
    current_time : float
        Current simulation time (years)
    dt_years : float
        Timestep size (years)

    Returns:
    --------
    None (updates config.rate_correction_factor in place)
    """
    # Skip if correction is disabled
    if not (hasattr(config, "adaptive_correction_enabled") and config.adaptive_correction_enabled):
        return

    if cumulative_loading <= 0:
        return

    # Compute observed coupling
    observed_coupling = cumulative_release / cumulative_loading

    # Target coupling
    target_coupling = 1.0
    coupling_error = target_coupling - observed_coupling

    # Continuous proportional control with gain from config
    # Increase rate if under-releasing, decrease if over-releasing
    # Multiply by dt to make adjustment continuous (not discrete)
    adjustment = config.adaptive_correction_gain * coupling_error * dt_years

    config.rate_correction_factor += adjustment

    # Bound correction factor to reasonable range from config
    config.rate_correction_factor = max(
        config.correction_factor_min,
        min(config.correction_factor_max, config.rate_correction_factor),
    )

    # Store coupling history periodically (every 100 years) for diagnostics
    # Avoid storing every timestep to save memory
    if int(current_time) % 100 == 0 and len(config.coupling_history) < int(current_time / 100) + 1:
        config.coupling_history.append(
            {
                "time": current_time,
                "coupling": observed_coupling,
                "correction_factor": config.rate_correction_factor,
            }
        )


def earthquake_rate(
    m_current,
    event_history,
    current_time,
    config,
    cumulative_loading,
    cumulative_release,
):
    """
    Compute instantaneous earthquake rate based on moment deficit

    λ(t) = C_base × correction_factor(t) × max(0, moment_deficit)

    The correction factor adapts to ensure moment balance

    Parameters:
    -----------
    m_current : array
        Current geometric moment (m³) at each element
    event_history : list
        List of past events
    current_time : float
        Current simulation time (years)
    config : Config object
    cumulative_loading : float
        Total geometric moment loaded since t=0 (m³)
    cumulative_release : float
        Total geometric moment released by events (m³)

    Returns:
    --------
    lambda_t : float
        Instantaneous earthquake rate (events/year)
    components : dict
        Breakdown of rate components
    """

    # Moment deficit (should always be ≥ 0)
    moment_deficit = cumulative_loading - cumulative_release
    moment_deficit = max(0.0, moment_deficit)

    # Base rate proportional to moment deficit, with adaptive correction
    lambda_loading = config.C_rate_base * config.rate_correction_factor * moment_deficit

    # Aftershock rate (Omori-Utsu decay)
    lambda_aftershock = 0.0
    n_active_sequences = 0

    if (
        len(event_history) > 0
        and hasattr(config, "omori_enabled")
        and config.omori_enabled
    ):
        # Use c directly in years
        omori_c_years = config.omori_c_years

        # Loop only over recent events (performance optimization)
        for event in event_history:
            dt_years = current_time - event["time"]

            # Only consider events within aftershock duration window
            if 0 < dt_years <= config.omori_duration_years:
                n_active_sequences += 1

                # Omori-Utsu law: λ(t) = K / (t + c)^p
                # K scales with mainshock magnitude: K = K_ref × 10^(alpha × (M - M_ref))
                M_mainshock = event["magnitude"]
                K = config.omori_K_ref * 10 ** (
                    config.omori_alpha * (M_mainshock - config.omori_M_ref)
                )

                # Add this mainshock's aftershock contribution
                lambda_aftershock += K / (dt_years + omori_c_years) ** config.omori_p

    # Total rate
    lambda_t = lambda_loading + lambda_aftershock
    lambda_t = max(0.0, lambda_t)

    # Components for diagnostics
    components = {
        "loading": lambda_loading,
        "aftershock": lambda_aftershock,
        "n_active_sequences": n_active_sequences,
        "moment_deficit": moment_deficit,
        "correction_factor": config.rate_correction_factor,
    }

    return lambda_t, components
