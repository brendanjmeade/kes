"""
Temporal probability/rate functions for earthquake generation

Adaptive rate formulation: Rate self-corrects to achieve moment balance
"""

import numpy as np
from moment import magnitude_to_seismic_moment


def compute_rate_parameters(config):
    """
    Compute initial rate parameters based on moment balance

    The rate will be adaptively corrected during simulation to ensure
    perfect moment balance regardless of magnitude distribution details

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

    # Estimate a "characteristic" recurrence time
    M0_char = magnitude_to_seismic_moment(config.M_max)
    geom_moment_char = M0_char / config.shear_modulus_Pa
    recurrence_time_char = geom_moment_char / geom_loading_rate

    # At equilibrium, accumulated moment is roughly half-cycle
    geom_moment_equilibrium = geom_loading_rate * (recurrence_time_char / 2)

    # Choose initial target rate at equilibrium
    # Start conservative - adaptive correction will optimize this
    if config.M_min < 5.0:
        lambda_target = 5.0  # events/year
    elif config.M_min < 6.0:
        lambda_target = 2.0
    elif config.M_min < 7.0:
        lambda_target = 0.5
    else:
        lambda_target = 0.2

    # Compute initial C
    C = lambda_target / geom_moment_equilibrium

    # Store for diagnostics
    config.C_rate_base = C
    config.geom_moment_equilibrium = geom_moment_equilibrium
    config.lambda_target = lambda_target
    config.recurrence_time_char = recurrence_time_char

    # Initialize adaptive correction factor
    config.rate_correction_factor = 1.0
    config.coupling_history = []

    print("\n" + "=" * 70)
    print("ADAPTIVE MOMENT-BASED RATE MODEL")
    print("=" * 70)
    print(f"  Geometric loading rate: {geom_loading_rate:.2e} m³/yr")
    print(
        f"  Seismic loading rate: {config.shear_modulus_Pa * geom_loading_rate:.2e} N·m/yr"
    )
    print(f"  Characteristic M_max event: M {config.M_max:.1f} ({M0_char:.2e} N·m)")
    print(f"  Estimated recurrence time: {recurrence_time_char:.1f} years")
    print(f"\n  Equilibrium accumulated moment: {geom_moment_equilibrium:.2e} m³")
    print(f"  Initial target rate at equilibrium: {lambda_target:.3f} events/year")
    print(f"  Base rate coefficient C: {C:.3e} (events/yr)/(m³)")
    print(f"\n  λ(t) = C × correction_factor(t) × moment_deficit(t)")
    print(f"  Rate will adapt to maintain moment balance")
    print("=" * 70)

    return C


def update_rate_correction(
    config, cumulative_loading, cumulative_release, current_time
):
    """
    Update adaptive rate correction factor based on observed coupling

    Uses proportional control to drive coupling toward 1.0

    Parameters:
    -----------
    config : Config object
    cumulative_loading : float
        Total geometric moment loaded (m³)
    cumulative_release : float
        Total geometric moment released (m³)
    current_time : float
        Current simulation time (years)

    Returns:
    --------
    None (updates config.rate_correction_factor in place)
    """
    if cumulative_loading <= 0:
        return

    # Compute observed coupling
    observed_coupling = cumulative_release / cumulative_loading
    config.coupling_history.append(
        {
            "time": current_time,
            "coupling": observed_coupling,
            "correction_factor": config.rate_correction_factor,
        }
    )

    # Target coupling
    target_coupling = 1.0
    coupling_error = target_coupling - observed_coupling

    # Proportional control with conservative gain
    # Increase rate if under-releasing, decrease if over-releasing
    gain = 0.1  # Conservative - adjust slowly
    adjustment = gain * coupling_error

    config.rate_correction_factor += adjustment

    # Bound correction factor to reasonable range
    config.rate_correction_factor = max(0.1, min(10.0, config.rate_correction_factor))

    # Log adjustment
    if len(config.coupling_history) % 10 == 0:  # Every 10 updates
        print(
            f"  Rate correction update at t={current_time:.0f} yr: "
            f"coupling={observed_coupling:.3f}, "
            f"correction_factor={config.rate_correction_factor:.3f}"
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

    # Aftershock rate (Omori decay) - optional
    lambda_aftershock = 0.0
    if len(event_history) > 0 and hasattr(config, "omori_law") and config.omori_law:
        for event in event_history:
            dt = current_time - event["time"]
            if dt > 0:
                M0_event = event["M0"]
                omori_amplitude = config.omori_c_rate * (
                    M0_event / magnitude_to_seismic_moment(config.M_max)
                )
                lambda_aftershock += omori_amplitude / (dt + config.omori_c_time)

    # Total rate
    lambda_t = lambda_loading + lambda_aftershock
    lambda_t = max(0.0, lambda_t)

    # Components for diagnostics
    components = {
        "loading": lambda_loading,
        "aftershock": lambda_aftershock,
        "moment_deficit": moment_deficit,
        "correction_factor": config.rate_correction_factor,
    }

    return lambda_t, components
