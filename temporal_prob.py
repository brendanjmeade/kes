"""
Temporal earthquake rate (not probability!) with loading, Omori, and depletion
Rate-based approach: λ(t) gives expected events per unit time
"""

import numpy as np
from moment import magnitude_to_seismic_moment, seismic_moment_to_magnitude


def compute_rate_parameters(config):
    """
    Compute parameters for linear rate model

    λ(t) = λ₀ + C_a × Σm(t) + Σ(Omori) - C_r × M_cum^ψ

    All parameters derived from moment balance
    """
    print("\n=== LINEAR RATE MODEL (not exponential!) ===")

    # === Step 1: Required average rate from moment balance ===

    geom_loading_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )  # m³/year
    seismic_loading_rate = config.shear_modulus_Pa * geom_loading_rate  # N·m/year

    # Average M0 per event from Gutenberg-Richter
    M0_max = magnitude_to_seismic_moment(config.M_max)
    M0_avg = M0_max / (1.5 + config.b_value)

    lambda_required = seismic_loading_rate / M0_avg  # events/year

    print(f"\n=== Target Rate from Moment Balance ===")
    print(f"  Geom loading rate: {geom_loading_rate:.2e} m³/yr")
    print(f"  Seismic loading rate: {seismic_loading_rate:.2e} N·m/yr")
    print(
        f"  Average event: M {seismic_moment_to_magnitude(M0_avg):.1f} ({M0_avg:.2e} N·m)"
    )
    print(
        f"  Required rate: {lambda_required:.6f} events/yr = 1 per {1 / lambda_required:.0f} yr"
    )

    # === Step 2: Decompose rate into components ===

    recurrence_time = 1.0 / lambda_required
    typical_geom_moment = geom_loading_rate * (recurrence_time / 2)

    # Background rate (when no moment accumulated, no aftershocks, no depletion)
    lambda_0 = lambda_required * 0.1  # 10% background

    # Loading coefficient: contributes 90% of rate at mid-cycle
    # At mid-cycle: C_a × typical_geom_moment = 0.9 × lambda_required
    C_a = (0.9 * lambda_required) / typical_geom_moment

    print(f"\n=== Rate Components ===")
    print(
        f"  Lambda_0 (background): {lambda_0:.6f} events/yr ({lambda_0 / lambda_required:.1%} of total)"
    )
    print(f"  C_a (loading coeff): {C_a:.2e} (m³·yr)⁻¹")
    print(f"  At mid-cycle moment ({typical_geom_moment:.2e} m³):")
    print(
        f"    λ = {lambda_0:.6f} + {C_a * typical_geom_moment:.6f} = {lambda_0 + C_a * typical_geom_moment:.6f} events/yr"
    )

    # === Step 3: Depletion to balance loading ===

    tau_recovery = recurrence_time
    M_char_magnitude = config.M_max - 0.5
    M_char = magnitude_to_seismic_moment(M_char_magnitude)

    accumulated_during_recovery = geom_loading_rate * tau_recovery

    # After M_char event, depletion should suppress rate by comparable amount to loading
    # -C_r × M_char^ψ ≈ C_a × accumulated_during_recovery
    C_r = (C_a * accumulated_during_recovery) / (M_char**config.psi)

    print(f"\n=== Depletion Parameters ===")
    print(f"  Characteristic event: M {M_char_magnitude:.1f} ({M_char:.2e} N·m)")
    print(f"  Recovery time: {tau_recovery:.1f} years")
    print(f"  C_r: {C_r:.2e}")
    print(f"  After M {M_char_magnitude:.1f}:")
    print(f"    Depletion = -{C_r * M_char**config.psi:.6f} events/yr")
    print(
        f"    Ratio to loading: {(C_r * M_char**config.psi) / (C_a * typical_geom_moment):.2f}×"
    )

    # === Verification ===

    print(f"\n=== Self-Consistency Check ===")
    print(f"  At t=0 (no moment): λ = {lambda_0:.6f} events/yr")
    print(f"  At mid-cycle: λ = {lambda_0 + C_a * typical_geom_moment:.6f} events/yr")
    print(f"  Target average rate: {lambda_required:.6f} events/yr")

    print(f"\n=== Moment Balance ===")
    print(f"  Input: {seismic_loading_rate:.2e} N·m/yr")
    print(
        f"  Output: {lambda_required:.6f}/yr × {M0_avg:.2e} N·m = {lambda_required * M0_avg:.2e} N·m/yr"
    )
    print(f"  Ratio: {(lambda_required * M0_avg) / seismic_loading_rate:.3f}")

    # === Rate examples ===

    print(f"\n=== Rate Examples (LINEAR model) ===")
    for m_example in [
        0,
        typical_geom_moment / 2,
        typical_geom_moment,
        typical_geom_moment * 2,
    ]:
        rate = lambda_0 + C_a * m_example
        print(f"  Σm = {m_example:.2e} m³: λ = {rate:.6f} events/yr")

    return lambda_0, C_a, C_r


def earthquake_rate(m_current, event_history, current_time, config):
    """
    Compute instantaneous earthquake rate λ(t) in events/year

    LINEAR MODEL: λ(t) = λ₀ + C_a·Σm + Σ(Omori) - C_r·M_cum^ψ

    Returns:
    --------
    lambda_t : earthquake rate (events per year)
    components : dict with breakdown
    """
    # Component 1: Background
    rate_background = config.lambda_0

    # Component 2: Loading (from geometric moment)
    total_geom_moment = np.sum(m_current)  # m³
    rate_loading = config.C_a * total_geom_moment  # events/year

    # Component 3: Omori aftershocks
    rate_omori = 0.0
    for event in event_history:
        dt_days = (current_time - event["time"]) * 365.25
        if 0 < dt_days < 365.25 * 10:  # Only last 10 years
            # Productivity scales with magnitude
            beta_omori = config.omori_beta_0 * 10 ** (
                config.omori_alpha_beta * event["magnitude"]
            )

            # Omori decay gives RATE (events/year)
            # Need to convert from days⁻¹ to year⁻¹
            rate_omori += (
                beta_omori / (dt_days + config.omori_c_days) ** config.omori_p * 365.25
            )

    # Component 4: Moment depletion (from seismic moment)
    if len(event_history) > 0:
        # Only recent events (last 50 years)
        recent_events = [e for e in event_history if (current_time - e["time"]) < 50.0]
        if len(recent_events) > 0:
            M_cumulative = sum([e["M0"] for e in recent_events])  # N·m
            rate_depletion = -config.C_r * M_cumulative**config.psi
        else:
            rate_depletion = 0.0
    else:
        rate_depletion = 0.0

    # Combine (linear sum!)
    lambda_t = rate_background + rate_loading + rate_omori + rate_depletion

    # Floor at minimum rate (can't be negative)
    lambda_t = max(config.lambda_min, lambda_t)

    # DEBUG: Print occasionally
    if (
        current_time > 0
        and int(current_time) % 100 == 0
        and abs(current_time - int(current_time)) < 0.01
    ):
        print(f"\nt={current_time:.1f}yr: Σm={total_geom_moment:.2e} m³")
        print(
            f"  λ = {rate_background:.6f} (bkgd) + {rate_loading:.6f} (load) + "
            f"{rate_omori:.6f} (omori) + {rate_depletion:.6f} (depl) = {lambda_t:.6f} events/yr"
        )

    components = {
        "rate_background": rate_background,
        "rate_loading": rate_loading,
        "rate_omori": rate_omori,
        "rate_depletion": rate_depletion,
        "total_geom_moment": total_geom_moment,
    }

    return lambda_t, components
