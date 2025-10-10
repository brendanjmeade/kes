"""
Temporal probability with loading, Omori, and depletion
Using exponential formulation from statistical mechanics
"""

import numpy as np
from moment import magnitude_to_seismic_moment, seismic_moment_to_magnitude


def compute_exponential_rate_parameters(config):
    """
    Compute lambda_0, beta, and C_a for exponential rate equation

    λ(t) = λ₀ × exp(β × r_total)

    Derives all parameters from moment balance requirement
    """
    print("DEBUG: Using NEW version with r_typical = 1.0")
    # === Step 1: Required average rate from moment balance ===

    geom_loading_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )  # m³/year
    seismic_loading_rate = config.shear_modulus_Pa * geom_loading_rate  # N·m/year

    # Average M0 per event (dominated by large events in G-R)
    M0_max = magnitude_to_seismic_moment(config.M_max)
    M0_avg = M0_max / (1.5 + config.b_value)

    # Required average event rate for moment balance
    lambda_required = seismic_loading_rate / M0_avg  # events/year

    print(f"\n=== Exponential Rate Parameters (from moment balance) ===")
    print(f"  Geometric loading rate: {geom_loading_rate:.2e} m³/year")
    print(f"  Seismic loading rate: {seismic_loading_rate:.2e} N·m/year")
    print(f"  Average event magnitude: M {seismic_moment_to_magnitude(M0_avg):.1f}")
    print(f"  Average event M0: {M0_avg:.2e} N·m")
    print(f"  Required average rate: {lambda_required:.4f} events/year")
    print(f"  = 1 event per {1 / lambda_required:.1f} years")

    # === Step 2: Estimate typical r_total ===

    # In steady state, moment accumulates for approximately half the recurrence interval
    recurrence_interval = 1.0 / lambda_required  # years
    typical_accumulation_time = recurrence_interval / 2  # years
    typical_geom_moment = geom_loading_rate * typical_accumulation_time  # m³

    # CRITICAL: Make r_typical ~ O(1) so beta is reasonable
    # We'll set C_a such that C_a × typical_geom_moment ~ 1
    r_typical = 1.0  # Target dimensionless value

    print(f"  Typical recurrence interval: {recurrence_interval:.1f} years")
    print(f"  Mid-cycle accumulated moment: {typical_geom_moment:.2e} m³")
    print(f"  Target typical r_total: {r_typical:.2e}")

    # === Step 3: Choose beta for reasonable sensitivity ===

    # We want: exp(beta × r_typical) ~ amplification_factor
    # With r_typical = 1, this becomes: exp(beta) ~ amplification_factor
    # So: beta = ln(amplification_factor)
    amplification_factor = 10.0

    beta = np.log(amplification_factor)

    print(f"  Target amplification at typical state: {amplification_factor:.0f}×")
    print(f"  Beta: {beta:.2f}")

    # === Step 4: Solve for lambda_0 ===

    # At typical state: lambda_required = lambda_0 × exp(beta × r_typical)
    # Therefore: lambda_0 = lambda_required / amplification_factor

    lambda_0 = lambda_required / amplification_factor

    print(f"  Lambda_0 (background rate): {lambda_0:.6f} events/year")
    print(f"  = 1 event per {1 / lambda_0:.0f} years when r_total = 0")

    # === Step 5: Compute C_a to achieve r_typical at mid-cycle ===

    # We want: C_a × typical_geom_moment = r_typical
    C_a = r_typical / typical_geom_moment

    print(f"  C_a: {C_a:.2e} (m³·year)⁻¹")
    print(
        f"  At mid-cycle: r_accumulation = C_a × {typical_geom_moment:.2e} = {r_typical:.2e}"
    )

    # === Step 6: Compute C_r from depletion theory ===

    tau_recovery_years = 30.0
    M_char_magnitude = config.M_max - 1.0
    M_char = magnitude_to_seismic_moment(M_char_magnitude)

    accumulated_during_recovery = geom_loading_rate * tau_recovery_years

    C_r_base = (C_a * accumulated_during_recovery) / (M_char**config.psi)
    C_r = C_r_base * 5.0

    print(f"\n=== Depletion Parameters ===")
    print(f"  Characteristic event: M {M_char_magnitude:.1f} = {M_char:.2e} N·m")
    print(f"  Recovery timescale: {tau_recovery_years} years")
    print(f"  C_r: {C_r:.2e} (N·m)^(-{config.psi:.3f}) year⁻¹")

    # === Verification ===

    print(f"\n=== Verification ===")
    print(f"  When r_total = 0: λ = {lambda_0:.6f}/yr")
    print(
        f"  When r_total = {r_typical:.2f}: λ = {lambda_0 * np.exp(beta * r_typical):.6f}/yr"
    )
    print(f"  Target average rate: {lambda_required:.6f}/yr")

    if abs(lambda_0 * np.exp(beta * r_typical) - lambda_required) < 0.0001:
        print(f"  ✓ Rates match!")
    else:
        print(f"  ✗ WARNING: Rates don't match!")

    print(f"  Moment balance: {seismic_loading_rate:.2e} N·m/yr in")
    print(f"                = {lambda_required:.6f}/yr × {M0_avg:.2e} N·m out")

    return lambda_0, beta, C_a, C_r


def temporal_probability(m_current, event_history, current_time, config):
    """
    Compute λ(t) - temporal event rate using exponential formulation

    λ(t) = λ₀ × exp(β × r_total)

    where r_total = r_accumulation + r_omori + r_depletion

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
        if 0 < dt_days < 365.25 * 10:  # Only include last 10 years
            # Productivity scales with magnitude
            beta_omori = config.omori_beta_0 * 10 ** (
                config.omori_alpha_beta * event["magnitude"]
            )

            # Omori decay
            r_omori += beta_omori / (dt_days + config.omori_c_days) ** config.omori_p

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

    # Combine all components
    r_total = r_accumulation + r_omori + r_depletion

    # EXPONENTIAL FORM - no artificial saturation!
    lambda_t = config.lambda_0 * np.exp(config.beta_rate * r_total)

    # Safety bounds (should rarely be needed with proper calibration)
    lambda_t = np.clip(lambda_t, 1e-6, 10.0)

    # DEBUG: Print occasionally
    if (
        current_time > 0
        and int(current_time) % 100 == 0
        and abs(current_time - int(current_time)) < 0.01
    ):
        print(
            f"\nt={current_time:.1f}yr: Σm={total_geom_moment:.2e} m³, "
            f"r_acc={r_accumulation:.2e}, r_omori={r_omori:.2e}, r_depl={r_depletion:.2e}, "
            f"r_tot={r_total:.2e}, λ={lambda_t:.4f}/yr"
        )

    components = {
        "r_accumulation": r_accumulation,
        "r_omori": r_omori,
        "r_depletion": r_depletion,
        "r_total": r_total,
        "total_geom_moment": total_geom_moment,
    }

    if len(event_history) > 3 and current_time > 93 and current_time < 94:
        print(
            f"t={current_time:.3f}: r_acc={r_accumulation:.2e}, r_omori={r_omori:.2e}, "
            f"r_depl={r_depletion:.2e}, r_tot={r_total:.2e}, λ={lambda_t:.4f}/yr"
        )

    return lambda_t, components
