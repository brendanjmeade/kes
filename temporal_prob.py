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

    Beta scaled to give reasonable amplification
    """
    print("DEBUG: Using FULLY PHYSICS-BASED version with scaled beta")

    # === Step 1: Required average rate from moment balance ===

    geom_loading_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )
    seismic_loading_rate = config.shear_modulus_Pa * geom_loading_rate

    M0_max = magnitude_to_seismic_moment(config.M_max)
    M0_avg = M0_max / (1.5 + config.b_value)

    lambda_required = seismic_loading_rate / M0_avg

    print(f"\n=== Physics-Based Rate Parameters ===")
    print(f"  Geom loading rate: {geom_loading_rate:.2e} m³/yr")
    print(f"  Seismic loading rate: {seismic_loading_rate:.2e} N·m/yr")
    print(
        f"  Average event: M {seismic_moment_to_magnitude(M0_avg):.1f} ({M0_avg:.2e} N·m)"
    )
    print(
        f"  Required rate: {lambda_required:.6f} events/yr = 1 per {1 / lambda_required:.0f} yr"
    )

    # === Step 2: Natural scales ===

    recurrence_time = 1.0 / lambda_required
    typical_geom_moment = geom_loading_rate * (recurrence_time / 2)

    print(f"  Recurrence time: {recurrence_time:.1f} years")
    print(f"  Mid-cycle moment: {typical_geom_moment:.2e} m³")

    # === Step 3: Choose beta for e² amplification at mid-cycle ===

    r_mid_target = lambda_required

    # Target: exp(beta × r_mid) = e² ≈ 7.4
    # So: beta = 2 / r_mid
    target_exponent = 2.0  # Gives e² ≈ 7.4× amplification
    beta = target_exponent / r_mid_target

    print(f"\n=== Rate Parameters ===")
    print(f"  Beta: {beta:.2f} (scaled for amplification)")
    print(f"  Target mid-cycle r: {r_mid_target:.6f}")
    print(f"  Target amplification: {np.e**target_exponent:.2f}×")

    # Solve for lambda_0
    lambda_0 = lambda_required / np.exp(target_exponent)

    print(f"  Lambda_0: {lambda_0:.6f} events/yr (background)")
    print(f"  = 1 per {1 / lambda_0:.0f} years at r=0")

    # C_a gives r_mid_target at mid-cycle
    C_a = r_mid_target / typical_geom_moment

    print(f"  C_a: {C_a:.2e} (m³·yr)⁻¹")
    print(f"  At mid-cycle: r_acc = {C_a * typical_geom_moment:.6f}")

    # === Step 4: Depletion ===

    tau_recovery = recurrence_time
    M_char_magnitude = config.M_max - 0.5
    M_char = magnitude_to_seismic_moment(M_char_magnitude)

    accumulated_during_recovery = geom_loading_rate * tau_recovery
    C_r = (C_a * accumulated_during_recovery) / (M_char**config.psi)

    print(f"\n=== Depletion Parameters ===")
    print(f"  Characteristic event: M {M_char_magnitude:.1f} ({M_char:.2e} N·m)")
    print(f"  Recovery time: {tau_recovery:.1f} years")
    print(f"  C_r: {C_r:.2e}")

    r_depl_after_char = -C_r * M_char**config.psi
    print(f"  After M {M_char_magnitude:.1f}: r_depl = {r_depl_after_char:.6f}")
    print(f"  Ratio to r_mid: {abs(r_depl_after_char / r_mid_target):.2f}×")

    # === Verification ===

    r_mid_actual = C_a * typical_geom_moment
    lambda_mid_actual = lambda_0 * np.exp(beta * r_mid_actual)

    print(f"\n=== Self-Consistency Check ===")
    print(f"  At t=0: r_total ≈ 0, λ = {lambda_0:.6f}/yr")
    print(
        f"  At mid-cycle: r_total = {r_mid_actual:.6f}, λ = {lambda_mid_actual:.6f}/yr"
    )
    print(f"  Target rate: {lambda_required:.6f}/yr")

    error = abs(lambda_mid_actual - lambda_required) / lambda_required
    print(f"  Relative error: {error:.2%}")
    print(f"  Match: {error < 0.01}")

    print(f"\n=== Moment Balance ===")
    print(f"  Input: {seismic_loading_rate:.2e} N·m/yr")
    print(
        f"  Output: {lambda_required:.6f}/yr × {M0_avg:.2e} N·m = {lambda_required * M0_avg:.2e} N·m/yr"
    )
    print(f"  Ratio: {(lambda_required * M0_avg) / seismic_loading_rate:.3f}")

    print(f"\n=== Sensitivity Examples (with beta={beta:.1f}) ===")
    for r_example in [0.001, 0.003, 0.005, 0.01, 0.05]:
        lambda_example = lambda_0 * np.exp(beta * r_example)
        print(
            f"  If r_total = {r_example:.3f}: λ = {lambda_example:.6f}/yr ({lambda_example / lambda_0:.2f}×)"
        )

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
