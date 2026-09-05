"""
Temporal probability/rate functions for earthquake generation

Adaptive rate formulation: Rate self-corrects to achieve moment balance
"""

import numpy as np
from moment import magnitude_to_seismic_moment
from magnitude_law import TiltedGR


def compute_expected_moment_per_event(config):
    """
    Compute expected geometric moment per event from G-R distribution

    Integrates moment * probability over the magnitude range to get
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
        Expected geometric moment per event (m^3)
    """
    # Sample magnitude range finely
    M_array = np.linspace(config.M_min, config.M_max, 1000)
    dM = M_array[1] - M_array[0]

    # Gutenberg-Richter probability density
    # P(M) proportional to 10^(-b*M)
    # Normalize over [M_min, M_max]
    b = config.b_value
    P_unnormalized = 10 ** (-b * M_array)
    P_normalized = P_unnormalized / (np.sum(P_unnormalized) * dM)

    # Convert magnitudes to geometric moments
    M0_array = magnitude_to_seismic_moment(M_array)  # N-m (seismic)
    geom_moment_array = M0_array / config.shear_modulus_Pa  # m^3 (geometric)

    # Expected value: E[M] = integral of M * P(M) dM
    expected_geom_moment = np.sum(geom_moment_array * P_normalized * dM)

    return expected_geom_moment


def _gr_quadrature(config, n=1000):
    """Magnitude grid and normalized G-R density on [M_min, M_max]."""
    M_array = np.linspace(config.M_min, config.M_max, n)
    dM = M_array[1] - M_array[0]
    P_unnormalized = 10 ** (-config.b_value * M_array)
    P_normalized = P_unnormalized / (np.sum(P_unnormalized) * dM)
    return M_array, dM, P_normalized


def compute_expected_clipped_moment(config, reservoir_geom_moment, quadrature=None):
    """
    Expected geometric moment per event when release is clipped to capacity

    The slip generator scales an event down when the requested moment exceeds
    the deficit available on the ruptured elements (slip_generator.py). In a
    mean-field sense an event of nominal magnitude M ruptures an area
    A_r(M) = 10^(M - 3.99) km^2 (capped at the fault area) and can release at
    most A_r * D / A_fault, where D is the reservoir (total geometric moment
    on the fault). This returns E[min(m(M), capacity(M, D))] over G-R.

    Parameters:
    -----------
    config : Config object (needs n_elements, element_area_m2)
    reservoir_geom_moment : float
        Total geometric moment on the fault, D (m^3)

    quadrature : (M_array, dM, P_normalized), optional
        Magnitude density to average over (default: the unconditional G-R law;
        pass the tilted law's pdf at the reference so the law is not ignored)

    Returns:
    --------
    expected_clipped_moment : float (m^3)
    """
    if quadrature is None:
        M_array, dM, P_normalized = _gr_quadrature(config)
    else:
        M_array, dM, P_normalized = quadrature
    geom_moment_array = magnitude_to_seismic_moment(M_array) / config.shear_modulus_Pa
    fault_area_m2 = config.n_elements * config.element_area_m2
    rupture_area_m2 = np.minimum(10 ** (M_array - 3.99) * 1e6, fault_area_m2)
    capacity = rupture_area_m2 * reservoir_geom_moment / fault_area_m2
    clipped = np.minimum(geom_moment_array, capacity)
    return np.sum(clipped * P_normalized * dM)


def saturation_weights(m_current, config):
    """
    Per-element recharge state g_i in [0, 1] for the saturating rate law

    g(x) with x = m_i / (v_i * T_s): "exp" -> 1 - exp(-x); "hill" -> x^n/(1+x^n).
    Requires config._m_sat = slip_rate * rate_saturation_years (set by
    calibrate_rate_to_reservoir).
    """
    x = np.maximum(m_current, 0.0) / config._m_sat
    if getattr(config, "rate_saturation_shape", "exp") == "hill":
        n = getattr(config, "rate_saturation_hill_n", 2.0)
        xn = x ** n
        return xn / (1.0 + xn)
    return -np.expm1(-x)


MEMORY_LOG_XI_MAX = 60.0
MEMORY_H_FLOOR = 1e-6


def memory_relax(xi, dt_years, config):
    """Relax the rate-state variable toward 1 in place: xi -> 1 + (xi - 1) e^{-dt/t_a}"""
    decay = np.exp(-dt_years / config.memory_relaxation_years)
    xi *= decay
    xi += 1.0 - decay


def memory_rupture_update(xi, idx, slip, slip_rate, config):
    """
    Apply a slip (stress-drop) step to the rate-state variable in place

    ln xi_i += f * s_i / (v_i * t_a) on the given element indices.
    """
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return
    f = config.memory_reload_fraction
    t_a = config.memory_relaxation_years
    logxi = np.log(xi[idx]) + f * slip[idx] / (np.maximum(slip_rate[idx], 1e-12) * t_a)
    xi[idx] = np.exp(np.minimum(logxi, MEMORY_LOG_XI_MAX))


def memory_weights(xi):
    """Per-element shadow weight h_i = 1/xi_i in (0, 1]"""
    return np.maximum(1.0 / xi, MEMORY_H_FLOOR)


def memory_reference_H(config):
    """
    Long-run mean of h under moment balance

    Each element spends a fraction f * S_i/(v_i T) of the time in shadow, and
    the seismic share of release is (1 - f_as), so H_bar = 1 - f (1 - f_as)
    (or 1 - f if afterslip increments are treated as stress steps too).
    """
    override = getattr(config, "memory_reference_H", 0.0)
    if override > 0:
        return override
    f = config.memory_reload_fraction
    f_as = getattr(config, "afterslip_release_fraction", 0.0) if config.afterslip_enabled else 0.0
    if getattr(config, "memory_afterslip_steps", False):
        return max(1.0 - f, 0.05)
    return max(1.0 - f * (1.0 - f_as), 0.05)


def memory_steady_state_init(config, m_current, slip_rate):
    """
    Initial rate-state population

    "steady": a synthetic steady-state shadow population drawn from the seeded
    RNG: a fraction f (1 - f_as) of elements are in shadow with a remaining
    integrated shadow tau ~ U[0, f * m_i(0) / v_i], i.e. ln xi = tau / t_a.
    "fresh": xi = 1 everywhere (no RNG consumed). f = 0 consumes no RNG.
    """
    n = config.n_elements
    xi = np.ones(n)
    f = config.memory_reload_fraction
    if getattr(config, "memory_init", "steady") != "steady" or f <= 0:
        return xi
    f_as = getattr(config, "afterslip_release_fraction", 0.0) if config.afterslip_enabled else 0.0
    u = np.random.random(n)
    in_shadow = u < f * (1.0 - f_as)
    tau = np.random.random(n) * f * np.maximum(m_current, 0.0) / np.maximum(slip_rate, 1e-12)
    logxi = np.where(in_shadow, tau / config.memory_relaxation_years, 0.0)
    return np.exp(np.minimum(logxi, MEMORY_LOG_XI_MAX))


def saturation_reference(config, tau_D):
    """
    Long-run mean of g for the saturating law, assuming element ages
    (time since last rupture) are Erlang-2 distributed with mean tau_D.
    """
    override = getattr(config, "rate_saturation_reference", 0.0)
    if override > 0:
        return override
    T_s = getattr(config, "rate_saturation_years", 30.0)
    if getattr(config, "rate_saturation_shape", "exp") == "hill":
        a = np.linspace(0.0, 20.0 * tau_D, 20001)
        pdf = (4.0 * a / tau_D**2) * np.exp(-2.0 * a / tau_D)
        x = a / T_s
        nn = getattr(config, "rate_saturation_hill_n", 2.0)
        g = x**nn / (1.0 + x**nn)
        return float(np.trapezoid(g * pdf, a))
    return 1.0 - (1.0 + tau_D / (2.0 * T_s)) ** (-2)


def omori_integral(t1, t2, c, p):
    """
    int_{t1}^{t2} (tau + c)^(-p) dtau  (expected aftershocks per unit K)
    """
    if p == 1.0:
        return np.log((t2 + c) / (t1 + c))
    return ((t2 + c) ** (1.0 - p) - (t1 + c) ** (1.0 - p)) / (1.0 - p)


def omori_productivity(magnitude, config):
    """K = K_ref * 10^(alpha * (M - M_ref))"""
    return config.omori_K_ref * 10 ** (
        config.omori_alpha * (magnitude - config.omori_M_ref)
    )


def omori_step_rate(K, lag_steps, dt, config):
    """
    Step-averaged aftershock rate (events/yr) at integer lag k >= 1

    Integrated mode: K * int_{(k-1) dt}^{k dt} (tau + c)^-p dtau / dt, i.e.
    the expected count over the step divided by dt (so that rate * dt is the
    expected count). Legacy mode: point sample K / (k dt + c)^p.
    """
    c, p = config.omori_c_years, config.omori_p
    if getattr(config, "omori_integrate_over_step", False):
        return K * omori_integral((lag_steps - 1) * dt, lag_steps * dt, c, p) / dt
    return K / (lag_steps * dt + c) ** p


def omori_lag_steps(config, dt):
    """Maximum integer lag (in steps) tracked for a sequence."""
    return int(round(config.omori_duration_years / dt))


def omori_branching_ratio(config, dt):
    """
    Expected aftershocks per event (G-R averaged K times the kernel sum)
    for the active Omori mode on the simulation grid.
    """
    M_array, dM, P_normalized = _gr_quadrature(config)
    K_mean = np.sum(omori_productivity(M_array, config) * P_normalized * dM)
    k_max = omori_lag_steps(config, dt)
    kernel_sum = sum(omori_step_rate(1.0, k, dt, config) * dt for k in range(1, k_max + 1))
    return K_mean * kernel_sum


def omori_parent_rates(event_history, current_time, dt, config):
    """
    Per-parent Omori step rates under the Bath split

    Returns (parent_idx, rates): indices into event_history of the parents
    that can still produce an aftershock above M_min (magnitude >= M_min +
    omori_bath_dM) and are inside the Omori window, with their step-averaged
    rate (events/yr; K scaled by omori_bath_K_scale). Also frees the spatial
    activation kernel of expired non-afterslip events, as
    afterslip.compute_aftershock_spatial_weights does on the aggregated path
    (which the split bypasses).
    """
    integrated = getattr(config, "omori_integrate_over_step", False)
    k_max = omori_lag_steps(config, dt)
    M_floor = config.M_min + config.omori_bath_dM
    K_scale = getattr(config, "omori_bath_K_scale", 1.0)
    c, p_om = config.omori_c_years, config.omori_p
    parents, rates = [], []
    for idx, event in enumerate(event_history):
        dt_event = current_time - event["time"]
        if integrated:
            lag = int(round(dt_event / dt))
            active = 1 <= lag <= k_max
            expired = lag > k_max
        else:
            lag = 0
            active = 0.0 < dt_event <= config.omori_duration_years
            expired = dt_event > config.omori_duration_years
        if (
            expired
            and event.get("spatial_activation") is not None
            and event.get("afterslip_sequence_id") is None
        ):
            event["spatial_activation"] = None  # free expired kernels
        if not active or event["magnitude"] < M_floor:
            continue
        K = omori_productivity(event["magnitude"], config) * K_scale
        if integrated:
            rate = omori_step_rate(K, lag, dt, config)
        else:
            rate = K / (dt_event + c) ** p_om
        parents.append(idx)
        rates.append(rate)
    return np.array(parents, dtype=int), np.array(rates, dtype=float)


def expected_moment_budget(config, law, dt, E_L):
    """
    Mean-field cascade moment budget under the Bath split

    A loading event of magnitude M triggers n(M) = K(M) K_scale * sum_k kernel_k
    children (only if M - dM >= M_min), each drawing G-R on [M_min, M - dM];
    grandchildren likewise. With n_L = E_GR[n(M)], E_O the productivity-weighted
    mean child moment and n_O the mean grandchildren per child, the moment
    released per loading event including its cascade is
        m_cascade = E_L + n_L E_O / (1 - n_O)
    so that lambda_load = (1 - f_as) L_tot / m_cascade balances loading.
    """
    dM_bath = config.omori_bath_dM
    K_scale = getattr(config, "omori_bath_K_scale", 1.0)
    k_max = omori_lag_steps(config, dt)
    kernel_sum = sum(omori_step_rate(1.0, k, dt, config) * dt for k in range(1, k_max + 1))
    M, p = law.M, law.pgr  # loading law (parents)
    # expected direct children of an event of magnitude x (any origin)
    def n_of(x):
        return np.where(x - dM_bath >= law.M_min_child,
                        omori_productivity(x, config) * K_scale * kernel_sum, 0.0)
    n_M = n_of(M)
    eligible, mean_child, mean_offspring = law.child_curves(dM_bath, n_child=n_of(law.M_child))
    n_L = float(law._integral(p * n_M))
    E_O = float(law._integral(p * n_M * mean_child) / n_L) if n_L > 0 else 0.0
    n_O = float(law._integral(p * n_M * mean_offspring) / n_L) if n_L > 0 else 0.0
    n_O = min(n_O, 0.999)
    m_cascade = E_L + n_L * E_O / (1.0 - n_O)
    return {"E_L": E_L, "n_L": n_L, "E_O": E_O, "n_O": n_O,
            "m_cascade": m_cascade, "kernel_sum": kernel_sum}


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
        Base rate coefficient (events/year per m^3 of accumulated moment)
    """

    # Estimate equilibrium accumulated moment
    geom_loading_rate = (
        config.background_slip_rate_m_yr * config.n_elements * config.element_area_m2
    )

    # Method 1: Analytical from G-R distribution (NEW - better!)
    expected_geom_moment_per_event = compute_expected_moment_per_event(config)

    # Estimate equilibrium: balance requires loading_rate = lambda * <M_event>
    # At equilibrium with deficit D: lambda = C * D
    # So: loading_rate = C * D * <M_event>
    # Assuming D ~= half-cycle worth of moment
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
    # At equilibrium: geom_loading_rate = lambda_target * expected_moment_per_event
    # And lambda_target = C * geom_moment_equilibrium
    # So: C = geom_loading_rate / (geom_moment_equilibrium * expected_moment_per_event)
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

    # Reference deficit for the rate law lambda = C * corr * D_ref * (D/D_ref)^nu
    # (legacy mode: the assumed equilibrium; moment_balance mode overrides it
    # in calibrate_rate_to_reservoir once the actual reservoir exists)
    config.deficit_reference = geom_moment_equilibrium
    config.lambda_0 = C * geom_moment_equilibrium

    # Initialize adaptive correction factor
    config.rate_correction_factor = 1.0
    config.coupling_history = []  # Store periodically (every 100 years) for diagnostics

    print("\n" + "=" * 70)
    print("MOMENT-BASED RATE MODEL")
    print("=" * 70)
    print(f"  Geometric loading rate: {geom_loading_rate:.2e} m^3/yr")
    print(
        f"  Seismic loading rate: {config.shear_modulus_Pa * geom_loading_rate:.2e} N-m/yr"
    )
    print(f"  Characteristic M_max event: M {config.M_max:.1f} ({M0_char:.2e} N-m)")
    print(f"  Estimated recurrence time: {recurrence_time_char:.1f} years")
    print(f"\n  Equilibrium accumulated moment: {geom_moment_equilibrium:.2e} m^3")
    print(f"  Expected moment per event (G-R analytical): {expected_geom_moment_per_event:.2e} m^3")
    print(f"  Target rate at equilibrium: {lambda_target:.3f} events/year")
    print(f"\n  Base rate coefficient C (analytical): {C:.3e} (events/yr)/(m^3)")
    print(f"  Base rate coefficient C (old method): {C_old:.3e} (events/yr)/(m^3)")
    print(f"  Improvement ratio: {C_old/C:.2f}x")
    print(f"\n  lambda(t) = lambda_background + C * correction_factor(t) * moment_deficit(t) + lambda_aftershock(t) + lambda_perturbation(t)")

    # Print adaptive correction status
    mode = getattr(config, "adaptive_correction_mode", "legacy")
    if mode == "legacy":
        if hasattr(config, "adaptive_correction_enabled") and config.adaptive_correction_enabled:
            print(f"  ADAPTIVE CORRECTION: ENABLED (legacy, continuous updates every timestep)")
            print(f"    Gain: {config.adaptive_correction_gain}")
            print(f"    Will drive coupling -> 1.0")
        else:
            print(f"  ADAPTIVE CORRECTION: DISABLED (fixed C, natural coupling)")
            print(f"    Coupling will depend on G-R distribution and slip heterogeneity")
    elif mode == "integral":
        print(f"  ADAPTIVE CORRECTION: INTEGRAL trim (gain {config.adaptive_correction_gain} /yr, "
              f"window {config.adaptive_correction_window_years} yr, "
              f"bounds [{config.correction_factor_min}, {config.correction_factor_max}])")
    else:
        print(f"  ADAPTIVE CORRECTION: OFF (correction factor fixed at 1.0)")
    print(f"  Rate mode: {getattr(config, 'rate_mode', 'legacy')}, "
          f"deficit source: {getattr(config, 'deficit_source', 'tracked')}, "
          f"deficit exponent nu = {getattr(config, 'deficit_exponent', 1.0)}")
    print(f"  Event sampling: {getattr(config, 'event_sampling', 'deterministic')}")

    # Print Omori aftershock parameters if enabled
    if hasattr(config, "omori_enabled") and config.omori_enabled:
        print(f"\n  OMORI AFTERSHOCKS ENABLED:")
        print(f"    Law: lambda_aftershock = K / (t + c)^p")
        print(f"    p = {config.omori_p:.2f}")
        print(f"    c = {config.omori_c_years:.6f} years")
        print(f"    K_ref = {config.omori_K_ref:.3f} events/yr (at M={config.omori_M_ref:.1f})")
        print(f"    alpha = {config.omori_alpha:.2f} (magnitude scaling)")
        print(f"    Duration: {config.omori_duration_years:.1f} years per sequence")
        K_M7 = config.omori_K_ref * 10 ** (config.omori_alpha * (7.0 - config.omori_M_ref))
        print(f"    Example: M7.0 -> K = {K_M7:.3f} events/yr")
        integrated = getattr(config, "omori_integrate_over_step", False)
        print(f"    Time integration: {'integrated over each step' if integrated else 'point-sampled at grid lags (legacy)'}")
        n_branch = omori_branching_ratio(config, config.time_step_years)
        print(f"    Branching ratio (expected aftershocks per event on the grid): {n_branch:.3f}")
    else:
        print(f"\n  OMORI AFTERSHOCKS DISABLED")

    # Print background rate if enabled
    if hasattr(config, "lambda_background") and config.lambda_background > 0:
        print(f"\n  BACKGROUND RATE ENABLED:")
        print(f"    lambda_background = {config.lambda_background:.4f} events/yr")

    # Print perturbation parameters if enabled
    if hasattr(config, "perturbation_type") and config.perturbation_type != "none":
        print(f"\n  RANDOM PERTURBATIONS ENABLED:")
        print(f"    Type: {config.perturbation_type}")
        if config.perturbation_type == "white_noise":
            print(f"    sigma = {config.perturbation_sigma:.4f} events/yr")
        elif config.perturbation_type == "ornstein_uhlenbeck":
            print(f"    Mean: {config.perturbation_mean:.4f} events/yr")
            print(f"    sigma (diffusion): {config.perturbation_sigma:.4f}")
            print(f"    theta (reversion): {config.perturbation_theta:.2f} /yr")

    print("=" * 70)

    return C


def calibrate_rate_to_reservoir(config, m_current, slip_rate):
    """
    Set the rate coefficient from moment balance at the actual reservoir

    rate_mode == "moment_balance":
        lambda_0 = (1 - f_as) * L_tot / E[m](D_ref)
        C = lambda_0 / D_ref
    where L_tot is the full geometric loading rate (background + pulses),
    D_ref is the reference reservoir (the spin-up reservoir D_res(0) unless
    deficit_reference_years > 0), E[m] is the expected moment per event
    (nominal G-R, or clipped to capacity at D_ref if
    rate_expected_moment == "clipped"), and f_as is the fraction of loading
    expected to be released aseismically by afterslip.

    Must be called after initialize_moment (needs m_current and slip_rate).
    In legacy mode this only records L_tot and D_res(0) for diagnostics.

    Returns:
    --------
    C : float  (events/yr per m^3)
    """
    area = config.element_area_m2
    L_tot = float(np.sum(slip_rate) * area)
    D_0 = float(np.sum(m_current) * area)
    config.geom_loading_rate_total = L_tot
    config.initial_reservoir = D_0
    # Per-element saturation deficit for the saturating rate law (private:
    # arrays must not be serialized into the HDF5 config group)
    T_s = getattr(config, "rate_saturation_years", 30.0)
    config._m_sat = np.maximum(slip_rate, 1e-12) * T_s

    ref_years = getattr(config, "deficit_reference_years", 0.0)
    D_ref = ref_years * L_tot if ref_years > 0 else D_0

    # Deficit-weighted magnitude law and/or Bath split (None = legacy paths)
    tilt_on = getattr(config, "magnitude_tilt_mu", 0.0) != 0.0
    split_on = bool(getattr(config, "omori_split_enabled", False)) and config.omori_enabled
    load_law_on = (getattr(config, "magnitude_load_M_min", 0.0) > config.M_min
                   or getattr(config, "magnitude_load_b", 0.0) > 0)
    law = TiltedGR(config, D_ref, slip_rate) if (tilt_on or split_on or load_law_on) else None
    config._magnitude_law = law
    config._omori_split = split_on

    if getattr(config, "rate_mode", "legacy") != "moment_balance":
        return config.C_rate_base

    if law is not None:
        M_grid, p_ref = law.pdf(0.0)
        if getattr(config, "rate_expected_moment", "gr") == "clipped":
            expected_moment = compute_expected_clipped_moment(
                config, D_ref, quadrature=(M_grid, law.dM, p_ref)
            )
        else:
            expected_moment = law.expected_moment(0.0)
    elif getattr(config, "rate_expected_moment", "gr") == "clipped":
        expected_moment = compute_expected_clipped_moment(config, D_ref)
    else:
        expected_moment = compute_expected_moment_per_event(config)

    f_as = getattr(config, "afterslip_release_fraction", 0.0)
    if not config.afterslip_enabled:
        f_as = 0.0
    f_as = min(max(f_as, 0.0), 0.95)

    budget = None
    if split_on:
        # Cascade budget: aftershocks are Bath-capped, so they release far less
        # than E[m]; the (1 - n_b) factor below would leave the reservoir 60% high
        budget = expected_moment_budget(config, law, config.time_step_years, expected_moment)
        lambda_bar = (1.0 - f_as) * L_tot / budget["m_cascade"]
        n_branch = budget["n_L"]
    else:
        lambda_bar = (1.0 - f_as) * L_tot / expected_moment
        n_branch = 0.0
        if getattr(config, "rate_omori_branching_correction", False) and config.omori_enabled:
            n_branch = omori_branching_ratio(config, config.time_step_years)
            lambda_bar *= (1.0 - n_branch)
    config.omori_branching_used = n_branch
    if budget is not None:
        config.omori_bath_child_moment = budget["E_O"]
        config.omori_bath_grandchild_branching = budget["n_O"]
        config.expected_cascade_moment_per_event = budget["m_cascade"]
    rate_law = getattr(config, "rate_law", "power")
    if rate_law == "memory":
        state_ref = memory_reference_H(config)
    elif rate_law == "saturating":
        state_ref = saturation_reference(config, D_ref / L_tot)
    else:
        state_ref = 1.0
    lambda_0 = lambda_bar / state_ref  # rate in the fully recovered state
    C = lambda_0 / D_ref

    config.C_rate_base = C
    config.lambda_bar = lambda_bar
    config.lambda_0 = lambda_0
    config.rate_state_reference = state_ref
    config.deficit_reference = D_ref
    config.expected_geom_moment_per_event = expected_moment
    config.rate_correction_factor = 1.0

    print("\n" + "=" * 70)
    print("RATE CALIBRATED TO RESERVOIR (moment_balance mode)")
    print("=" * 70)
    print(f"  Total geometric loading rate L_tot: {L_tot:.3e} m^3/yr")
    print(f"  Initial reservoir D_res(0): {D_0:.3e} m^3 ({D_0 / L_tot:.1f} yr of loading)")
    print(f"  Reference deficit D_ref: {D_ref:.3e} m^3 ({D_ref / L_tot:.1f} yr of loading)")
    print(f"  Expected moment per event ({getattr(config, 'rate_expected_moment', 'gr')}): {expected_moment:.3e} m^3")
    print(f"  Afterslip release fraction f_as: {f_as:.2f}")
    if budget is not None:
        print(f"  Bath split: dM = {config.omori_bath_dM}, K_scale = {getattr(config, 'omori_bath_K_scale', 1.0)}; "
              f"eligible branching n_L = {budget['n_L']:.4f}, child moment E_O = {budget['E_O']:.3e} m^3 "
              f"({budget['E_O'] / expected_moment:.3f} E[m]), grandchild n_O = {budget['n_O']:.2e}")
        print(f"  lambda_bar = (1 - f_as) L_tot / (E[m] + n_L E_O / (1 - n_O)) = {lambda_bar:.4f} events/yr "
              f"(loading term; total with cascade ~{lambda_bar * (1 + budget['n_L'] / (1 - budget['n_O'])):.3f}/yr)")
    else:
        print(f"  lambda_bar = (1 - n_b) (1 - f_as) L_tot / E[m] = {lambda_bar:.4f} events/yr "
              f"(loading term; branching n_b = {n_branch:.3f} -> total ~{lambda_bar / max(1 - n_branch, 1e-6):.3f}/yr)")
    if law is not None:
        print(law.describe())
        kappa = getattr(config, "rate_size_coupling", 0.0)
        print(f"  Rate-size coupling kappa = {kappa} (ratio clamp {getattr(config, 'rate_size_coupling_max', 3.0)})")
    print(f"  Controller target: {getattr(config, 'adaptive_correction_target', 'coupling')}, "
          f"deficit exponent nu = {getattr(config, 'deficit_exponent', 1.0)}")
    print(f"  Rate law: {rate_law}; reference state = {state_ref:.3f}; "
          f"lambda_0 = lambda_bar / ref = {lambda_0:.4f} events/yr (recovered state)")
    if rate_law == "memory":
        f, t_a = config.memory_reload_fraction, config.memory_relaxation_years
        print(f"  Memory shadow: f = {f}, t_a = {t_a} yr; a 1.2 m slip at 15 mm/yr shadows its patch "
              f"for ~{f * 1.2 / 0.015:.0f} yr (depth exp(-{f * 1.2 / (0.015 * t_a):.1f}))")
    elif rate_law == "saturating":
        print(f"  Saturating shadow: T_s = {config.rate_saturation_years} yr ({config.rate_saturation_shape})")
    else:
        print(f"  C = lambda_0 / D_ref = {C:.3e} (events/yr)/m^3, nu = {getattr(config, 'deficit_exponent', 1.0)}")
    print("=" * 70)
    return C


def update_rate_correction(
    config, cumulative_loading, cumulative_release, current_time, dt_years
):
    """
    Update adaptive rate correction factor based on observed coupling

    adaptive_correction_mode:
      "legacy":   proportional update on the run-cumulative coupling, gated by
                  adaptive_correction_enabled (the original scheme).
      "integral": slow integral trim corr += gain * (1 - kappa_w) * dt with
                  anti-windup, where kappa_w = R_w / L_w is the coupling over
                  an exponentially weighted window of
                  adaptive_correction_window_years (0 = run-cumulative).
      "off":      correction factor stays at 1.0.

    Parameters:
    -----------
    config : Config object
    cumulative_loading : float
        Total geometric moment loaded (m^3)
    cumulative_release : float
        Total geometric moment released (m^3)
    current_time : float
        Current simulation time (years)
    dt_years : float
        Timestep size (years)

    Returns:
    --------
    None (updates config.rate_correction_factor in place)
    """
    mode = getattr(config, "adaptive_correction_mode", "legacy")

    if mode == "off":
        config.rate_correction_factor = 1.0
        return

    if mode == "integral":
        if getattr(config, "adaptive_correction_target", "coupling") == "reservoir":
            _update_rate_correction_reservoir(
                config, cumulative_loading, cumulative_release, dt_years
            )
        else:
            _update_rate_correction_integral(
                config, cumulative_loading, cumulative_release, dt_years
            )
        return

    # Legacy proportional scheme
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


def _update_rate_correction_integral(
    config, cumulative_loading, cumulative_release, dt_years
):
    """
    Slow integral trim on a windowed coupling with anti-windup

    State is kept in private attributes on config (excluded from to_dict):
      _ctrl_prev_L, _ctrl_prev_R : cumulative values at the previous call
      _ctrl_L_w, _ctrl_R_w       : EWMA sums of loading and release
    The window sums are initialized at balance (L_w = R_w = L_tot * tau_w)
    so there is no start-up kick.
    """
    tau_w = getattr(config, "adaptive_correction_window_years", 0.0)
    gain = config.adaptive_correction_gain

    if not hasattr(config, "_ctrl_prev_L"):
        config._ctrl_prev_L = 0.0
        config._ctrl_prev_R = 0.0
        L_tot = getattr(config, "geom_loading_rate_total", config.geom_loading_rate)
        config._ctrl_L_w = L_tot * tau_w if tau_w > 0 else 0.0
        config._ctrl_R_w = L_tot * tau_w if tau_w > 0 else 0.0

    dL = cumulative_loading - config._ctrl_prev_L
    dR = cumulative_release - config._ctrl_prev_R
    config._ctrl_prev_L = cumulative_loading
    config._ctrl_prev_R = cumulative_release

    if tau_w > 0:
        decay = min(dt_years / tau_w, 1.0)
        config._ctrl_L_w = config._ctrl_L_w * (1.0 - decay) + dL
        config._ctrl_R_w = config._ctrl_R_w * (1.0 - decay) + dR
    else:
        config._ctrl_L_w += dL
        config._ctrl_R_w += dR

    if config._ctrl_L_w <= 0:
        return

    kappa = config._ctrl_R_w / config._ctrl_L_w
    config.windowed_coupling = kappa
    adjustment = gain * (1.0 - kappa) * dt_years

    corr = config.rate_correction_factor
    lo, hi = config.correction_factor_min, config.correction_factor_max
    # Anti-windup: do not integrate further into a bound
    if (corr >= hi and adjustment > 0) or (corr <= lo and adjustment < 0):
        return
    config.rate_correction_factor = max(lo, min(hi, corr + adjustment))


def _update_rate_correction_reservoir(
    config, cumulative_loading, cumulative_release, dt_years
):
    """
    Integral trim on the reservoir error with anti-windup

    corr += gain * (D / D_ref - 1) * dt, with D = D_0 + L_cum - R_cum (exact
    reservoir identity). Pair with deficit_exponent > 0 (proportional term)
    for a damped loop: zeta = nu / (2 sqrt(gain * T_ref)).
    """
    D = config.initial_reservoir + cumulative_loading - cumulative_release
    D_ref = config.deficit_reference
    if D_ref <= 0:
        return
    err = D / D_ref - 1.0
    config.reservoir_error = err
    adjustment = config.adaptive_correction_gain * err * dt_years
    corr = config.rate_correction_factor
    lo, hi = config.correction_factor_min, config.correction_factor_max
    if (corr >= hi and adjustment > 0) or (corr <= lo and adjustment < 0):
        return
    config.rate_correction_factor = max(lo, min(hi, corr + adjustment))


def earthquake_rate(
    m_current,
    event_history,
    current_time,
    config,
    cumulative_loading,
    cumulative_release,
    dt_years=None,
    memory_h=None,
):
    """
    Compute instantaneous earthquake rate based on moment deficit

    lambda(t) = C_base * correction_factor(t) * D_ref * (D / D_ref)^nu
                + lambda_background + lambda_aftershock + lambda_perturbation

    D is the tracked deficit max(0, L_cum - R_cum) (deficit_source="tracked")
    or the actual reservoir element_area * sum(m_current) ("reservoir");
    nu = deficit_exponent (1 reproduces lambda = C * corr * D exactly).

    Parameters:
    -----------
    m_current : array
        Current slip deficit (m) at each element
    event_history : list
        List of past events
    current_time : float
        Current simulation time (years)
    config : Config object
    cumulative_loading : float
        Total geometric moment loaded since t=0 (m^3)
    cumulative_release : float
        Total geometric moment released by events (m^3)
    dt_years : float, optional
        Timestep (years); defaults to config.time_step_years. Used for the
        integer-lag Omori integration.

    Returns:
    --------
    lambda_t : float
        Instantaneous earthquake rate (events/year)
    components : dict
        Breakdown of rate components
    """
    if dt_years is None:
        dt_years = config.time_step_years

    # Moment deficit (should always be >= 0)
    if getattr(config, "deficit_source", "tracked") == "reservoir":
        moment_deficit = float(np.sum(m_current)) * config.element_area_m2
    else:
        moment_deficit = cumulative_loading - cumulative_release
    moment_deficit = max(0.0, moment_deficit)

    # Base rate from the moment deficit, with adaptive correction
    nu = getattr(config, "deficit_exponent", 1.0)
    saturation = np.nan
    rate_law = getattr(config, "rate_law", "power")
    if rate_law == "saturating":
        # Stress-shadow law: lambda_0 times the mean per-element recharge state
        g = saturation_weights(m_current, config)
        saturation = float(np.mean(g))
        lambda_loading = config.lambda_0 * config.rate_correction_factor * saturation
    elif rate_law == "memory":
        # Dieterich rate-state shadow: lambda_0 times the mean of h_i = 1/xi_i
        saturation = float(np.mean(memory_h))
        lambda_loading = config.lambda_0 * config.rate_correction_factor * saturation
    elif nu == 1.0:
        lambda_loading = config.C_rate_base * config.rate_correction_factor * moment_deficit
    else:
        # lambda_0 = C * D_ref is the rate at the reference deficit
        D_ref = config.deficit_reference
        lambda_loading = (
            config.lambda_0
            * config.rate_correction_factor
            * (moment_deficit / D_ref) ** nu
        )

    # Proportional reservoir feedback for the shadow laws
    if rate_law in ("memory", "saturating") and nu != 0.0 and config.deficit_reference > 0:
        lambda_loading *= (moment_deficit / config.deficit_reference) ** nu

    # Deficit-weighted magnitude law: state, expected event size and the
    # rate-size coupling lambda_load *= (E_GR / E[m | D])^kappa
    law = getattr(config, "_magnitude_law", None)
    tilt_theta = np.nan
    expected_moment = np.nan
    if law is not None and law.mu != 0.0:
        ell = law.log_deficit(m_current, memory_h if rate_law == "memory" else None)
        tilt_theta = float(law.theta(ell))
        expected_moment = law.expected_moment(ell)
        kappa = getattr(config, "rate_size_coupling", 0.0)
        if kappa > 0.0:
            R = getattr(config, "rate_size_coupling_max", 3.0)
            ratio = min(max(law.E_gr / expected_moment, 1.0 / R), R)
            lambda_loading *= ratio ** kappa

    # Steady background rate (external forcing independent of moment deficit)
    lambda_background = getattr(config, 'lambda_background', 0.0)

    # Aftershock rate (Omori-Utsu decay)
    lambda_aftershock = 0.0
    n_active_sequences = 0
    omori_parents = None
    omori_rates = None

    if (
        len(event_history) > 0
        and hasattr(config, "omori_enabled")
        and config.omori_enabled
    ):
        if getattr(config, "_omori_split", False):
            # Bath split: only parents that can still produce M >= M_min
            omori_parents, omori_rates = omori_parent_rates(
                event_history, current_time, dt_years, config
            )
            lambda_aftershock = float(np.sum(omori_rates))
            n_active_sequences = int(omori_parents.size)
        elif getattr(config, "omori_integrate_over_step", False):
            # Expected count over the step (t - dt, t], at integer grid lags.
            # Events created during the current step are not yet in
            # event_history, so lag 0 never occurs; the previous step's
            # event delivers its whole first year at lag 1.
            k_max = omori_lag_steps(config, dt_years)
            for event in event_history:
                lag = int(round((current_time - event["time"]) / dt_years))
                if 1 <= lag <= k_max:
                    n_active_sequences += 1
                    K = omori_productivity(event["magnitude"], config)
                    lambda_aftershock += omori_step_rate(K, lag, dt_years, config)
        else:
            # Legacy point sampling at the raw grid lag
            omori_c_years = config.omori_c_years
            for event in event_history:
                dt_event = current_time - event["time"]

                # Only consider events within aftershock duration window
                if 0 < dt_event <= config.omori_duration_years:
                    n_active_sequences += 1

                    # Omori-Utsu law: lambda(t) = K / (t + c)^p
                    # K scales with mainshock magnitude: K = K_ref * 10^(alpha * (M - M_ref))
                    M_mainshock = event["magnitude"]
                    K = config.omori_K_ref * 10 ** (
                        config.omori_alpha * (M_mainshock - config.omori_M_ref)
                    )

                    # Add this mainshock's aftershock contribution
                    lambda_aftershock += K / (dt_event + omori_c_years) ** config.omori_p

    # Random perturbations (stochastic external forcing)
    lambda_perturbation = 0.0
    perturbation_type = getattr(config, 'perturbation_type', 'none')
    if perturbation_type == "white_noise":
        # Uncorrelated additive noise (each timestep independent)
        lambda_perturbation = np.abs(np.random.normal(0, config.perturbation_sigma))
    elif perturbation_type == "ornstein_uhlenbeck":
        # Time-correlated process (state updated in simulator.py)
        lambda_perturbation = max(0.0, config.perturbation_mean + config.perturbation_state)

    # Total rate (all components)
    lambda_t = lambda_background + lambda_loading + lambda_aftershock + lambda_perturbation
    lambda_t = max(0.0, lambda_t)

    # Components for diagnostics
    components = {
        "background": lambda_background,
        "loading": lambda_loading,
        "aftershock": lambda_aftershock,
        "perturbation": lambda_perturbation,
        "n_active_sequences": n_active_sequences,
        "moment_deficit": moment_deficit,
        "correction_factor": config.rate_correction_factor,
        "saturation": saturation,
        "tilt_theta": tilt_theta,
        "expected_moment": expected_moment,
        "lambda_omori_eligible": lambda_aftershock if omori_rates is not None else np.nan,
    }
    if omori_rates is not None:
        # Runtime side channel for the origin draw (arrays: not written to HDF5)
        components["omori_parents"] = omori_parents
        components["omori_rates"] = omori_rates

    return lambda_t, components
