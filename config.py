"""
Configuration file for strike-slip fault simulator
All parameters in one place for easy tuning
"""

import numpy as np


class Config:
    """Simulation configuration"""

    # Fault geometry
    fault_length_km = 200.0  # Along-strike length (km)
    fault_depth_km = 25.0  # Down-dip depth (km)
    element_size_km = 1.0  # Grid cell size (km)
    element_size_km = 0.5  # Grid cell size (km)

    # Background moment (slip deficit) rate
    background_slip_rate_mm_yr = 10.0  # mm/year

    # Gaussian moment pulse(s) - list of dicts
    moment_pulses = [
        {
            "center_x_km": 100.0,  # Along-strike position
            "center_z_km": 0.0,  # Down-dip position
            "sigma_km": 25.0,  # Width of Gaussian
            "amplitude_mm_yr": 20.0,  # Additional slip rate at center
        }
    ]

    # The lone physical parameter
    shear_modulus_Pa = 3e10  # Pa (30 GPa)

    # Magnitude dependent spatial probability
    gamma_min = 0.0  # Small events anywhere
    gamma_max = 1.5  # Large events need a pool of moment
    alpha_spatial = 0.35  # Decay rate
    M_min = 5.0  # Minimum magnitude
    M_max = 7.5  # Maximum magnitude (8.0 needs 2x the fault area; M>=7.7 draws empty the whole fault)

    # Adaptive rate correction
    adaptive_correction_enabled = (
        True  # Enable adaptive correction (True = drives coupling toward 1.0)
    )
    adaptive_correction_gain = (
        2e-4  # Controller gain (/yr). Legacy proportional scheme used 5.0. For the
        # reservoir-keyed integral with nu = 0.5 the critically damped value is
        # nu^2 / (4 T_ref) ~ 2e-4 at T_ref = 250 yr; coupling-keyed trim used 1e-3
    )
    correction_factor_min = 0.5  # Minimum allowed correction factor (legacy 0.1)
    correction_factor_max = 2.0  # Maximum allowed correction factor (legacy 10.0)
    # Controller mode
    #   "legacy":   corr += gain * (1 - R_cum/L_cum) * dt, gated by
    #               adaptive_correction_enabled (bang-bang at gain ~ 5).
    #   "integral": slow integral trim with anti-windup on a windowed
    #               (EWMA) coupling; recommended gain ~ 1e-3 /yr, window
    #               ~ 300 yr, bounds [0.5, 2.0].
    #   "off":      correction factor fixed at 1.0.
    adaptive_correction_mode = "integral"
    adaptive_correction_window_years = 300.0  # EWMA window for "integral" (0 = cumulative)
    # adaptive_correction_target (integral mode):
    #   "coupling":  corr += gain * (1 - kappa_w) * dt on the windowed coupling
    #                (a washout filter: restores toward its own moving average)
    #   "reservoir": corr += gain * (D / D_ref - 1) * dt  (set point D_ref;
    #                combine with deficit_exponent > 0 as the proportional term)
    adaptive_correction_target = "reservoir"

    # Rate law
    #   lambda_loading = C * corr * D_ref * (D / D_ref)^deficit_exponent
    # rate_mode:
    #   "legacy":         C from compute_rate_parameters (assumes the tracked
    #                     deficit equilibrates at L*T_char/2), D_ref = that
    #                     equilibrium value.
    #   "moment_balance": lambda_0 = (1 - f_as) * L_tot / E[m_clipped](D_ref)
    #                     calibrated from the actual reservoir after
    #                     initialize_moment; C = lambda_0 / D_ref.
    rate_mode = "moment_balance"
    # deficit_source:
    #   "tracked":   D = max(0, L_cum - R_cum)  (legacy; excludes the spin-up
    #                reservoir, can hit zero after large events)
    #   "reservoir": D = element_area * sum(m_current) (the actual deficit)
    deficit_source = "reservoir"
    deficit_exponent = 0.5  # nu: 1 = legacy (rate ~ deficit), 0 = stationary rate (Dieterich steady state).
    # For rate_law memory/saturating, nu > 0 multiplies lambda_load by (D/D_ref)^nu
    # (the proportional term of the reservoir controller).
    # Fold the Omori branching ratio into the calibration: the loading term is
    # (1 - n_b) of the total rate that balances moment (aftershocks release the
    # same E[m]). Legacy False left the reservoir to sink until clipping paid
    # for the extra ~30% of events.
    rate_omori_branching_correction = True
    # rate_law:
    #   "power":      lambda_load = lambda_0 * corr * (D/D_ref)^nu (global reservoir)
    #   "saturating": lambda_load = lambda_0 * corr * mean_i g(m_i / (v_i T_s)),
    #                 a per-element stress-shadow: g(0) = 0, g -> 1 once an element
    #                 has reloaded T_s years of its own slip rate. Suppresses the
    #                 rate after ruptures (in proportion to the area emptied) and is
    #                 flat once recharged, so there is no pre-mainshock crescendo.
    #   "memory":     lambda_load = lambda_0 * corr * mean_i h_i with a Dieterich
    #                 rate-state shadow per element: h_i = 1/xi_i, xi_i >= 1;
    #                 a rupture putting slip s_i on element i sets
    #                 ln xi_i += f * s_i / (v_i * t_a) and xi relaxes as
    #                 xi -> 1 + (xi - 1) exp(-dt/t_a). Exact identity:
    #                 int (1 - h_i) dt = f * s_i / v_i, i.e. the shadow lasts a
    #                 fraction f of the time needed to reload the slip released,
    #                 whatever the remaining deficit. Bounded above (h <= 1), so
    #                 no pre-mainshock crescendo.
    rate_law = "memory"
    rate_saturation_years = 30.0  # T_s: recharge time (years of local loading)
    rate_saturation_shape = "exp"  # "exp": 1 - exp(-x); "hill": x^n / (1 + x^n)
    rate_saturation_hill_n = 2.0
    rate_saturation_reference = 0.0  # S_ref for lambda_0 = lambda_bar / S_ref (0 = analytic)
    memory_reload_fraction = 0.5  # f: shadow duration as a fraction of the reload time s_i / v_i
    memory_relaxation_years = 10.0  # t_a: Dieterich aftereffect time (width of the recovery)
    memory_afterslip_steps = False  # treat afterslip increments as stress steps too
    memory_init = "steady"  # "steady": synthetic steady-state shadow population; "fresh": xi = 1
    memory_reference_H = 0.0  # H_bar for lambda_0 = lambda_bar / H_bar (0 = analytic 1 - f (1 - f_as))
    # Also multiply the MaxEnt nucleation weights by g_i / h_i (so events do
    # not nucleate on patches still in the shadow). Off = pure m_i^gamma(M).
    rate_spatial_weighting = True
    deficit_reference_years = 250.0  # D_ref = this * L_tot if > 0, else D_res(0)
    rate_expected_moment = "clipped"  # "gr": E[m] of nominal G-R; "clipped": E[min(m, capacity)]
    afterslip_release_fraction = 0.2  # f_as: fraction of release that is aseismic (moment_balance mode; realized ~0.20 with the capped budget)
    # Clip afterslip to the available local deficit so m_current never goes
    # negative (legacy False: overlapping sequences can overshoot).
    afterslip_clip_to_deficit = True
    # Assert D_res == D_0 + L_cum - R_cum every step (debug only)
    check_reservoir_identity = False

    # Omori aftershock parameters
    # Standard Omori-Utsu law: rho_aftershock(t) = K / (t + c)^p
    # where K scales with mainshock magnitude: K = K_ref * 10^(alpha * (M - M_ref))
    omori_enabled = True  # Enable/disable aftershock sequences
    omori_p = 1.0  # Decay exponent (typically ~1.0)
    omori_c_years = 1.0 / 365.25  # Time offset in years (~0.00274 years = 1 day)
    omori_K_ref = 0.043  # Productivity (events/year) for M=6 mainshock (0.1 with legacy point sampling)
    omori_M_ref = 6.0  # Reference magnitude for productivity
    omori_alpha = 0.8  # Magnitude scaling (Reasenberg & Jones 1989)
    omori_duration_years = (
        30.0  # Only track aftershocks for this many years after mainshock
    )
    # Omori time integration over a timestep.
    #   False: legacy point sampling K / (k*dt + c)^p at integer grid lags k
    #          (with c = 1 day and dt = 1 yr this drops the day-to-year part
    #          of every sequence and spreads the remainder over 30 years).
    #   True:  expected count over the step, K * [F(k*dt) - F((k-1)*dt)] with
    #          F(t) = int_0^t (tau + c)^-p dtau, divided by dt. The previous
    #          step's event delivers its whole first year at lag k = 1 (causal).
    #          To preserve the branching ratio of the legacy scheme (~0.235 for
    #          30-yr sequences) use omori_K_ref ~ 0.043 with this mode.
    omori_integrate_over_step = True
    # Compute the spatial activation kernel Phi (for aftershock localization)
    # for all events with M >= this value, independent of afterslip. inf =
    # legacy behaviour (Phi only exists for afterslip-triggering events).
    omori_spatial_M_min = 5.0

    # Event-count sampling per timestep
    #   "deterministic": fractional accumulator, emit floor(debt) (legacy).
    #                    N_obs == int(lambda dt) to within one event; no
    #                    counting noise, no gaps longer than ceil(1/lambda).
    #   "poisson":       n ~ Poisson(lambda * dt) from the seeded RNG.
    event_sampling = "poisson"
    # Store sorted U[0, dt) offsets as event["time_offset"] for catalog
    # statistics. event["time"] stays on the grid and drives all dynamics.
    event_substep_times = True

    # Background rate of seismicity
    lambda_background = 0.0

    # Random perturbations
    perturbation_type = (
        "none"  # Maybe options: "none", "white_noise", "ornstein_uhlenbeck"
    )
    perturbation_sigma = (
        0.01  # White noise: std dev (events/yr); OU: diffusion coefficient
    )
    perturbation_mean = 0.0  # OU process only: mean perturbation level (events/yr)
    perturbation_theta = 1.0  # OU process only: reversion rate (1/years)

    # Afterslip
    # To decrease the total afterslip moment
    # - Decrease reference velocity: afterslip_v_ref_m_yr
    # - Reduce spatial extent
    #   afterslip_spatial_threshold = 0.5  # Was 0.3 - larger threshold = smaller halo
    #   afterslip_correlation_length_x_km = 5.0  # Was 7.5 - faster spatial decay
    #   afterslip_correlation_length_z_km = 1.5  # Was 2.5
    # - Increase magnitude threshold so that only large events generate afterslip
    #   afterslip_M_min = 7.0
    # - Increase minimum moment threshold to require more residual moment to participate
    #   afterslip_m_critical = 0.1
    # - Disable afterslip: afterslip_enabled = False

    afterslip_enabled = True  # Enable/disable afterslip physics
    afterslip_v_ref_m_yr = 0.01  # Reference initial velocity (m/yr) at M_ref
    afterslip_M_ref = 7.0  # Reference magnitude for velocity scaling
    afterslip_beta = 0.33  # Magnitude scaling exponent (geometric argument)
    afterslip_correlation_length_x_km = (
        10.001  # Spatial correlation length xi_x (along-strike)
    )
    afterslip_correlation_length_z_km = (
        10.001  # Spatial correlation length xi_z (down-dip)
    )
    afterslip_kernel_type = "exponential"  # 'exponential' or 'power_law'
    afterslip_power_law_exponent = 2.5  # Exponent if using power_law kernel
    afterslip_duration_years = 100.0  # Track sequences for this many years
    afterslip_m_critical = 0.1  # Minimum residual moment for afterslip (m)
    afterslip_v_min = 1e-6  # Minimum velocity for numerical stability (m/yr)
    afterslip_M_min = 7.0  # Only trigger afterslip for M >= this threshold [RAISED]
    afterslip_spatial_threshold = (
        0.3  # Only allow afterslip where Phi > threshold [NEW]
    )
    # If > 0, cap a sequence's moment budget at this fraction of the
    # mainshock's coseismic geometric moment (legacy 0 = budget is all the
    # residual deficit in the halo, which scales with the reservoir).
    afterslip_budget_fraction = 1.0

    # Moment initialization fraction
    spinup_fraction = 0.25  # Initialize with this fraction of mid-cycle moment (0.25 = recurrence_time/4)
    # If > 0, initialize m = slip_rate * spinup_years directly (D_0 = spinup_years * L_tot),
    # bypassing the M_max-keyed recurrence-time formula above.
    spinup_years = 250.0

    # Coseismic slip parameters
    slip_decay_rate = 2.0  # Exponential decay rate of slip from hypocenter
    slip_heterogeneity = 0.01  # Random perturbation amplitude

    # GR scaling
    b_value = 1.0

    # Deficit-weighted magnitude law for loading-origin events (magnitude_law.py):
    #   ln p(M | D) = ln p_GR(M) + theta(D) * phi(M), a moment-constrained MaxEnt
    #   tilt of the truncated G-R law with theta(D_ref) = 0 (so the calibration
    #   at D_ref is unchanged). mu = d ln E[m] / d ln D at D_ref; 0 = legacy
    #   i.i.d. G-R with a bit-identical random stream.
    magnitude_tilt_mu = 0.0
    # phi(M): "moment" = (m/m_max)^p (Kagan tilt, acts on 10^(1.5 M): top-selective);
    # "gamma" = gamma(M)/gamma(M_max) (joint MaxEnt over (i, M), concave, lifts
    # M6-6.5 nearly as much as M7+); "linear" = (M - M_min)/(M_max - M_min) (b-value tilt)
    magnitude_tilt_shape = "moment"
    magnitude_tilt_power = 1.0  # p for the moment shape (2 = more top-selective)
    # "loglinear": theta = mu ln(D/D_w) / c_phi (soft; default); "power": theta
    # solved so E[m|D] = E_GR (D/D_w)^mu exactly (hard low-D taper)
    magnitude_tilt_target = "loglinear"
    # "reservoir": D = A sum(m_i) (global); "shadow": stress-shadow-weighted log
    # effective deficit (needs rate_law = memory; ~10% extra local variance)
    magnitude_tilt_deficit = "reservoir"
    magnitude_tilt_pivot = 5.0  # tilt acts on max(phi(M) - phi(M_p), 0); M_min = off
    magnitude_tilt_reference_years = 0.0  # neutral point D_w (yr of loading); 0 = deficit_reference_years
    magnitude_tilt_theta_max = 200.0
    magnitude_cap_fill_fraction = 0.0  # ex-ante cap m(M) <= this * D (0 = off; ~0.5 with M_max = 8)
    magnitude_grid_n = 2001  # magnitude grid for the tabulated inverse CDF
    # Loading-origin magnitude law floor / b-value (0 = the catalog M_min / b_value).
    # With the Bath split, magnitude_load_M_min = 6 makes loading nucleate only
    # M >= 6 events and leaves every smaller event to the aftershock cascade:
    # moment balance then gives lambda_load = 0.8 L / E_GR[m; 6, 7.5] ~ 0.17/yr.
    magnitude_load_M_min = 0.0
    magnitude_load_b = 0.0
    # Rate coupling to the size law: lambda_load *= (E_GR / E[m|D])^kappa with the
    # ratio clamped to [1/R, R]. kappa = 0: lambda_0 fixed, the size feedback
    # stiffens the reservoir loop (nu_eff = nu + mu); kappa = 1: instantaneous
    # moment balance, count rate ~ D^(nu - mu) (fewer-but-bigger when full).
    rate_size_coupling = 0.0
    rate_size_coupling_max = 3.0
    # Bath split (library version of experiment_bath_split.py): each event is
    # loading-origin (deficit-weighted draw, nucleation ~ m^gamma h, no aftershock
    # weighting) or the child of an Omori parent chosen in proportion to its
    # step rate (G-R capped at M_parent - dM, nucleation ~ Phi_parent m^gamma h).
    # Parents with M - dM < M_min produce no rate; the moment balance uses the
    # cascade budget lambda = (1 - f_as) L / (E_L + n_L E_O / (1 - n_O)).
    # False = legacy aggregated Omori path (bit-identical).
    omori_split_enabled = False
    omori_bath_dM = 1.2
    omori_bath_K_scale = 1.0  # K multiplier for eligible parents (n_b/n_L ~ 2.64 restores the total branching ratio)

    # Time
    duration_years = 1000.0  # Full simulation duration
    time_step_years = 1.0  # Time resolution (years)

    # Random seed for reproducibility
    random_seed = 43

    # Output
    output_dir = "results"
    output_hdf5 = "simulation_results.h5"
    hdf5_compression = (
        0  # gzip compression level (0=none for speed, 4=balanced, 9=max compression)
    )
    snapshot_interval_years = (
        1.0  # Save moment snapshots every N years (1.0 = every timestep)
    )
    # Per-timestep scalar history (rate components, correction factor,
    # coupling, deficit, event counts) in the HDF5 'step/' group
    save_step_history = True

    def compute_derived_parameters(self):
        """Compute parameters that depend on others"""
        # Grid dimensions
        self.n_along_strike = int(self.fault_length_km / self.element_size_km)
        self.n_down_dip = int(self.fault_depth_km / self.element_size_km)
        self.n_elements = self.n_along_strike * self.n_down_dip

        # Element area
        self.element_area_m2 = (self.element_size_km * 1000) ** 2

        # Time steps
        self.n_time_steps = int(self.duration_years / self.time_step_years)

        # Convert slip rates to m/year
        self.background_slip_rate_m_yr = self.background_slip_rate_mm_yr / 1000.0

        # Guard the size-law / rate-coupling combination that runs away
        # (instantaneous balance with a steep tilt: bursty release, corr pinned)
        kappa = getattr(self, "rate_size_coupling", 0.0)
        mu = getattr(self, "magnitude_tilt_mu", 0.0)
        if kappa > 0.5 and mu > 3.0:
            raise ValueError(
                f"rate_size_coupling = {kappa} with magnitude_tilt_mu = {mu}: keep mu <= 3 "
                "when kappa > 0.5 (reservoir runaway)"
            )

        # Total moment accumulation rate (N-m/year)
        total_slip_rate = self.background_slip_rate_m_yr * self.n_elements
        self.total_moment_rate = (
            self.shear_modulus_Pa * self.element_area_m2 * total_slip_rate
        )

        print(
            f"Grid: {self.n_along_strike} x {self.n_down_dip} = {self.n_elements} elements"
        )
        print(f"Total moment rate: {self.total_moment_rate:.2e} N-m/year")

    def to_dict(self):
        """
        Serialize configuration to dictionary for HDF5 storage

        Returns:
        --------
        config_dict : dict
            Dictionary of all config parameters (excludes methods and private attrs).
            Includes class-level defaults (walking the MRO, derived classes win)
            overlaid with instance attributes (overrides and derived values).
        """
        config_dict = {}
        for klass in reversed(type(self).__mro__):
            if klass is object:
                continue
            for key, value in vars(klass).items():
                if key.startswith("_") or callable(value):
                    continue
                if isinstance(value, (classmethod, staticmethod, property)):
                    continue
                config_dict[key] = value
        for key, value in vars(self).items():
            if key.startswith("_") or callable(value):
                continue
            config_dict[key] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        """
        Reconstruct Config object from dictionary

        Parameters:
        -----------
        config_dict : dict
            Dictionary of config parameters

        Returns:
        --------
        config : Config
            Reconstructed configuration object
        """
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config
