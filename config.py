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
    # element_size_km = 0.1  # Grid cell size (km)

    # Background moment (slip deficit) rate
    background_slip_rate_mm_yr = 10.0  # mm/year

    # Gaussian moment pulse(s) - list of dicts
    moment_pulses = [
        {
            "center_x_km": 100.0,  # Along-strike position
            "center_z_km": 12.5,  # Down-dip position
            "sigma_km": 20.0,  # Width of Gaussian
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
    M_max = 8.0  # Maximum magnitude

    # Adaptive rate correction
    adaptive_correction_enabled = (
        True  # Enable adaptive correction (True = drives coupling toward 1.0)
    )
    adaptive_correction_gain = (
        5.0  # Proportional control gain for coupling correction (continuous updates)
    )
    correction_factor_min = 0.1  # Minimum allowed correction factor
    correction_factor_max = 10.0  # Maximum allowed correction factor

    # Omori aftershock parameters
    # Standard Omori-Utsu law: \rho_aftershock(t) = K / (t + c)^p
    # where K scales with mainshock magnitude: K = K_ref × 10^(\alpha × (M - M_ref))
    omori_enabled = True  # Enable/disable aftershock sequences
    omori_p = 1.0  # Decay exponent (typically ~1.0)
    omori_c_years = 1.0 / 365.25  # Time offset in years (~0.00274 years = 1 day)
    omori_K_ref = 0.1  # Productivity (events/year) for M=6 mainshock
    omori_M_ref = 6.0  # Reference magnitude for productivity
    omori_alpha = 0.8  # Magnitude scaling (Reasenberg & Jones 1989)
    omori_duration_years = (
        30.0  # Only track aftershocks for this many years after mainshock
    )

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
        10.001  # Spatial correlation length ξ_x (along-strike)
    )
    afterslip_correlation_length_z_km = (
        10.001  # Spatial correlation length ξ_z (down-dip)
    )
    afterslip_kernel_type = "exponential"  # 'exponential' or 'power_law'
    afterslip_power_law_exponent = 2.5  # Exponent if using power_law kernel
    afterslip_duration_years = 100.0  # Track sequences for this many years
    afterslip_m_critical = 0.1  # Minimum residual moment for afterslip (m)
    afterslip_v_min = 1e-6  # Minimum velocity for numerical stability (m/yr)
    afterslip_M_min = 7.0  # Only trigger afterslip for M ≥ this threshold [RAISED]
    afterslip_spatial_threshold = (
        0.3  # Only allow afterslip where Phi > threshold [NEW]
    )

    # Moment initialization fraction
    spinup_fraction = 0.25  # Initialize with this fraction of mid-cycle moment (0.25 = recurrence_time/4)

    # Coseismic slip parameters
    slip_decay_rate = 2.0  # Exponential decay rate of slip from hypocenter
    slip_heterogeneity = 0.01  # Random perturbation amplitude

    # GR scaling
    b_value = 1.0

    # Time
    duration_years = 1000.0  # Full simulation duration
    time_step_years = 1.0  # Time resolution (years)

    # Random seed for reproducibility
    random_seed = 42

    # Output
    output_dir = "results"
    output_hdf5 = "simulation_results.h5"
    hdf5_compression = (
        0  # gzip compression level (0=none for speed, 4=balanced, 9=max compression)
    )
    snapshot_interval_years = (
        1.0  # Save moment snapshots every N years (1.0 = every timestep)
    )

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

        # Total moment accumulation rate (N·m/year)
        total_slip_rate = self.background_slip_rate_m_yr * self.n_elements
        self.total_moment_rate = (
            self.shear_modulus_Pa * self.element_area_m2 * total_slip_rate
        )

        print(
            f"Grid: {self.n_along_strike} x {self.n_down_dip} = {self.n_elements} elements"
        )
        print(f"Total moment rate: {self.total_moment_rate:.2e} N·m/year")

    def to_dict(self):
        """
        Serialize configuration to dictionary for HDF5 storage

        Returns:
        --------
        config_dict : dict
            Dictionary of all config parameters (excludes methods and private attrs)
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            # Skip private attributes and methods
            if key.startswith("_") or callable(value):
                continue
            # Store all simple types
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
