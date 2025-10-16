"""
Configuration file for strike-slip fault simulator
All parameters in one place for easy tuning
"""

import numpy as np


class Config:
    """Simulation configuration"""

    # === GEOMETRY ===
    fault_length_km = 200.0  # Along-strike length (km)
    fault_depth_km = 25.0  # Down-dip depth (km)
    element_size_km = 1.0  # Grid cell size (km)

    # === MOMENT ACCUMULATION ===
    # Background slip deficit rate
    background_slip_rate_mm_yr = 10.0  # mm/year

    # Gaussian moment pulse(s) - list of dicts
    moment_pulses = [
        {
            "center_x_km": 100.0,  # Along-strike position
            "center_z_km": 12.5,  # Down-dip position
            "sigma_km": 20.0,  # Width of Gaussian
            "amplitude_mm_yr": 30.0,  # Additional slip rate at center
        }
    ]

    # === PHYSICAL PARAMETERS ===
    shear_modulus_Pa = 3e10  # Pa (30 GPa)

    # === MAGNITUDE-DEPENDENT SPATIAL PROBABILITY ===
    gamma_min = 0.5  # Small events (SOC)
    gamma_max = 1.5  # Large events (G-R)
    alpha_spatial = 0.35  # Decay rate
    M_min = 5.0  # Minimum magnitude
    M_max = 8.0  # Maximum magnitude

    # === ADAPTIVE RATE CORRECTION ===
    adaptive_correction_enabled = (
        False  # Enable adaptive correction (False = fixed C, natural coupling)
    )
    adaptive_correction_gain = (
        5.0  # Proportional control gain for coupling correction (continuous updates)
    )
    correction_factor_min = 0.1  # Minimum allowed correction factor
    correction_factor_max = 10.0  # Maximum allowed correction factor

    # === OMORI AFTERSHOCK PARAMETERS ===
    # Standard Omori-Utsu law: λ_aftershock(t) = K / (t + c)^p
    # where K scales with mainshock magnitude: K = K_ref × 10^(alpha × (M - M_ref))
    omori_enabled = True  # Enable/disable aftershock sequences
    omori_p = 1.0  # Decay exponent (typically ~1.0)
    omori_c_days = 1.0  # Time offset in days (will be converted to years)
    omori_K_ref = 0.1  # Productivity (events/year) for M=6 mainshock
    omori_M_ref = 6.0  # Reference magnitude for productivity
    omori_alpha = 0.8  # Magnitude scaling (Reasenberg & Jones 1989)
    omori_duration_years = (
        10.0  # Only track aftershocks for this many years after mainshock
    )

    # === MOMENT INITIALIZATION ===
    spinup_fraction = 0.25  # Initialize with this fraction of mid-cycle moment (0.25 = recurrence_time/4)

    # === SLIP DISTRIBUTION ===
    slip_decay_rate = 2.0  # Exponential decay rate of slip from hypocenter
    slip_heterogeneity = 0.3  # Random perturbation amplitude (±30%)

    # === GUTENBERG-RICHTER ===
    b_value = 1.0

    # === SIMULATION ===
    duration_years = 1000.0  # Full simulation duration
    time_step_days = 1.0  # Time resolution

    # Random seed for reproducibility
    random_seed = 42

    # === OUTPUT ===
    output_dir = "results"
    output_pickle = "simulation_results.pkl"
    snapshot_interval_days = (
        1.0  # Save moment snapshots every N days (1.0 = every timestep)
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
        self.n_time_steps = int(self.duration_years * 365.25 / self.time_step_days)

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
