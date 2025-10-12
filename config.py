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
    M_min = 6.5  # Minimum magnitude
    M_max = 8.0  # Maximum magnitude

    # === TEMPORAL PROBABILITY ===
    # Loading coefficient (will be computed from moment balance)
    C_a = None  # Computed in simulator

    # Omori parameters
    omori_p = 1.0  # Universal exponent
    omori_c_days = 1.0  # Changed from 0.1 to 1.0
    omori_alpha_beta = 0.8  # Productivity scaling
    omori_beta_0 = 1e-12  # Increased from 1.0 to 10.0

    # Depletion parameters
    psi = 2.0 / 3.0  # Sublinear exponent (from theory)
    C_r = None  # Will be computed from theory

    # Wrapping function parameters
    lambda_min = 1e-6  # Minimum event rate
    lambda_max = 1.0  # Increased from 0.05 to 1.0
    gamma_temporal = 1.0  # Keep at 1.0 since we rescaled C_a
    # === GUTENBERG-RICHTER ===
    b_value = 1.0

    # === SIMULATION ===
    duration_years = 3000.0
    time_step_days = 1.0  # Time resolution

    # Random seed for reproducibility
    random_seed = 42

    # === OUTPUT ===
    output_dir = "results"
    output_pickle = "simulation_results.pkl"

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
