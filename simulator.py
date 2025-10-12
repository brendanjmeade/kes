"""
Main simulation loop with rate-based event generation using deterministic accumulator
"""

import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

from geometry import create_fault_mesh
from moment import (
    initialize_moment,
    accumulate_moment,
    release_moment,
    seismic_moment_to_magnitude,
)
from temporal_prob import compute_rate_parameters, earthquake_rate
from spatial_prob import spatial_probability
from slip_generator import generate_slip_distribution


def draw_magnitude(config):
    """
    Sample magnitude from Gutenberg-Richter distribution
    """
    b = config.b_value
    M_min = config.M_min
    M_max = config.M_max

    # Inverse transform sampling
    denom = 10 ** (-b * M_min) - 10 ** (-b * M_max)
    u = np.random.random()

    magnitude = -np.log10(10 ** (-b * M_min) - u * denom) / b

    return magnitude


def run_simulation(config):
    """
    Run full earthquake simulation with rate-based generation

    Uses deterministic accumulator instead of Poisson sampling
    to ensure exact moment balance over long timescales

    Returns:
    --------
    results : dict containing all simulation data
    """
    print("=" * 70)
    print("STRIKE-SLIP FAULT EARTHQUAKE SIMULATOR")
    print("Rate-based event generation with deterministic accumulator")
    print("=" * 70)

    print("\nInitializing simulation...")

    # Set random seed
    np.random.seed(config.random_seed)

    # Compute derived parameters
    config.compute_derived_parameters()

    # Compute rate parameters from moment balance
    lambda_0, C_a, C_r = compute_rate_parameters(config)

    # Store in config
    config.lambda_0 = lambda_0
    config.C_a = C_a
    config.C_r = C_r

    print("\n" + "=" * 70)
    print("PARAMETERS SET - Ready to simulate")
    print("=" * 70)

    # Create fault mesh
    print("\nCreating fault mesh...")
    mesh = create_fault_mesh(config)

    # Initialize moment distribution
    m_current, slip_rate = initialize_moment(config, mesh)

    # Storage for results
    event_history = []
    moment_snapshots = []
    snapshot_times = []

    # Time array
    times = np.linspace(0, config.duration_years, config.n_time_steps)
    dt_years = config.time_step_days / 365.25

    # Deterministic accumulator for fractional events (KEY FIX!)
    event_debt = 0.0

    print(f"\n" + "=" * 70)
    print(f"RUNNING SIMULATION")
    print(f"  Duration: {config.duration_years} years")
    print(f"  Time steps: {config.n_time_steps}")
    print(f"  Time step size: {config.time_step_days} days = {dt_years:.6f} years")
    print(f"  Event generation: Deterministic accumulator (not Poisson)")
    print("=" * 70 + "\n")

    # Simulation loop
    for i, current_time in enumerate(tqdm(times, desc="Simulating")):
        # Accumulate moment
        m_current = accumulate_moment(
            m_current, slip_rate, config.element_area_m2, dt_years
        )

        # Compute instantaneous rate
        lambda_t, components = earthquake_rate(
            m_current, event_history, current_time, config
        )

        # Accumulate fractional events
        event_debt += lambda_t * dt_years

        # Generate integer number of events when debt ≥ 1
        n_events = int(event_debt)
        event_debt -= n_events

        # Generate each event
        if n_events > 0:
            m_working = m_current.copy()

            for j in range(n_events):
                # Draw magnitude from G-R distribution
                magnitude = draw_magnitude(config)

                # Compute spatial probability for this magnitude
                p_spatial, gamma_used = spatial_probability(
                    m_working, magnitude, config
                )

                # Sample hypocenter location
                hypocenter_idx = np.random.choice(config.n_elements, p=p_spatial)

                # Generate slip distribution
                slip, ruptured_elements = generate_slip_distribution(
                    hypocenter_idx, magnitude, m_working, mesh, config
                )

                # Compute actual seismic moment released
                M0_actual = config.shear_modulus_Pa * np.sum(
                    slip * config.element_area_m2
                )
                M_actual = seismic_moment_to_magnitude(M0_actual)

                # Update moment distribution (release slip)
                m_working = release_moment(m_working, slip, config.element_area_m2)

                # Create event record
                hypo_x = mesh["centroids"][hypocenter_idx, 0]
                hypo_z = mesh["centroids"][hypocenter_idx, 2]

                event = {
                    "time": current_time,
                    "magnitude": M_actual,
                    "M0": M0_actual,
                    "hypocenter_idx": hypocenter_idx,
                    "hypocenter_x_km": hypo_x,
                    "hypocenter_z_km": hypo_z,
                    "ruptured_elements": ruptured_elements,
                    "slip": slip,
                    "gamma_used": gamma_used,
                    "lambda_t": lambda_t,
                    "components": components,
                }

                event_history.append(event)

                # Print progress
                if len(event_history) == 1:
                    tqdm.write(
                        f"Event 1: t={current_time:.2f} yr, M={M_actual:.2f}, λ={lambda_t:.6f}/yr"
                    )
                elif len(event_history) % 10 == 0:
                    tqdm.write(
                        f"Event {len(event_history)}: t={current_time:.2f} yr, M={M_actual:.2f}"
                    )

            # Update m_current with all releases from this timestep
            m_current = m_working

        # Save snapshots (every year)
        if i % int(365.25 / config.time_step_days) == 0:
            moment_snapshots.append(m_current.copy())
            snapshot_times.append(current_time)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print(f"  Total events: {len(event_history)}")
    if len(event_history) > 0:
        print(
            f"  Average rate: {len(event_history) / config.duration_years:.6f} events/year"
        )
        print(
            f"  Target rate: {config.lambda_0 + config.C_a * np.mean([np.sum(s) for s in moment_snapshots]):.6f} events/year"
        )
    else:
        print(f"  WARNING: No events generated!")
    print("=" * 70)

    # Compile results
    results = {
        "config": config,
        "mesh": mesh,
        "slip_rate": slip_rate,
        "event_history": event_history,
        "moment_snapshots": moment_snapshots,
        "snapshot_times": snapshot_times,
        "final_moment": m_current,
    }

    return results


def save_results(results, config):
    """
    Save results to pickle file
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / config.output_pickle

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {output_path}")

    return output_path
