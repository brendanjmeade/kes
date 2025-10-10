"""
Main simulation loop
"""

import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

from geometry import create_fault_mesh
from moment import initialize_moment, accumulate_moment
from event_generator import generate_event
from temporal_prob import compute_exponential_rate_parameters


def run_simulation(config):
    """
    Run full earthquake simulation

    Returns:
    --------
    results : dict containing all simulation data
    """
    print("=" * 70)
    print("STRIKE-SLIP FAULT EARTHQUAKE SIMULATOR")
    print("With MaxEnt spatial distribution and exponential temporal rate")
    print("=" * 70)

    print("\nInitializing simulation...")

    # Set random seed
    np.random.seed(config.random_seed)

    # Compute derived parameters
    config.compute_derived_parameters()

    # Compute temporal probability parameters from moment balance
    lambda_0, beta, C_a, C_r = compute_exponential_rate_parameters(config)

    # Store in config
    config.lambda_0 = lambda_0
    config.beta_rate = beta
    config.C_a = C_a
    config.C_r = C_r

    print("\n" + "=" * 70)
    print("PARAMETERS SET - Ready to simulate")
    print("=" * 70)

    # Create fault mesh
    print("\nCreating fault mesh...")
    mesh = create_fault_mesh(config)

    # Initialize moment distribution
    print("Initializing moment distribution...")
    m_current, slip_rate = initialize_moment(config, mesh)

    print(f"  Background slip rate: {config.background_slip_rate_mm_yr} mm/year")
    print(f"  Number of moment pulses: {len(config.moment_pulses)}")

    # Storage for results
    event_history = []
    moment_snapshots = []  # Store moment field at intervals
    snapshot_times = []

    # Time array
    times = np.linspace(0, config.duration_years, config.n_time_steps)
    dt_years = config.time_step_days / 365.25

    print(f"\n" + "=" * 70)
    print(f"RUNNING SIMULATION")
    print(f"  Duration: {config.duration_years} years")
    print(f"  Time steps: {config.n_time_steps}")
    print(f"  Time step size: {config.time_step_days} days = {dt_years:.6f} years")
    print("=" * 70 + "\n")

    # Simulation loop
    for i, current_time in enumerate(tqdm(times, desc="Simulating")):
        # Accumulate moment
        m_current = accumulate_moment(
            m_current, slip_rate, config.element_area_m2, dt_years
        )

        # Try to generate event
        event, m_current = generate_event(
            m_current, event_history, current_time, mesh, config
        )

        if event is not None:
            event_history.append(event)
            tqdm.write(
                f"Event {len(event_history)}: t={current_time:.2f} yr, M={event['magnitude']:.2f}"
            )

        # Save snapshots (every year)
        if i % int(365.25 / config.time_step_days) == 0:
            moment_snapshots.append(m_current.copy())
            snapshot_times.append(current_time)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print(f"  Total events: {len(event_history)}")
    print(
        f"  Average rate: {len(event_history) / config.duration_years:.4f} events/year"
    )
    print(
        f"  Target rate: {lambda_0 * np.exp(beta * C_a * np.mean([np.sum(m) for m in moment_snapshots])):.4f} events/year"
    )
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
