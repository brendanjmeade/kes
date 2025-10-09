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
from temporal_prob import compute_C_a, compute_C_r


def run_simulation(config):
    """
    Run full earthquake simulation

    Returns:
    --------
    results : dict containing all simulation data
    """
    print("Initializing simulation...")

    # Set random seed
    np.random.seed(config.random_seed)

    # Compute derived parameters
    config.compute_derived_parameters()

    # Compute C_a and C_r
    config.C_a = compute_C_a(config)
    config.C_r = compute_C_r(config, config.C_a)

    print(f"C_a = {config.C_a:.2e}")
    print(f"C_r = {config.C_r:.2e}")

    # Create fault mesh
    mesh = create_fault_mesh(config)

    # Initialize moment distribution
    m_current, slip_rate = initialize_moment(config, mesh)

    # Storage for results
    event_history = []
    moment_snapshots = []  # Store moment field at intervals
    snapshot_times = []

    # Time array
    times = np.linspace(0, config.duration_years, config.n_time_steps)
    dt_years = config.time_step_days / 365.25

    print(f"\nRunning simulation for {config.duration_years} years...")
    print(f"Time steps: {config.n_time_steps}")

    # Simulation loop
    for i, current_time in enumerate(times):
        # for i, current_time in enumerate(tqdm(times, desc="Simulating")):
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
            print(
                f"Event {len(event_history)}: t={current_time:.2f} yr, M={event['magnitude']:.2f}"
            )

        # Save snapshots (every 1 year)
        if i % int(365.25 / config.time_step_days) == 0:
            moment_snapshots.append(m_current.copy())
            snapshot_times.append(current_time)

    print(f"\nSimulation complete!")
    print(f"Total events: {len(event_history)}")

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
