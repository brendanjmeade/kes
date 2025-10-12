"""
Main simulation loop with rate-based event generation
"""

import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

from geometry import create_fault_mesh
from moment import initialize_moment, accumulate_moment
from event_generator import generate_events_in_timestep
from temporal_prob import compute_rate_parameters


def run_simulation(config):
    """
    Run full earthquake simulation with rate-based generation

    Returns:
    --------
    results : dict containing all simulation data
    """
    print("=" * 70)
    print("STRIKE-SLIP FAULT EARTHQUAKE SIMULATOR")
    print("Rate-based event generation (can have multiple events per step)")
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
    print("Initializing moment distribution...")
    m_current, slip_rate = initialize_moment(config, mesh)

    # Storage for results
    event_history = []
    moment_snapshots = []
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

        # Generate events (can be 0, 1, 2, ...)
        events, m_current = generate_events_in_timestep(
            m_current, event_history, current_time, dt_years, mesh, config
        )

        # Add to history
        if len(events) > 0:
            event_history.extend(events)
            if len(events) == 1:
                tqdm.write(
                    f"Event {len(event_history)}: t={current_time:.2f} yr, M={events[0]['magnitude']:.2f}"
                )
            else:
                tqdm.write(
                    f"{len(events)} events at t={current_time:.2f} yr: M={[e['magnitude'] for e in events]}"
                )

        # Save snapshots (every year)
        if i % int(365.25 / config.time_step_days) == 0:
            moment_snapshots.append(m_current.copy())
            snapshot_times.append(current_time)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print(f"  Total events: {len(event_history)}")
    print(
        f"  Average rate: {len(event_history) / config.duration_years:.6f} events/year"
    )

    # Count timesteps with multiple events
    timesteps_with_events = {}
    for event in event_history:
        t = event["time"]
        timesteps_with_events[t] = timesteps_with_events.get(t, 0) + 1

    multi_event_steps = sum(1 for count in timesteps_with_events.values() if count > 1)
    print(f"  Timesteps with multiple events: {multi_event_steps}")
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
