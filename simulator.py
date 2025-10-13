"""
Main simulation loop with moment-based rate generation
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
from temporal_prob import (
    compute_rate_parameters,
    earthquake_rate,
    update_rate_correction,
)
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
    Run full earthquake simulation with adaptive moment-based rate generation

    Uses λ(t) = C × correction_factor(t) × moment_deficit(t)
    The correction factor adapts to maintain moment balance

    Returns:
    --------
    results : dict containing all simulation data
    """
    print("=" * 70)
    print("STRIKE-SLIP FAULT EARTHQUAKE SIMULATOR")
    print("Adaptive moment-based rate generation")
    print("=" * 70)

    print("\nInitializing simulation...")

    # Set random seed
    np.random.seed(config.random_seed)

    # Compute derived parameters
    config.compute_derived_parameters()

    # Compute rate coefficient from moment balance
    C_rate = compute_rate_parameters(config)

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

    # Deterministic accumulator for fractional events
    event_debt = 0.0

    # Track cumulative moment for rate calculation
    initial_moment = np.sum(m_current)
    cumulative_loading = 0.0  # FIX: Start at zero, add initial moment to first update
    cumulative_release = 0.0  # No events yet

    # Adaptive correction update interval
    correction_update_interval = int(
        100 * 365.25 / config.time_step_days
    )  # Every 100 years (increased from 1000 for faster response)

    print("\n" + "=" * 70)
    print("RUNNING SIMULATION")
    print(f"  Duration: {config.duration_years} years")
    print(f"  Time steps: {config.n_time_steps}")
    print(f"  Time step size: {config.time_step_days} days = {dt_years:.6f} years")
    print("  Event generation: Deterministic accumulator with adaptive rate")
    print(f"  Initial moment (spin-up): {initial_moment:.2e} m³")
    print(f"  Cumulative loading accounting: starts at 0.0 m³")
    print(
        f"  Rate correction update interval: {correction_update_interval * dt_years:.0f} years"
    )
    print("=" * 70 + "\n")

    # Diagnostics: Track expected vs actual loading
    total_loading_rate = np.sum(slip_rate * config.element_area_m2)
    print(f"DEBUG: Total loading rate = {total_loading_rate:.2e} m³/yr")

    # Simulation loop
    for i, current_time in enumerate(tqdm(times, desc="Simulating")):
        # Accumulate moment
        m_current = accumulate_moment(
            m_current, slip_rate, config.element_area_m2, dt_years
        )

        # Update cumulative loading
        moment_added = np.sum(slip_rate * config.element_area_m2) * dt_years
        cumulative_loading += moment_added

        # Compute instantaneous rate based on moment deficit
        # NOTE: Correction update moved to END of loop (after events)
        lambda_t, components = earthquake_rate(
            m_current,
            event_history,
            current_time,
            config,
            cumulative_loading,
            cumulative_release,
        )

        # Accumulate fractional events
        event_debt += lambda_t * dt_years

        # Generate integer number of events when debt ≥ 1
        n_events = int(event_debt)
        event_debt -= n_events

        # Generate each event
        if n_events > 0:
            m_working = m_current.copy()

            # for j in range(n_events):
            #     # Draw magnitude from G-R distribution
            #     magnitude = draw_magnitude(config)

            #     # Compute spatial probability for this magnitude
            #     p_spatial, gamma_used = spatial_probability(
            #         m_working, magnitude, config
            #     )

            #     # Sample hypocenter location
            #     hypocenter_idx = np.random.choice(config.n_elements, p=p_spatial)

            #     # Generate slip distribution (constrained to available moment)
            #     slip, ruptured_elements, M0_actual = generate_slip_distribution(
            #         hypocenter_idx, magnitude, m_working, mesh, config
            #     )

            #     M_actual = seismic_moment_to_magnitude(M0_actual)

            #     # Compute geometric moment released
            #     geom_moment_released = M0_actual / config.shear_modulus_Pa

            #     # Update moment distribution (release slip)
            #     m_working = release_moment(m_working, slip, config.element_area_m2)

            #     # Update cumulative release
            #     cumulative_release += geom_moment_released

            #     # Create event record
            #     hypo_x = mesh["centroids"][hypocenter_idx, 0]
            #     hypo_z = mesh["centroids"][hypocenter_idx, 2]

            #     event = {
            #         "time": current_time,
            #         "magnitude": M_actual,
            #         "M0": M0_actual,
            #         "hypocenter_idx": hypocenter_idx,
            #         "hypocenter_x_km": hypo_x,
            #         "hypocenter_z_km": hypo_z,
            #         "ruptured_elements": ruptured_elements,
            #         "slip": slip,
            #         "gamma_used": gamma_used,
            #         "lambda_t": lambda_t,
            #         "components": components,
            #     }

            #     event_history.append(event)

            #     # Print progress
            #     if len(event_history) == 1:
            #         tqdm.write(
            #             f"Event 1: t={current_time:.2f} yr, M={M_actual:.2f}, λ={lambda_t:.6f}/yr"
            #         )
            #     elif len(event_history) % 100 == 0:
            #         tqdm.write(
            #             f"Event {len(event_history)}: t={current_time:.2f} yr, M={M_actual:.2f}, "
            #             f"coupling={cumulative_release / cumulative_loading:.3f}"
            #         )
            # Around line 140-180 in the event generation loop

            for j in range(n_events):
                # Draw magnitude from G-R distribution
                magnitude = draw_magnitude(config)

                # Compute spatial probability for this magnitude
                p_spatial, gamma_used = spatial_probability(
                    m_working, magnitude, config
                )

                # Sample hypocenter location
                hypocenter_idx = np.random.choice(config.n_elements, p=p_spatial)

                # Generate slip distribution (constrained to available moment)
                slip, ruptured_elements, M0_actual = generate_slip_distribution(
                    hypocenter_idx, magnitude, m_working, mesh, config
                )

                M_actual = seismic_moment_to_magnitude(M0_actual)

                # Compute ACTUAL geometric moment released from the slip distribution
                # This is the authoritative value - use slip that was actually applied
                geom_moment_released = np.sum(slip * config.element_area_m2)

                # Update moment distribution (release slip)
                m_working = release_moment(m_working, slip, config.element_area_m2)

                # Update cumulative release with actual geometric moment
                cumulative_release += geom_moment_released

                # Create event record
                hypo_x = mesh["centroids"][hypocenter_idx, 0]
                hypo_z = mesh["centroids"][hypocenter_idx, 2]

                event = {
                    "time": current_time,
                    "magnitude": M_actual,
                    "M0": M0_actual,
                    "geom_moment": geom_moment_released,  # CRITICAL: Store actual geometric moment
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
                elif len(event_history) % 100 == 0:
                    aftershock_info = ""
                    if components["aftershock"] > 0:
                        aftershock_info = f", aftershock λ={components['aftershock']:.4f}/yr ({components['n_active_sequences']} sequences)"
                    tqdm.write(
                        f"Event {len(event_history)}: t={current_time:.2f} yr, M={M_actual:.2f}, "
                        f"coupling={cumulative_release / cumulative_loading:.3f}{aftershock_info}"
                    )

                # Print diagnostic for large events that will trigger significant aftershocks
                if M_actual >= 6.5 and hasattr(config, "omori_enabled") and config.omori_enabled:
                    K = config.omori_K_ref * 10 ** (
                        config.omori_alpha * (M_actual - config.omori_M_ref)
                    )
                    tqdm.write(
                        f"  → Large event (M={M_actual:.2f}) will trigger aftershocks with productivity K={K:.3f} events/yr"
                    )
            # Update m_current with all releases from this timestep
            m_current = m_working

        # Save snapshots (every year)
        if i % int(365.25 / config.time_step_days) == 0:
            moment_snapshots.append(m_current.copy())
            snapshot_times.append(current_time)

        # FIX: Update adaptive rate correction periodically AFTER events are generated
        # This ensures correction sees actual coupling from completed events
        if i % correction_update_interval == 0 and i > 0:
            update_rate_correction(
                config, cumulative_loading, cumulative_release, current_time
            )

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print(f"  Total events: {len(event_history)}")

    # Diagnostic: correction update statistics
    expected_updates = int(config.duration_years / 100)
    actual_updates = len(config.coupling_history)
    print(f"\n  Adaptive Correction Updates:")
    print(f"    Expected: {expected_updates} (every 100 years)")
    print(f"    Actual: {actual_updates}")
    print(f"    Match: {actual_updates == expected_updates}")

    if len(event_history) > 0:
        total_M0_released = sum(e["M0"] for e in event_history)
        total_M0_loaded = (
            config.shear_modulus_Pa
            * np.sum(slip_rate)
            * config.element_area_m2
            * config.duration_years
        )
        coupling = total_M0_released / total_M0_loaded
        print(
            f"\n  Average rate: {len(event_history) / config.duration_years:.6f} events/year"
        )
        print(f"  Final seismic coupling: {coupling:.4f}")
        print(f"  Final correction factor: {config.rate_correction_factor:.4f}")
        print(f"  Cumulative loading: {cumulative_loading:.2e} m³")
        print(f"  Cumulative release: {cumulative_release:.2e} m³")
        print(f"  Final deficit: {cumulative_loading - cumulative_release:.2e} m³")
        print(f"  Geometric coupling: {cumulative_release / cumulative_loading:.4f}")
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
        "cumulative_loading": cumulative_loading,
        "cumulative_release": cumulative_release,
        "coupling_history": config.coupling_history,
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
