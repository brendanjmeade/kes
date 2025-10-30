"""
Main simulation loop with moment-based rate generation
"""

import numpy as np
from tqdm import tqdm
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
from hdf5_io import create_hdf5_file, BufferedHDF5Writer, append_event, finalize_simulation
from afterslip import (
    initialize_afterslip_sequence,
    update_afterslip_sequences,
    get_active_afterslip_sequences,
    compute_aftershock_spatial_weights,
)


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

    # Initialize HDF5 storage
    output_path = Path(config.output_dir) / config.output_hdf5
    output_path.parent.mkdir(exist_ok=True)
    h5file = create_hdf5_file(output_path, config, mesh)
    hdf5_writer = BufferedHDF5Writer(h5file)  # Uses default buffer_size=5000
    print(f"\nHDF5 output file created: {output_path}")
    print(f"  Buffered writes: {hdf5_writer.buffer_size} snapshots (~{hdf5_writer.buffer_size * config.n_elements * 8 / (1024**2):.0f} MB RAM)")
    print(f"  Compression: {'disabled (fast)' if config.hdf5_compression == 0 else f'gzip level {config.hdf5_compression}'}")

    # Storage for in-memory diagnostics
    event_history = []

    # Time array
    times = np.linspace(0, config.duration_years, config.n_time_steps)
    dt_years = config.time_step_years

    # Deterministic accumulator for fractional events
    event_debt = 0.0
    event_debt_history = []  # Track debt over time for visualization
    lambda_history = []  # Track instantaneous rate λ(t) for visualization

    # Afterslip sequence tracking
    afterslip_sequences = []  # Track active afterslip sequences
    afterslip_cumulative = np.zeros(config.n_elements)  # Total afterslip over simulation

    # Track cumulative moment for rate calculation
    initial_moment = np.sum(m_current)
    cumulative_loading = 0.0  # FIX: Start at zero, add initial moment to first update
    cumulative_release = 0.0  # No events yet

    # Track spatial cumulative release for visualization
    m_release_cumulative = np.zeros(config.n_elements)  # Cumulative slip released at each element

    # Track per-element extrema for colorbar scaling
    min_moment_elem = np.inf
    max_moment_elem = -np.inf
    min_release_elem = np.inf
    max_release_elem = -np.inf
    min_deficit_elem = np.inf
    max_deficit_elem = -np.inf

    print("\n" + "=" * 70)
    print("RUNNING SIMULATION")
    print(f"  Duration: {config.duration_years} years")
    print(f"  Time steps: {config.n_time_steps}")
    print(f"  Time step size: {config.time_step_years} years")
    print("  Event generation: Deterministic accumulator")
    print(f"  Initial moment (spin-up): {initial_moment:.2e} m³")
    print(f"  Cumulative loading accounting: starts at 0.0 m³")
    if hasattr(config, "adaptive_correction_enabled") and config.adaptive_correction_enabled:
        print(f"  Rate correction: CONTINUOUS (updated every timestep)")
    else:
        print(f"  Rate correction: DISABLED (fixed C)")
    print("=" * 70 + "\n")

    # Calculate snapshot interval
    snapshot_interval = int(config.snapshot_interval_years / config.time_step_years)
    n_expected_snapshots = config.n_time_steps // snapshot_interval
    snapshot_memory_mb = (
        n_expected_snapshots * config.n_elements * 8 / (1024**2)
    )  # 8 bytes per float64

    print(f"  Snapshot interval: {config.snapshot_interval_years} years ({snapshot_interval} timesteps)")
    print(f"  Expected snapshots: {n_expected_snapshots}")
    print(f"  Estimated snapshot memory: {snapshot_memory_mb:.1f} MB")

    # Simulation loop
    for i, current_time in enumerate(tqdm(times, desc="Simulating")):
        # Update afterslip BEFORE tectonic loading (afterslip reduces moment deficit)
        if config.afterslip_enabled and len(afterslip_sequences) > 0:
            # Get sequences still active at current time
            active_seqs = get_active_afterslip_sequences(
                afterslip_sequences, current_time, config.afterslip_duration_years
            )

            if len(active_seqs) > 0:
                # Update velocities and compute moment release
                afterslip_release = update_afterslip_sequences(
                    active_seqs, current_time, dt_years, config
                )

                # Apply afterslip release to moment field (reduces deficit)
                m_current -= afterslip_release  # Element-wise slip (m)

                # Track cumulative afterslip
                afterslip_cumulative += afterslip_release

                # Track in global cumulative release (afterslip is aseismic release)
                geom_moment_afterslip = np.sum(afterslip_release * config.element_area_m2)
                cumulative_release += geom_moment_afterslip

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

        # Store debt BEFORE event subtraction (to capture peaks > 1.0)
        event_debt_history.append(event_debt)

        # Store instantaneous rate
        lambda_history.append(lambda_t)

        # Generate integer number of events when debt ≥ 1
        n_events = int(event_debt)
        event_debt -= n_events

        # Generate each event
        if n_events > 0:
            m_working = m_current.copy()

            # Compute spatial weights for aftershock localization (if enabled)
            aftershock_spatial_weights, n_active_aftershock_seqs = compute_aftershock_spatial_weights(
                event_history, current_time, config
            )

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

                # Compute spatial probability for this magnitude (with aftershock weighting)
                p_spatial, gamma_used = spatial_probability(
                    m_working, magnitude, config, aftershock_spatial_weights
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

                # Update spatial cumulative release (for visualization)
                m_release_cumulative += slip

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

                # Write event to HDF5
                append_event(h5file, event)

                # Trigger afterslip if magnitude is large enough
                if config.afterslip_enabled and M_actual >= config.afterslip_M_min:
                    # Initialize afterslip sequence (uses m_working which already has slip removed)
                    afterslip_seq = initialize_afterslip_sequence(
                        event, m_working, mesh, config
                    )
                    afterslip_sequences.append(afterslip_seq)

                    # Store reference to spatial activation in event (for aftershock localization)
                    event['spatial_activation'] = afterslip_seq['Phi']
                    event['afterslip_sequence_id'] = len(afterslip_sequences) - 1
                else:
                    event['spatial_activation'] = None
                    event['afterslip_sequence_id'] = None

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

        # Update adaptive rate correction continuously (every timestep)
        # This ensures correction sees actual coupling from completed events
        update_rate_correction(
            config, cumulative_loading, cumulative_release, current_time, dt_years
        )

        # Save snapshots AFTER events and correction (captures full state)
        # Save at configured interval (default: every timestep)
        if i % snapshot_interval == 0:
            # Update per-element extrema
            min_moment_elem = min(min_moment_elem, np.min(m_current))
            max_moment_elem = max(max_moment_elem, np.max(m_current))
            min_release_elem = min(min_release_elem, np.min(m_release_cumulative))
            max_release_elem = max(max_release_elem, np.max(m_release_cumulative))

            deficit = m_current - m_release_cumulative
            min_deficit_elem = min(min_deficit_elem, np.min(deficit))
            max_deficit_elem = max(max_deficit_elem, np.max(deficit))

            # Buffered write to HDF5
            hdf5_writer.append(current_time, m_current, m_release_cumulative, event_debt, lambda_t, afterslip_cumulative)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print(f"  Total events: {len(event_history)}")

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
        if hasattr(config, "adaptive_correction_enabled") and config.adaptive_correction_enabled:
            print(f"  Final correction factor: {config.rate_correction_factor:.4f}")
        else:
            print(f"  Correction factor: {config.rate_correction_factor:.4f} (fixed, no adaptation)")
        print(f"  Cumulative loading: {cumulative_loading:.2e} m³")
        print(f"  Cumulative release: {cumulative_release:.2e} m³")
        print(f"  Final deficit: {cumulative_loading - cumulative_release:.2e} m³")
        print(f"  Geometric coupling: {cumulative_release / cumulative_loading:.4f}")
    else:
        print(f"  WARNING: No events generated!")
    print("=" * 70)

    # Flush any remaining buffered snapshots
    hdf5_writer.flush()
    print(f"  Flushed remaining buffered snapshots")

    finalize_simulation(
        h5file,
        cumulative_loading,
        cumulative_release,
        config.coupling_history,
        m_current,
        slip_rate,
        min_moment_elem,
        max_moment_elem,
        min_release_elem,
        max_release_elem,
        min_deficit_elem,
        max_deficit_elem
    )
    h5file.close()
    print(f"\nHDF5 file closed: {Path(config.output_dir) / config.output_hdf5}")

    # Return minimal results dict (data already in HDF5)
    results = {
        "config": config,
        "mesh": mesh,
        "slip_rate": slip_rate,
        "event_history": event_history,
        "final_moment": m_current,
        "cumulative_loading": cumulative_loading,
        "cumulative_release": cumulative_release,
    }

    return results


def save_results(results, config):
    """
    Return HDF5 output path (data already written during simulation)
    """
    output_dir = Path(config.output_dir)
    output_path = output_dir / config.output_hdf5
    print(f"\nResults already saved to HDF5: {output_path}")
    return output_path
