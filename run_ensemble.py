"""
Ensemble simulation runner for earthquake simulator

Runs N simulations with different parameters to study sensitivity of
earthquake sequences to loading conditions and stochastic variability.

Usage:
    python run_ensemble.py [options]

Examples:
    # Vary only random seed (fixed loading) - shows natural variability
    python run_ensemble.py --vary_seed

    # Vary pulse amplitude (same seed) - isolates loading effect
    python run_ensemble.py --vary_amplitude

    # Vary pulse location (same seed)
    python run_ensemble.py --vary_location --delta 5.0

    # Vary both amplitude and seed
    python run_ensemble.py --vary_amplitude --vary_seed

Defaults:
    n_runs = 5
    delta = 2.0 (mm/yr for amplitude, km for location)
    vary_seed = False
    vary_amplitude = False
    vary_location = False
    output_dir = results/ensemble
"""

import argparse
import copy
from pathlib import Path

from config import Config
from simulator import run_simulation


def run_ensemble(
    n_runs=5,
    delta=2.0,
    output_dir="results/ensemble",
    vary_seed=False,
    vary_amplitude=False,
    vary_location=False,
    base_seed=42,
):
    """
    Run ensemble of simulations with varying parameters

    Parameters:
    -----------
    n_runs : int
        Number of simulations to run
    delta : float
        Change per run (mm/yr for amplitude, km for location)
    output_dir : str
        Directory for output files
    vary_seed : bool
        If True, use different random seed for each run
    vary_amplitude : bool
        If True, decrease pulse amplitude by delta each run
    vary_location : bool
        If True, shift pulse location by delta each run
    base_seed : int
        Base random seed (default: 42)

    Returns:
    --------
    output_files : list
        List of paths to output HDF5 files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_files = []

    # Get base values from default config
    base_config = Config()
    base_x = base_config.moment_pulses[0]["center_x_km"] if base_config.moment_pulses else 100.0
    base_amplitude = base_config.moment_pulses[0]["amplitude_mm_yr"] if base_config.moment_pulses else 20.0

    print("=" * 70)
    print("ENSEMBLE SIMULATION")
    print("=" * 70)
    print(f"Number of runs: {n_runs}")
    print(f"Delta: {delta}")
    print()
    print("Varying parameters:")
    if vary_seed:
        print(f"  Random seed: {base_seed}, {base_seed + 1}, ..., {base_seed + n_runs - 1}")
    else:
        print(f"  Random seed: {base_seed} (fixed)")
    if vary_amplitude:
        print(f"  Pulse amplitude: {base_amplitude}, {base_amplitude - delta}, ..., {base_amplitude - (n_runs-1)*delta} mm/yr")
    else:
        print(f"  Pulse amplitude: {base_amplitude} mm/yr (fixed)")
    if vary_location:
        print(f"  Pulse location: {base_x}, {base_x + delta}, ..., {base_x + (n_runs-1)*delta} km")
    else:
        print(f"  Pulse location: {base_x} km (fixed)")
    print()
    print(f"Output directory: {output_path}")
    print()

    for i in range(n_runs):
        # Create fresh config for this run
        config = Config()

        # IMPORTANT: Deep copy the moment_pulses list to avoid modifying class-level default
        config.moment_pulses = copy.deepcopy(Config.moment_pulses)

        # Apply parameter variations
        if vary_amplitude:
            amplitude = base_amplitude - i * delta
            config.moment_pulses[0]["amplitude_mm_yr"] = amplitude
        else:
            amplitude = base_amplitude

        if vary_location:
            pulse_x = base_x + i * delta
            config.moment_pulses[0]["center_x_km"] = pulse_x
        else:
            pulse_x = base_x

        if vary_seed:
            seed = base_seed + i
        else:
            seed = base_seed

        # Build descriptive filename
        parts = [f"ensemble_run_{i:02d}"]
        if vary_amplitude:
            parts.append(f"amp{amplitude:05.1f}")
        if vary_location:
            parts.append(f"x{pulse_x:05.1f}")
        if vary_seed:
            parts.append(f"seed{seed:03d}")
        output_filename = "_".join(parts) + ".h5"

        # Build parameter string for display
        param_parts = []
        if vary_amplitude:
            param_parts.append(f"amp={amplitude:.1f} mm/yr")
        if vary_location:
            param_parts.append(f"x={pulse_x:.1f} km")
        if vary_seed:
            param_parts.append(f"seed={seed}")
        if not param_parts:
            param_parts.append("(no variation)")
        param_str = ", ".join(param_parts)

        print("=" * 70)
        print(f"RUN {i + 1}/{n_runs}: {param_str}")
        print("=" * 70)

        output_filepath = output_path / output_filename

        # Check if file already exists - don't overwrite
        if output_filepath.exists():
            print(f"  WARNING: {output_filepath} already exists, skipping...")
            output_files.append(output_filepath)
            continue

        config.output_dir = str(output_path)
        config.output_hdf5 = output_filename
        config.random_seed = seed

        # Compute derived parameters
        config.compute_derived_parameters()

        print(f"  Pulse: x={config.moment_pulses[0]['center_x_km']:.1f} km, "
              f"amplitude={config.moment_pulses[0]['amplitude_mm_yr']:.1f} mm/yr")
        print(f"  Random seed: {config.random_seed}")
        print(f"  Output: {output_filepath}")
        print()

        # Run simulation
        results = run_simulation(config)

        output_files.append(output_filepath)

        print(f"\n  Completed run {i + 1}/{n_runs}")
        print(f"  Events: {len(results['event_history'])}")
        print()

    print("=" * 70)
    print("ENSEMBLE COMPLETE")
    print("=" * 70)
    print(f"Output files:")
    for f in output_files:
        print(f"  {f}")

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Run ensemble of earthquake simulations with varying parameters"
    )
    parser.add_argument(
        "--n_runs", type=int, default=5, help="Number of simulations (default: 5)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Change per run: mm/yr for amplitude, km for location (default: 2.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ensemble",
        help="Output directory (default: results/ensemble)",
    )
    parser.add_argument(
        "--vary_seed",
        action="store_true",
        help="Use different random seed for each run",
    )
    parser.add_argument(
        "--vary_amplitude",
        action="store_true",
        help="Decrease pulse amplitude by delta each run",
    )
    parser.add_argument(
        "--vary_location",
        action="store_true",
        help="Shift pulse location by delta each run",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )

    args = parser.parse_args()

    run_ensemble(
        n_runs=args.n_runs,
        delta=args.delta,
        output_dir=args.output_dir,
        vary_seed=args.vary_seed,
        vary_amplitude=args.vary_amplitude,
        vary_location=args.vary_location,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
