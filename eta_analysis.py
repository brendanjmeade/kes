"""
Entropy-Normalized Transition Density (eta) Analysis for KES Ensembles

Computes eta, a metric that quantifies how sensitive earthquake nucleation
locations are to loading perturbations, normalized by the theoretical spatial
uncertainty from the MaxEnt probability distribution.

Usage:
    python eta_analysis.py [--input_dir DIR] [--n_events N] [--m_threshold M]
                          [--delta_threshold D]

Examples:
    # Analyze first 5 M>=6 events with 5 km transition threshold
    python eta_analysis.py

    # Analyze first 10 M>=7 events with 2 km transition threshold
    python eta_analysis.py --n_events 10 --m_threshold 7.0 --delta_threshold 2.0
"""

import argparse
from pathlib import Path

import h5py
import json
import matplotlib.pyplot as plt
import numpy as np

from spatial_prob import gamma_magnitude_dependent

FONTSIZE = 8


def conditional_entropy(m_i, gamma, magnitude=None):
    """
    Compute H(x|M) = -sum_i p(i|M) log p(i|M)

    Parameters
    ----------
    m_i : array (n_elements,)
        Moment at each mesh element
    gamma : float or callable
        MaxEnt exponent. If callable, gamma(magnitude) returns the value.
    magnitude : float, optional
        Event magnitude (required if gamma is callable)

    Returns
    -------
    H : float
        Conditional entropy in nats
    N_eff : float
        Effective number of sites = exp(H)
    """
    if callable(gamma):
        if magnitude is None:
            raise ValueError("magnitude required when gamma is callable")
        g = gamma(magnitude)
    else:
        g = gamma

    # Avoid numerical issues with zero or negative moment
    m_positive = np.maximum(m_i, 1e-10)

    # Unnormalized probability
    p_unnorm = m_positive ** g

    # Normalize
    p_sum = np.sum(p_unnorm)
    if p_sum < 1e-15:
        # Degenerate case: no moment anywhere
        return 0.0, 1.0

    p = p_unnorm / p_sum

    # Entropy (exclude zeros)
    mask = p > 1e-15
    H = -np.sum(p[mask] * np.log(p[mask]))

    N_eff = np.exp(H)

    return H, N_eff


def detect_transitions(hypocenters, x_pulse_values, delta_threshold):
    """
    Count transitions where hypocenter jumps by more than delta_threshold
    between adjacent pulse locations.

    Parameters
    ----------
    hypocenters : array (n_runs, 2)
        [x_hypo, z_hypo] for one event across all runs, sorted by x_pulse
    x_pulse_values : array (n_runs,)
        Pulse locations, sorted
    delta_threshold : float
        Minimum jump distance (km) to count as transition

    Returns
    -------
    n_transitions : int
        Number of transitions detected
    transition_locations : array
        x_pulse values where transitions occurred
    """
    if len(hypocenters) < 2:
        return 0, np.array([])

    # Compute distances between adjacent runs
    dx = np.diff(hypocenters[:, 0])
    dz = np.diff(hypocenters[:, 1])
    distances = np.sqrt(dx**2 + dz**2)

    # Find transitions
    transition_mask = distances > delta_threshold
    n_transitions = np.sum(transition_mask)

    # Midpoints of pulse intervals where transitions occurred
    transition_locations = 0.5 * (x_pulse_values[:-1][transition_mask] +
                                   x_pulse_values[1:][transition_mask])

    return n_transitions, transition_locations


def transition_density(n_transitions, x_pulse_range):
    """
    Transitions per unit pulse shift.

    Parameters
    ----------
    n_transitions : int
    x_pulse_range : float
        Total range of x_pulse values (max - min)

    Returns
    -------
    rho_T : float
        Transition density (transitions per km)
    """
    if x_pulse_range < 1e-10:
        return np.inf
    return n_transitions / x_pulse_range


def compute_eta(rho_T, H):
    """
    eta = rho_T / H

    Transitions per unit pulse shift, per unit entropy.

    Interpretation:
    - High eta: Many transitions relative to theoretical uncertainty.
              The probability landscape is rugged with many competing modes.
    - Low eta:  Few transitions despite high entropy.
              Broad basins in probability space; system is stable.
    """
    if H < 1e-10:
        return np.inf  # Degenerate case: all probability on one element
    return rho_T / H


def load_ensemble_data(input_dir, n_events=5, m_threshold=6.0):
    """
    Load ensemble data from HDF5 files.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing ensemble HDF5 files
    n_events : int
        Number of large events to track per simulation
    m_threshold : float
        Minimum magnitude threshold

    Returns
    -------
    ensemble_data : dict
        Contains 'x_pulse', 'events', 'moment_fields', 'config_params', 'mesh'
    """
    input_path = Path(input_dir)
    h5_files = sorted(input_path.glob("ensemble_run_*.h5"))

    if len(h5_files) == 0:
        raise FileNotFoundError(f"No ensemble files found in {input_path}")

    print(f"Loading {len(h5_files)} ensemble files...")

    ensemble_data = {
        'x_pulse': [],
        'events': [],  # List of event lists per run
        'moment_fields': [],  # List of moment field lists per run (one per event)
        'config_params': None,
        'mesh': None,
        'element_size_km': None,
    }

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            # Get pulse x-location from config
            moment_pulses_str = f['config'].attrs.get('moment_pulses', '[]')
            if isinstance(moment_pulses_str, str):
                moment_pulses = json.loads(moment_pulses_str)
            else:
                moment_pulses = []
            pulse_x = moment_pulses[0]["center_x_km"] if moment_pulses else 0
            ensemble_data['x_pulse'].append(pulse_x)

            # Get config parameters (only need once)
            if ensemble_data['config_params'] is None:
                ensemble_data['config_params'] = {
                    'gamma_min': f['config'].attrs.get('gamma_min', 0.5),
                    'gamma_max': f['config'].attrs.get('gamma_max', 1.5),
                    'alpha_spatial': f['config'].attrs.get('alpha_spatial', 0.35),
                    'M_min': f['config'].attrs.get('M_min', 5.0),
                }
                ensemble_data['element_size_km'] = f['config'].attrs.get('element_size_km', 1.0)

            # Get mesh (only need once)
            if ensemble_data['mesh'] is None:
                mesh_group = f['mesh']
                ensemble_data['mesh'] = {
                    'centroids': mesh_group['centroids'][:],
                    'x_coords': mesh_group['x_coords'][:],
                    'z_coords': mesh_group['z_coords'][:],
                    'n_along_strike': int(mesh_group.attrs['n_along_strike']),
                    'n_down_dip': int(mesh_group.attrs['n_down_dip']),
                }

            # Read events
            events = f['events'][:]

            # Read moment snapshots and times for moment field lookup
            moment_snapshots = f['moment_snapshots'][:]
            snapshot_times = f['times'][:]

            # Filter for large events
            large_mask = events['magnitude'] >= m_threshold
            large_events = events[large_mask][:n_events]

            # Extract event data and moment fields at event times
            run_events = []
            run_moment_fields = []

            for e in large_events:
                event_time = e['time']
                hypo_idx = e['hypocenter_idx']

                # Find snapshot closest to (but <= ) event time
                snapshot_idx = np.searchsorted(snapshot_times, event_time, side="right") - 1
                snapshot_idx = max(0, snapshot_idx)

                # Get moment field at this time
                moment_field = moment_snapshots[snapshot_idx, :]

                event_dict = {
                    'x_hypo': float(e['hypocenter_x_km']),
                    'z_hypo': float(e['hypocenter_z_km']),
                    'magnitude': float(e['magnitude']),
                    'element_idx': int(hypo_idx),
                    'time': float(event_time),
                    'gamma_used': float(e['gamma_used']),
                }
                run_events.append(event_dict)
                run_moment_fields.append(moment_field)

            ensemble_data['events'].append(run_events)
            ensemble_data['moment_fields'].append(run_moment_fields)

    ensemble_data['x_pulse'] = np.array(ensemble_data['x_pulse'])

    return ensemble_data


def analyze_ensemble(ensemble_data, delta_threshold=None):
    """
    Compute eta and related metrics for an ensemble of KES simulations.

    Parameters
    ----------
    ensemble_data : dict
        Output from load_ensemble_data()
    delta_threshold : float, optional
        Transition detection threshold in km. If None, uses element_size_km.

    Returns
    -------
    results : dict
        Contains per-event metrics
    """
    x_pulse = ensemble_data['x_pulse']
    sort_idx = np.argsort(x_pulse)
    x_pulse_sorted = x_pulse[sort_idx]
    x_pulse_range = x_pulse_sorted[-1] - x_pulse_sorted[0]

    n_runs = len(x_pulse)

    # Default threshold based on element size
    if delta_threshold is None:
        delta_threshold = ensemble_data['element_size_km'] * 2.0

    # Get config params for gamma calculation
    cfg = ensemble_data['config_params']

    # Determine number of events (may vary between runs due to filtering)
    min_events = min(len(run_events) for run_events in ensemble_data['events'])
    n_events = min_events

    print(f"Analyzing {n_events} events across {n_runs} runs")
    print(f"  Pulse range: {x_pulse_sorted[0]:.1f} - {x_pulse_sorted[-1]:.1f} km")
    print(f"  Transition threshold: {delta_threshold:.1f} km")

    results = {
        'event_idx': [],
        'magnitude': [],
        'H_mean': [],           # Mean entropy across runs
        'N_eff_mean': [],       # Mean effective sites
        'n_transitions': [],    # Transition count
        'rho_T': [],            # Transition density
        'eta': [],              # Normalized transition density
        'transition_locations': [],
        'hypocenters': [],      # Store hypocenters for plotting
        'x_pulse_sorted': x_pulse_sorted,
        'delta_threshold': delta_threshold,
    }

    for k in range(n_events):
        # Extract hypocenter positions for event k across all runs
        hypocenters = np.zeros((n_runs, 2))
        magnitudes = np.zeros(n_runs)
        H_values = []
        N_eff_values = []

        for i, run_idx in enumerate(sort_idx):
            # Skip if this run doesn't have event k
            if k >= len(ensemble_data['events'][run_idx]):
                hypocenters[i, :] = np.nan
                magnitudes[i] = np.nan
                continue

            event = ensemble_data['events'][run_idx][k]
            hypocenters[i, 0] = event['x_hypo']
            hypocenters[i, 1] = event['z_hypo']
            magnitudes[i] = event['magnitude']

            # Get moment field at time of this event
            moment_field = ensemble_data['moment_fields'][run_idx][k]

            # Use gamma_used from the event (already computed during simulation)
            gamma = event['gamma_used']

            # Compute entropy
            H, N_eff = conditional_entropy(moment_field, gamma)
            H_values.append(H)
            N_eff_values.append(N_eff)

        # Mean magnitude (should be nearly constant with fixed seed)
        M_mean = np.nanmean(magnitudes)

        # Mean entropy
        H_mean = np.mean(H_values) if H_values else 0.0
        N_eff_mean = np.mean(N_eff_values) if N_eff_values else 1.0

        # Transition detection (only on valid hypocenters)
        valid_mask = ~np.isnan(hypocenters[:, 0])
        valid_hypo = hypocenters[valid_mask]
        valid_pulse = x_pulse_sorted[valid_mask]

        n_trans, trans_locs = detect_transitions(valid_hypo, valid_pulse, delta_threshold)

        # Transition density
        rho_T = transition_density(n_trans, x_pulse_range)

        # eta
        eta = compute_eta(rho_T, H_mean)

        # Store results
        results['event_idx'].append(k)
        results['magnitude'].append(M_mean)
        results['H_mean'].append(H_mean)
        results['N_eff_mean'].append(N_eff_mean)
        results['n_transitions'].append(n_trans)
        results['rho_T'].append(rho_T)
        results['eta'].append(eta)
        results['transition_locations'].append(trans_locs)
        results['hypocenters'].append(hypocenters)

    # Convert to arrays
    for key in ['event_idx', 'magnitude', 'H_mean', 'N_eff_mean',
                'n_transitions', 'rho_T', 'eta']:
        results[key] = np.array(results[key])

    return results


def plot_eta_analysis(results, output_dir=None):
    """
    Generate diagnostic plots for eta analysis.

    Parameters
    ----------
    results : dict
        Output from analyze_ensemble()
    output_dir : str or Path, optional
        Directory to save plots

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. eta vs magnitude
    ax = axes[0, 0]
    valid_eta = np.isfinite(results['eta'])
    ax.scatter(results['magnitude'][valid_eta], results['eta'][valid_eta],
               s=60, c='steelblue', edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Magnitude', fontsize=FONTSIZE + 2)
    ax.set_ylabel('$\\eta$ (transitions per km per nat)', fontsize=FONTSIZE + 2)
    ax.set_title('Sensitivity vs Magnitude', fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3)

    # 2. Transition count vs effective sites
    ax = axes[0, 1]
    ax.scatter(results['N_eff_mean'], results['n_transitions'],
               s=60, c='coral', edgecolors='black', linewidths=0.5)
    max_val = max(max(results['N_eff_mean']), max(results['n_transitions'])) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
    ax.set_xlabel('$N_{eff}$ (effective sites)', fontsize=FONTSIZE + 2)
    ax.set_ylabel('Number of transitions', fontsize=FONTSIZE + 2)
    ax.set_title('Realized vs Theoretical Variability', fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3)

    # 3. Histogram of transition locations
    ax = axes[1, 0]
    all_transitions = np.concatenate(results['transition_locations'])
    if len(all_transitions) > 0:
        ax.hist(all_transitions, bins=20, edgecolor='black', color='purple', alpha=0.7)
    ax.set_xlabel('$x_{pulse}$ (km)', fontsize=FONTSIZE + 2)
    ax.set_ylabel('Transition count', fontsize=FONTSIZE + 2)
    ax.set_title('Where Do Transitions Occur?', fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3)

    # 4. eta distribution
    ax = axes[1, 1]
    finite_eta = results['eta'][np.isfinite(results['eta'])]
    if len(finite_eta) > 0:
        ax.hist(finite_eta, bins=15, edgecolor='black', color='green', alpha=0.7)
    ax.set_xlabel('$\\eta$', fontsize=FONTSIZE + 2)
    ax.set_ylabel('Count', fontsize=FONTSIZE + 2)
    ax.set_title('Distribution of $\\eta$ Across Events', fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is not None:
        output_path = Path(output_dir) / "eta_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_hypocenter_trajectories(results, output_dir=None):
    """
    Plot hypocenter trajectories for each event across pulse locations.

    Parameters
    ----------
    results : dict
        Output from analyze_ensemble()
    output_dir : str or Path, optional
        Directory to save plots

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_events = len(results['event_idx'])
    x_pulse = results['x_pulse_sorted']
    delta_threshold = results['delta_threshold']

    # Create figure with subplots for each event
    n_cols = min(3, n_events)
    n_rows = (n_events + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_events == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    colors = plt.cm.viridis(np.linspace(0, 1, len(x_pulse)))

    for k in range(n_events):
        row, col = k // n_cols, k % n_cols
        ax = axes[row, col]

        hypocenters = results['hypocenters'][k]

        # Plot trajectory
        valid_mask = ~np.isnan(hypocenters[:, 0])
        valid_hypo = hypocenters[valid_mask]
        valid_pulse = x_pulse[valid_mask]

        # Connect with lines
        ax.plot(valid_hypo[:, 0], valid_hypo[:, 1], '-', color='gray',
                linewidth=1, alpha=0.5, zorder=1)

        # Scatter points colored by pulse location
        scatter = ax.scatter(valid_hypo[:, 0], valid_hypo[:, 1],
                            c=valid_pulse, cmap='viridis', s=80,
                            edgecolors='black', linewidths=0.5, zorder=2)

        # Mark transitions
        n_trans = results['n_transitions'][k]

        ax.set_xlabel('$x$ (km)', fontsize=FONTSIZE)
        ax.set_ylabel('$z$ (km)', fontsize=FONTSIZE)
        ax.set_title(f"Event {k+1}: M={results['magnitude'][k]:.2f}, "
                    f"$\\eta$={results['eta'][k]:.3f}, n_trans={n_trans}",
                    fontsize=FONTSIZE)
        ax.tick_params(axis='both', labelsize=FONTSIZE - 1)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')

    # Hide empty subplots
    for k in range(n_events, n_rows * n_cols):
        row, col = k // n_cols, k % n_cols
        axes[row, col].set_visible(False)

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('$x_{pulse}$ (km)', fontsize=FONTSIZE)
    cbar.ax.tick_params(labelsize=FONTSIZE - 1)

    plt.suptitle('Hypocenter Trajectories with Pulse Location',
                fontsize=FONTSIZE + 3, y=1.02)

    if output_dir is not None:
        output_path = Path(output_dir) / "eta_trajectories.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def print_summary(results):
    """Print summary statistics for eta analysis."""

    print("\n" + "=" * 70)
    print("ETA ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nEvents analyzed: {len(results['event_idx'])}")
    print(f"Transition threshold: {results['delta_threshold']:.1f} km")

    print("\nPer-event results:")
    print("-" * 70)
    print(f"{'Event':>6} {'Mag':>6} {'H':>8} {'N_eff':>8} {'n_trans':>8} {'rho_T':>10} {'eta':>10}")
    print("-" * 70)

    for i in range(len(results['event_idx'])):
        eta_str = f"{results['eta'][i]:.4f}" if np.isfinite(results['eta'][i]) else "inf"
        print(f"{results['event_idx'][i]+1:>6} "
              f"{results['magnitude'][i]:>6.2f} "
              f"{results['H_mean'][i]:>8.3f} "
              f"{results['N_eff_mean'][i]:>8.1f} "
              f"{results['n_transitions'][i]:>8} "
              f"{results['rho_T'][i]:>10.4f} "
              f"{eta_str:>10}")

    print("-" * 70)

    # Summary statistics
    finite_eta = results['eta'][np.isfinite(results['eta'])]
    if len(finite_eta) > 0:
        print(f"\neta statistics:")
        print(f"  Mean: {np.mean(finite_eta):.4f}")
        print(f"  Median: {np.median(finite_eta):.4f}")
        print(f"  Std: {np.std(finite_eta):.4f}")
        print(f"  Range: {np.min(finite_eta):.4f} - {np.max(finite_eta):.4f}")

    print(f"\nTotal transitions detected: {np.sum(results['n_transitions'])}")
    print(f"Mean N_eff: {np.mean(results['N_eff_mean']):.1f}")
    print(f"Mean entropy H: {np.mean(results['H_mean']):.3f} nats")

    # Physical interpretation
    mean_eta = np.mean(finite_eta) if len(finite_eta) > 0 else 0
    print("\nInterpretation:")
    if mean_eta > 0.1:
        print("  HIGH eta (>0.1): Rugged probability landscape. Many competing")
        print("  nucleation sites exchange dominance with small loading changes.")
        print("  Spatial forecasts are sensitive to loading assumptions.")
    elif mean_eta > 0.01:
        print("  MEDIUM eta (0.01-0.1): Moderate sensitivity. Some discrete")
        print("  patches, but with identifiable basins.")
    else:
        print("  LOW eta (<0.01): Smooth landscape. Despite theoretical uncertainty,")
        print("  the system is stable. Refining loading estimates may not improve")
        print("  spatial forecasts.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute entropy-normalized transition density (eta) for KES ensembles"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/ensemble",
        help="Directory containing ensemble HDF5 files (default: results/ensemble)",
    )
    parser.add_argument(
        "--n_events",
        type=int,
        default=5,
        help="Number of large events to analyze per simulation (default: 5)",
    )
    parser.add_argument(
        "--m_threshold",
        type=float,
        default=6.0,
        help="Minimum magnitude threshold for events (default: 6.0)",
    )
    parser.add_argument(
        "--delta_threshold",
        type=float,
        default=None,
        help="Transition detection threshold in km (default: 2x element size)",
    )

    args = parser.parse_args()

    # Load data
    ensemble_data = load_ensemble_data(
        args.input_dir,
        n_events=args.n_events,
        m_threshold=args.m_threshold,
    )

    # Analyze
    results = analyze_ensemble(ensemble_data, delta_threshold=args.delta_threshold)

    # Print summary
    print_summary(results)

    # Generate plots
    plot_eta_analysis(results, output_dir=args.input_dir)
    plot_hypocenter_trajectories(results, output_dir=args.input_dir)

    plt.show()

    return results


if __name__ == "__main__":
    main()
