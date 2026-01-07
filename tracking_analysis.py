"""
Tracking-Normalized η Analysis for KES Ensembles

Decomposes location sensitivity (η) into two components:
1. Tracking: Events following the loading pulse (predictable)
2. Residual: "Address" uncertainty within the loading-defined region (irreducible)

Usage:
    python tracking_analysis.py [--input_dir DIR] [--n_events N] [--m_threshold M]
                                [--delta_threshold D]

Examples:
    python tracking_analysis.py
    python tracking_analysis.py --n_events 10 --m_threshold 7.0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from eta_analysis import (
    load_ensemble_data,
    analyze_ensemble,
    FONTSIZE,
)


def compute_tracking(x_hypo, z_hypo, x_pulse):
    """
    Regress hypocenter position against pulse location.

    x_hypo = alpha + beta * x_pulse + residual

    Parameters
    ----------
    x_hypo : array (n_runs,)
        Along-strike hypocenter positions
    z_hypo : array (n_runs,)
        Down-dip hypocenter positions
    x_pulse : array (n_runs,)
        Pulse locations

    Returns
    -------
    dict with regression results
    """
    # Sort by pulse location
    sort_idx = np.argsort(x_pulse)
    x_pulse_sorted = x_pulse[sort_idx]
    x_hypo_sorted = x_hypo[sort_idx]
    z_hypo_sorted = z_hypo[sort_idx]

    # Handle NaN values
    valid_x = ~np.isnan(x_hypo_sorted)
    valid_z = ~np.isnan(z_hypo_sorted)

    # Along-strike regression
    if np.sum(valid_x) >= 3:
        result_x = stats.linregress(x_pulse_sorted[valid_x], x_hypo_sorted[valid_x])
        beta_x, alpha_x, r_x = result_x.slope, result_x.intercept, result_x.rvalue
        residual_x = x_hypo_sorted.copy()
        residual_x[valid_x] = x_hypo_sorted[valid_x] - (alpha_x + beta_x * x_pulse_sorted[valid_x])
        residual_x[~valid_x] = np.nan
    else:
        beta_x, alpha_x, r_x = 0.0, 0.0, 0.0
        residual_x = x_hypo_sorted.copy()

    # Down-dip regression
    if np.sum(valid_z) >= 3:
        result_z = stats.linregress(x_pulse_sorted[valid_z], z_hypo_sorted[valid_z])
        beta_z, alpha_z, r_z = result_z.slope, result_z.intercept, result_z.rvalue
        residual_z = z_hypo_sorted.copy()
        residual_z[valid_z] = z_hypo_sorted[valid_z] - (alpha_z + beta_z * x_pulse_sorted[valid_z])
        residual_z[~valid_z] = np.nan
    else:
        beta_z, alpha_z, r_z = 0.0, 0.0, 0.0
        residual_z = z_hypo_sorted.copy()

    return {
        'beta_x': beta_x,
        'alpha_x': alpha_x,
        'beta_z': beta_z,
        'alpha_z': alpha_z,
        'r_squared_x': r_x**2,
        'r_squared_z': r_z**2,
        'residual_x': residual_x,
        'residual_z': residual_z,
        'x_pulse_sorted': x_pulse_sorted,
        'x_hypo_sorted': x_hypo_sorted,
        'z_hypo_sorted': z_hypo_sorted,
    }


def count_residual_transitions(residual_x, residual_z, x_pulse_sorted, delta_threshold=2.0):
    """
    Count transitions in the residual (detrended) coordinates.

    Parameters
    ----------
    residual_x : array (n_runs,)
        Residual along-strike positions
    residual_z : array (n_runs,)
        Residual down-dip positions
    x_pulse_sorted : array (n_runs,)
        Sorted pulse locations
    delta_threshold : float
        Minimum jump distance (km) to count as transition

    Returns
    -------
    n_transitions : int
    rho_T : float
        Transition density (transitions per km)
    """
    # Handle NaN values
    valid = ~np.isnan(residual_x) & ~np.isnan(residual_z)

    if np.sum(valid) < 2:
        return 0, 0.0

    # Get valid values in order
    res_x_valid = residual_x[valid]
    res_z_valid = residual_z[valid]
    pulse_valid = x_pulse_sorted[valid]

    # Compute distances between adjacent runs
    dx = np.diff(res_x_valid)
    dz = np.diff(res_z_valid)
    distances = np.sqrt(dx**2 + dz**2)

    n_transitions = np.sum(distances > delta_threshold)
    x_pulse_range = pulse_valid[-1] - pulse_valid[0]

    if x_pulse_range < 1e-10:
        return n_transitions, np.inf

    rho_T = n_transitions / x_pulse_range

    return n_transitions, rho_T


def compute_eta_residual(rho_T_residual, H):
    """η in the residual frame."""
    if H < 1e-10:
        return np.inf
    return rho_T_residual / H


def tracking_analysis(ensemble_data, eta_results, delta_threshold=2.0):
    """
    Full tracking analysis.

    Parameters
    ----------
    ensemble_data : dict
        Output from load_ensemble_data()
    eta_results : dict
        Output from analyze_ensemble()
    delta_threshold : float
        Transition threshold in km

    Returns
    -------
    results : dict
        Per-event tracking metrics
    """
    x_pulse = ensemble_data['x_pulse']
    n_events = len(eta_results['event_idx'])

    results = {
        'event_idx': [],
        'magnitude': [],
        'beta_x': [],
        'beta_z': [],
        'alpha_x': [],
        'alpha_z': [],
        'r_squared_x': [],
        'r_squared_z': [],
        'sigma_residual_x': [],
        'sigma_residual_z': [],
        'sigma_residual': [],
        'n_transitions_abs': [],
        'n_transitions_res': [],
        'eta_absolute': [],
        'eta_residual': [],
        'eta_reduction_frac': [],
        'H': [],
        'tracking_data': [],  # Store full tracking results for plotting
    }

    for k in range(n_events):
        # Extract hypocenters for event k across all runs
        x_hypo = np.array([
            run_events[k]['x_hypo'] if k < len(run_events) else np.nan
            for run_events in ensemble_data['events']
        ])
        z_hypo = np.array([
            run_events[k]['z_hypo'] if k < len(run_events) else np.nan
            for run_events in ensemble_data['events']
        ])
        mag = np.nanmean([
            run_events[k]['magnitude'] if k < len(run_events) else np.nan
            for run_events in ensemble_data['events']
        ])

        # Tracking regression
        track = compute_tracking(x_hypo, z_hypo, x_pulse)

        # Residual transitions
        n_trans_res, rho_T_res = count_residual_transitions(
            track['residual_x'],
            track['residual_z'],
            track['x_pulse_sorted'],
            delta_threshold
        )

        # Metrics from existing eta analysis
        H = eta_results['H_mean'][k]
        eta_abs = eta_results['eta'][k]
        n_trans_abs = eta_results['n_transitions'][k]

        # Compute residual eta
        eta_res = compute_eta_residual(rho_T_res, H)

        # Residual standard deviations
        valid_x = ~np.isnan(track['residual_x'])
        valid_z = ~np.isnan(track['residual_z'])
        sigma_res_x = np.std(track['residual_x'][valid_x]) if np.sum(valid_x) > 1 else 0.0
        sigma_res_z = np.std(track['residual_z'][valid_z]) if np.sum(valid_z) > 1 else 0.0
        sigma_res = np.sqrt(sigma_res_x**2 + sigma_res_z**2)

        # Reduction fraction
        if np.isfinite(eta_abs) and eta_abs > 0 and np.isfinite(eta_res):
            reduction = (eta_abs - eta_res) / eta_abs
        else:
            reduction = 0.0

        # Store
        results['event_idx'].append(k + 1)
        results['magnitude'].append(mag)
        results['beta_x'].append(track['beta_x'])
        results['beta_z'].append(track['beta_z'])
        results['alpha_x'].append(track['alpha_x'])
        results['alpha_z'].append(track['alpha_z'])
        results['r_squared_x'].append(track['r_squared_x'])
        results['r_squared_z'].append(track['r_squared_z'])
        results['sigma_residual_x'].append(sigma_res_x)
        results['sigma_residual_z'].append(sigma_res_z)
        results['sigma_residual'].append(sigma_res)
        results['n_transitions_abs'].append(n_trans_abs)
        results['n_transitions_res'].append(n_trans_res)
        results['eta_absolute'].append(eta_abs)
        results['eta_residual'].append(eta_res)
        results['eta_reduction_frac'].append(reduction)
        results['H'].append(H)
        results['tracking_data'].append(track)

    # Convert to arrays (except tracking_data which stays as list)
    for key in results:
        if key != 'tracking_data':
            results[key] = np.array(results[key])

    return results


def print_tracking_summary(results):
    """Print summary table for tracking analysis."""

    print("\n" + "=" * 80)
    print("TRACKING-NORMALIZED ETA ANALYSIS")
    print("=" * 80)

    print(f"\nEvents analyzed: {len(results['event_idx'])}")

    print("\nPer-event results:")
    print("-" * 80)
    print(f"{'Event':>6} {'Mag':>6} {'beta_x':>7} {'R^2_x':>6} {'sigma_res':>9} "
          f"{'eta_abs':>8} {'eta_res':>8} {'Reduction':>10}")
    print("-" * 80)

    for i in range(len(results['event_idx'])):
        eta_abs = results['eta_absolute'][i]
        eta_res = results['eta_residual'][i]
        reduction = results['eta_reduction_frac'][i]

        eta_abs_str = f"{eta_abs:.4f}" if np.isfinite(eta_abs) else "inf"
        eta_res_str = f"{eta_res:.4f}" if np.isfinite(eta_res) else "inf"
        reduction_str = f"{reduction*100:.1f}%" if np.isfinite(reduction) else "N/A"

        print(f"{results['event_idx'][i]:>6} "
              f"{results['magnitude'][i]:>6.2f} "
              f"{results['beta_x'][i]:>7.3f} "
              f"{results['r_squared_x'][i]:>6.2f} "
              f"{results['sigma_residual'][i]:>9.2f} "
              f"{eta_abs_str:>8} "
              f"{eta_res_str:>8} "
              f"{reduction_str:>10}")

    print("-" * 80)

    # Summary statistics
    valid_eta_abs = results['eta_absolute'][np.isfinite(results['eta_absolute'])]
    valid_eta_res = results['eta_residual'][np.isfinite(results['eta_residual'])]
    valid_reduction = results['eta_reduction_frac'][np.isfinite(results['eta_reduction_frac'])]

    print("\nSummary:")
    print(f"  Mean beta_x (tracking coefficient):  {np.mean(results['beta_x']):>7.3f}")
    print(f"  Mean beta_z (depth tracking):        {np.mean(results['beta_z']):>7.3f}")
    print(f"  Mean R^2_x (variance explained):     {np.mean(results['r_squared_x']):>7.2f}")
    print(f"  Mean sigma_residual (km):            {np.mean(results['sigma_residual']):>7.2f}")

    if len(valid_eta_abs) > 0:
        print(f"  Mean eta_absolute:                   {np.mean(valid_eta_abs):>7.4f}")
    if len(valid_eta_res) > 0:
        print(f"  Mean eta_residual:                   {np.mean(valid_eta_res):>7.4f}")
    if len(valid_reduction) > 0:
        print(f"  Mean reduction:                      {np.mean(valid_reduction)*100:>6.1f}%")

    # Interpretation
    mean_beta = np.mean(results['beta_x'])
    mean_r2 = np.mean(results['r_squared_x'])
    mean_reduction = np.mean(valid_reduction) if len(valid_reduction) > 0 else 0

    print("\nInterpretation:")
    if mean_beta > 0.5 and mean_r2 > 0.5:
        print(f"  Events strongly track the loading pulse (beta_x={mean_beta:.2f}, R^2={mean_r2:.2f}).")
        print(f"  ~{mean_reduction*100:.0f}% of apparent sensitivity is from tracking.")
        print(f"  Residual 'address' uncertainty is the irreducible component.")
    elif mean_beta > 0:
        print(f"  Events partially track the loading pulse (beta_x={mean_beta:.2f}).")
        print(f"  Significant unexplained scatter remains (R^2={mean_r2:.2f}).")
    else:
        print(f"  Events do not track the loading pulse (beta_x={mean_beta:.2f}).")
        print(f"  Sensitivity is intrinsic, not loading-dependent.")


def plot_tracking_analysis(results, output_dir=None):
    """
    Generate diagnostic plots for tracking analysis.

    Parameters
    ----------
    results : dict
        Output from tracking_analysis()
    output_dir : str or Path, optional
        Directory to save plots

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. eta_absolute vs eta_residual scatter
    ax = axes[0, 0]
    valid = np.isfinite(results['eta_absolute']) & np.isfinite(results['eta_residual'])
    ax.scatter(results['eta_absolute'][valid], results['eta_residual'][valid],
               s=80, c='steelblue', edgecolors='black', linewidths=0.5)

    # Add event labels
    for i in np.where(valid)[0]:
        ax.annotate(f"{results['event_idx'][i]:.0f}",
                   (results['eta_absolute'][i], results['eta_residual'][i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=FONTSIZE-1)

    max_eta = max(np.max(results['eta_absolute'][valid]),
                  np.max(results['eta_residual'][valid])) * 1.1
    ax.plot([0, max_eta], [0, max_eta], 'k--', alpha=0.5, label='1:1 (no reduction)')
    ax.set_xlabel('$\\eta_{absolute}$', fontsize=FONTSIZE + 2)
    ax.set_ylabel('$\\eta_{residual}$', fontsize=FONTSIZE + 2)
    ax.set_title('Tracking Reduces Sensitivity', fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_eta)
    ax.set_ylim(0, max_eta)

    # 2. beta_x distribution with histogram
    ax = axes[0, 1]
    ax.hist(results['beta_x'], bins=15, edgecolor='black', color='coral', alpha=0.7)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect tracking')
    ax.axvline(0.0, color='gray', linestyle=':', linewidth=2, label='No tracking')
    ax.axvline(np.mean(results['beta_x']), color='blue', linestyle='-', linewidth=2,
               label=f'Mean = {np.mean(results["beta_x"]):.2f}')
    ax.set_xlabel('$\\beta_x$ (tracking coefficient)', fontsize=FONTSIZE + 2)
    ax.set_ylabel('Count', fontsize=FONTSIZE + 2)
    ax.set_title('Along-Strike Tracking Strength', fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE - 1)
    ax.grid(True, alpha=0.3)

    # 3. R^2 histogram
    ax = axes[1, 0]
    ax.hist(results['r_squared_x'], bins=15, edgecolor='black', color='purple', alpha=0.7)
    ax.axvline(np.mean(results['r_squared_x']), color='blue', linestyle='-', linewidth=2,
               label=f'Mean = {np.mean(results["r_squared_x"]):.2f}')
    ax.set_xlabel('$R^2$ (variance explained)', fontsize=FONTSIZE + 2)
    ax.set_ylabel('Count', fontsize=FONTSIZE + 2)
    ax.set_title('Tracking Quality', fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # 4. Example trajectory with regression line (pick event with best R^2)
    ax = axes[1, 1]
    best_idx = np.argmax(results['r_squared_x'])
    track = results['tracking_data'][best_idx]

    # Plot data points
    valid = ~np.isnan(track['x_hypo_sorted'])
    ax.scatter(track['x_pulse_sorted'][valid], track['x_hypo_sorted'][valid],
               s=60, c='steelblue', edgecolors='black', linewidths=0.5,
               label='Hypocenters', zorder=2)

    # Plot regression line
    x_line = np.array([track['x_pulse_sorted'][valid].min(),
                       track['x_pulse_sorted'][valid].max()])
    y_line = track['alpha_x'] + track['beta_x'] * x_line
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f"$\\beta_x$={track['beta_x']:.2f}, R$^2$={results['r_squared_x'][best_idx]:.2f}",
            zorder=1)

    ax.set_xlabel('$x_{pulse}$ (km)', fontsize=FONTSIZE + 2)
    ax.set_ylabel('$x_{hypo}$ (km)', fontsize=FONTSIZE + 2)
    ax.set_title(f'Event {results["event_idx"][best_idx]:.0f}: Best Tracking Example',
                fontsize=FONTSIZE + 2)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is not None:
        output_path = Path(output_dir) / "tracking_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_residual_comparison(results, output_dir=None):
    """
    Plot original vs residual hypocenter positions for all events.

    Parameters
    ----------
    results : dict
        Output from tracking_analysis()
    output_dir : str or Path, optional
        Directory to save plots

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_events = len(results['event_idx'])
    n_cols = min(3, n_events)
    n_rows = (n_events + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_events == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for k in range(n_events):
        row, col = k // n_cols, k % n_cols
        ax = axes[row, col]

        track = results['tracking_data'][k]
        valid = ~np.isnan(track['residual_x']) & ~np.isnan(track['residual_z'])

        # Plot residual positions
        scatter = ax.scatter(track['residual_x'][valid], track['residual_z'][valid],
                            c=track['x_pulse_sorted'][valid], cmap='viridis', s=60,
                            edgecolors='black', linewidths=0.5, zorder=2)

        # Connect with lines
        ax.plot(track['residual_x'][valid], track['residual_z'][valid],
                '-', color='gray', linewidth=0.5, alpha=0.5, zorder=1)

        # Add origin reference
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        eta_res = results['eta_residual'][k]
        eta_res_str = f"{eta_res:.3f}" if np.isfinite(eta_res) else "inf"

        ax.set_xlabel('Residual $x$ (km)', fontsize=FONTSIZE)
        ax.set_ylabel('Residual $z$ (km)', fontsize=FONTSIZE)
        ax.set_title(f"Event {k+1}: $\\eta_{{res}}$={eta_res_str}, "
                    f"$\\sigma$={results['sigma_residual'][k]:.1f} km",
                    fontsize=FONTSIZE)
        ax.tick_params(axis='both', labelsize=FONTSIZE - 1)
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

    plt.suptitle('Residual Positions After Detrending',
                fontsize=FONTSIZE + 3, y=1.02)

    if output_dir is not None:
        output_path = Path(output_dir) / "tracking_residuals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def save_summary(results, output_dir):
    """Save tracking summary to text file."""
    output_path = Path(output_dir) / "tracking_summary.txt"

    with open(output_path, 'w') as f:
        f.write("TRACKING-NORMALIZED ETA ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Events analyzed: {len(results['event_idx'])}\n\n")

        f.write("Per-event results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Event':>6} {'Mag':>6} {'beta_x':>7} {'R^2_x':>6} "
                f"{'eta_abs':>8} {'eta_res':>8} {'Reduction':>10}\n")
        f.write("-" * 60 + "\n")

        for i in range(len(results['event_idx'])):
            eta_abs = results['eta_absolute'][i]
            eta_res = results['eta_residual'][i]
            reduction = results['eta_reduction_frac'][i]

            eta_abs_str = f"{eta_abs:.4f}" if np.isfinite(eta_abs) else "inf"
            eta_res_str = f"{eta_res:.4f}" if np.isfinite(eta_res) else "inf"
            reduction_str = f"{reduction*100:.1f}%" if np.isfinite(reduction) else "N/A"

            f.write(f"{results['event_idx'][i]:>6} "
                   f"{results['magnitude'][i]:>6.2f} "
                   f"{results['beta_x'][i]:>7.3f} "
                   f"{results['r_squared_x'][i]:>6.2f} "
                   f"{eta_abs_str:>8} "
                   f"{eta_res_str:>8} "
                   f"{reduction_str:>10}\n")

        f.write("-" * 60 + "\n\n")

        # Summary statistics
        valid_reduction = results['eta_reduction_frac'][np.isfinite(results['eta_reduction_frac'])]

        f.write("Summary statistics:\n")
        f.write(f"  Mean beta_x:       {np.mean(results['beta_x']):.3f}\n")
        f.write(f"  Mean R^2_x:        {np.mean(results['r_squared_x']):.2f}\n")
        f.write(f"  Mean eta_absolute: {np.mean(results['eta_absolute'][np.isfinite(results['eta_absolute'])]):.4f}\n")
        f.write(f"  Mean eta_residual: {np.mean(results['eta_residual'][np.isfinite(results['eta_residual'])]):.4f}\n")
        if len(valid_reduction) > 0:
            f.write(f"  Mean reduction:    {np.mean(valid_reduction)*100:.1f}%\n")

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tracking-normalized eta analysis for KES ensembles"
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

    # Load ensemble data
    print("Loading ensemble data...")
    ensemble_data = load_ensemble_data(
        args.input_dir,
        n_events=args.n_events,
        m_threshold=args.m_threshold,
    )

    # Run standard eta analysis first
    print("\nRunning standard eta analysis...")
    eta_results = analyze_ensemble(ensemble_data, delta_threshold=args.delta_threshold)

    # Run tracking analysis
    print("\nRunning tracking analysis...")
    delta_threshold = args.delta_threshold or ensemble_data['element_size_km'] * 2.0
    tracking_results = tracking_analysis(ensemble_data, eta_results, delta_threshold)

    # Print summary
    print_tracking_summary(tracking_results)

    # Generate plots
    plot_tracking_analysis(tracking_results, output_dir=args.input_dir)
    plot_residual_comparison(tracking_results, output_dir=args.input_dir)

    # Save summary to file
    save_summary(tracking_results, args.input_dir)

    return tracking_results


if __name__ == "__main__":
    main()
