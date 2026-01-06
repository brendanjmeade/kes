"""
Loading-normalized η analysis.

Decomposes location uncertainty into:
1. Predictable tracking of loading (β, R²)
2. Residual "address" uncertainty (η_residual)

This separates "which neighborhood" (controlled by loading) from 
"which address within the neighborhood" (the irreducible uncertainty).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrackingResult:
    """Results for a single event's tracking analysis."""
    event_idx: int
    magnitude: float
    
    # Tracking metrics
    beta_x: float           # Along-strike tracking coefficient
    beta_z: float           # Down-dip tracking coefficient
    alpha_x: float          # Along-strike intercept
    alpha_z: float          # Down-dip intercept
    r_squared_x: float      # Along-strike R²
    r_squared_z: float      # Down-dip R²
    
    # Residual metrics
    sigma_residual_x: float # Along-strike residual std (km)
    sigma_residual_z: float # Down-dip residual std (km)
    sigma_residual_total: float  # Combined residual std
    
    # Transition analysis in residual frame
    n_transitions_residual: int
    rho_T_residual: float
    eta_residual: float
    
    # Original metrics for comparison
    n_transitions_absolute: int
    eta_absolute: float
    H: float
    N_eff: float


def compute_tracking_metrics(
    x_hypo: np.ndarray,
    z_hypo: np.ndarray, 
    x_pulse: np.ndarray
) -> Tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Compute tracking coefficients and residuals.
    
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
    beta_x, alpha_x, r_squared_x : along-strike regression results
    beta_z, alpha_z, r_squared_z : down-dip regression results
    residual_x, residual_z : residual arrays
    """
    # Along-strike regression
    slope_x, intercept_x, r_value_x, _, _ = stats.linregress(x_pulse, x_hypo)
    residual_x = x_hypo - (intercept_x + slope_x * x_pulse)
    
    # Down-dip regression
    slope_z, intercept_z, r_value_z, _, _ = stats.linregress(x_pulse, z_hypo)
    residual_z = z_hypo - (intercept_z + slope_z * x_pulse)
    
    return (slope_x, intercept_x, r_value_x**2,
            slope_z, intercept_z, r_value_z**2,
            residual_x, residual_z)


def detect_transitions_residual(
    residual_x: np.ndarray,
    residual_z: np.ndarray,
    x_pulse_sorted: np.ndarray,
    delta_threshold: float
) -> Tuple[int, np.ndarray]:
    """
    Detect transitions in residual coordinate frame.
    """
    dx = np.diff(residual_x)
    dz = np.diff(residual_z)
    distances = np.sqrt(dx**2 + dz**2)
    
    transition_mask = distances > delta_threshold
    n_transitions = np.sum(transition_mask)
    
    transition_locations = 0.5 * (x_pulse_sorted[:-1][transition_mask] + 
                                   x_pulse_sorted[1:][transition_mask])
    
    return n_transitions, transition_locations


def analyze_tracking(
    ensemble_data: Dict,
    entropy_data: Dict,
    delta_threshold: float = 2.0
) -> List[TrackingResult]:
    """
    Full tracking analysis for ensemble.
    
    Parameters
    ----------
    ensemble_data : dict
        'x_pulse': array (n_runs,)
        'events': list of n_runs event lists, each event has:
            'x_hypo', 'z_hypo', 'magnitude'
    entropy_data : dict
        'H': array (n_events,) - entropy per event
        'N_eff': array (n_events,)
        'eta': array (n_events,) - original eta values
        'n_transitions': array (n_events,)
    delta_threshold : float
        Transition detection threshold (km)
        
    Returns
    -------
    results : list of TrackingResult
    """
    x_pulse = np.array(ensemble_data['x_pulse'])
    sort_idx = np.argsort(x_pulse)
    x_pulse_sorted = x_pulse[sort_idx]
    x_pulse_range = x_pulse_sorted[-1] - x_pulse_sorted[0]
    
    n_runs = len(x_pulse)
    n_events = len(ensemble_data['events'][0])
    
    results = []
    
    for k in range(n_events):
        # Extract hypocenters
        x_hypo = np.array([ensemble_data['events'][sort_idx[i]][k]['x_hypo'] 
                          for i in range(n_runs)])
        z_hypo = np.array([ensemble_data['events'][sort_idx[i]][k]['z_hypo'] 
                          for i in range(n_runs)])
        magnitude = np.mean([ensemble_data['events'][sort_idx[i]][k]['magnitude'] 
                            for i in range(n_runs)])
        
        # Tracking regression
        (beta_x, alpha_x, r2_x,
         beta_z, alpha_z, r2_z,
         residual_x, residual_z) = compute_tracking_metrics(
            x_hypo, z_hypo, x_pulse_sorted
        )
        
        # Residual statistics
        sigma_x = np.std(residual_x)
        sigma_z = np.std(residual_z)
        sigma_total = np.sqrt(sigma_x**2 + sigma_z**2)
        
        # Transitions in residual frame
        n_trans_resid, _ = detect_transitions_residual(
            residual_x, residual_z, x_pulse_sorted, delta_threshold
        )
        rho_T_resid = n_trans_resid / x_pulse_range
        
        # Get entropy from provided data
        H = entropy_data['H'][k] if k < len(entropy_data['H']) else 8.0
        N_eff = entropy_data['N_eff'][k] if k < len(entropy_data['N_eff']) else 3000.0
        eta_abs = entropy_data['eta'][k] if k < len(entropy_data['eta']) else 0.1
        n_trans_abs = entropy_data['n_transitions'][k] if k < len(entropy_data['n_transitions']) else 20
        
        # Residual eta
        eta_resid = rho_T_resid / H if H > 0 else 0.0
        
        results.append(TrackingResult(
            event_idx=k + 1,
            magnitude=magnitude,
            beta_x=beta_x,
            beta_z=beta_z,
            alpha_x=alpha_x,
            alpha_z=alpha_z,
            r_squared_x=r2_x,
            r_squared_z=r2_z,
            sigma_residual_x=sigma_x,
            sigma_residual_z=sigma_z,
            sigma_residual_total=sigma_total,
            n_transitions_residual=n_trans_resid,
            rho_T_residual=rho_T_resid,
            eta_residual=eta_resid,
            n_transitions_absolute=int(n_trans_abs),
            eta_absolute=eta_abs,
            H=H,
            N_eff=N_eff
        ))
    
    return results


def print_tracking_summary(results: List[TrackingResult]):
    """Print summary table of tracking analysis."""
    
    print("=" * 90)
    print("TRACKING ANALYSIS SUMMARY")
    print("=" * 90)
    print()
    print("Per-event results:")
    print("-" * 90)
    print(f" {'Event':>5} {'Mag':>6} {'β_x':>7} {'R²_x':>6} {'β_z':>7} {'R²_z':>6} "
          f"{'σ_res':>7} {'η_abs':>7} {'η_res':>7} {'Δη':>7}")
    print("-" * 90)
    
    for r in results:
        delta_eta = r.eta_absolute - r.eta_residual
        print(f" {r.event_idx:>5} {r.magnitude:>6.2f} {r.beta_x:>7.3f} {r.r_squared_x:>6.3f} "
              f"{r.beta_z:>7.3f} {r.r_squared_z:>6.3f} {r.sigma_residual_total:>7.2f} "
              f"{r.eta_absolute:>7.4f} {r.eta_residual:>7.4f} {delta_eta:>7.4f}")
    
    print("-" * 90)
    
    # Summary statistics
    beta_x_mean = np.mean([r.beta_x for r in results])
    r2_x_mean = np.mean([r.r_squared_x for r in results])
    beta_z_mean = np.mean([r.beta_z for r in results])
    r2_z_mean = np.mean([r.r_squared_z for r in results])
    sigma_mean = np.mean([r.sigma_residual_total for r in results])
    eta_abs_mean = np.mean([r.eta_absolute for r in results])
    eta_res_mean = np.mean([r.eta_residual for r in results])
    
    print()
    print("Summary statistics:")
    print(f"  Mean β_x (along-strike tracking):  {beta_x_mean:.3f}")
    print(f"  Mean R²_x:                         {r2_x_mean:.3f}")
    print(f"  Mean β_z (down-dip tracking):      {beta_z_mean:.3f}")
    print(f"  Mean R²_z:                         {r2_z_mean:.3f}")
    print(f"  Mean residual σ:                   {sigma_mean:.2f} km")
    print()
    print(f"  Mean η (absolute):                 {eta_abs_mean:.4f}")
    print(f"  Mean η (residual):                 {eta_res_mean:.4f}")
    print(f"  Reduction:                         {100*(eta_abs_mean - eta_res_mean)/eta_abs_mean:.1f}%")
    print()
    
    # Interpretation
    print("Interpretation:")
    if beta_x_mean > 0.5:
        print(f"  STRONG along-strike tracking (β_x = {beta_x_mean:.2f}): ")
        print(f"    Events follow the loading pulse along-strike.")
    elif beta_x_mean > 0.1:
        print(f"  MODERATE along-strike tracking (β_x = {beta_x_mean:.2f})")
    else:
        print(f"  WEAK along-strike tracking (β_x = {beta_x_mean:.2f})")
    
    reduction = (eta_abs_mean - eta_res_mean) / eta_abs_mean if eta_abs_mean > 0 else 0
    if reduction > 0.5:
        print(f"  LARGE η reduction ({100*reduction:.0f}%): Most sensitivity is due to")
        print(f"    tracking the loading. Residual 'address' uncertainty is small.")
    elif reduction > 0.2:
        print(f"  MODERATE η reduction ({100*reduction:.0f}%): Loading explains some")
        print(f"    sensitivity, but significant 'address' uncertainty remains.")
    else:
        print(f"  SMALL η reduction ({100*reduction:.0f}%): Sensitivity is NOT primarily")
        print(f"    from tracking loading. 'Address' uncertainty dominates.")


def plot_tracking_analysis(
    results: List[TrackingResult],
    ensemble_data: Optional[Dict] = None,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """Generate diagnostic plots for tracking analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    magnitudes = [r.magnitude for r in results]
    beta_x = [r.beta_x for r in results]
    r2_x = [r.r_squared_x for r in results]
    sigma_res = [r.sigma_residual_total for r in results]
    eta_abs = [r.eta_absolute for r in results]
    eta_res = [r.eta_residual for r in results]
    
    # Panel 1: β_x vs magnitude
    ax = axes[0, 0]
    ax.scatter(magnitudes, beta_x, c=r2_x, cmap='viridis', s=80, edgecolor='black')
    ax.axhline(1.0, color='red', ls='--', alpha=0.7, label='Perfect tracking')
    ax.axhline(0.0, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('Magnitude', fontsize=12)
    ax.set_ylabel(r'$\beta_x$ (tracking coefficient)', fontsize=12)
    ax.set_title('Along-Strike Tracking vs Magnitude', fontsize=12)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('R²')
    ax.legend()
    
    # Panel 2: R² histogram
    ax = axes[0, 1]
    ax.hist(r2_x, bins=20, edgecolor='black', alpha=0.7, label='Along-strike')
    ax.hist([r.r_squared_z for r in results], bins=20, edgecolor='black', 
            alpha=0.5, label='Down-dip')
    ax.set_xlabel('R²', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Variance Explained by Loading', fontsize=12)
    ax.legend()
    ax.axvline(np.mean(r2_x), color='blue', ls='--', alpha=0.7)
    
    # Panel 3: Residual σ vs magnitude
    ax = axes[0, 2]
    ax.scatter(magnitudes, sigma_res, c=eta_res, cmap='plasma', s=80, edgecolor='black')
    ax.set_xlabel('Magnitude', fontsize=12)
    ax.set_ylabel(r'Residual $\sigma$ (km)', fontsize=12)
    ax.set_title('"Address" Uncertainty vs Magnitude', fontsize=12)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label(r'$\eta_{residual}$')
    
    # Panel 4: η absolute vs η residual
    ax = axes[1, 0]
    ax.scatter(eta_abs, eta_res, c=magnitudes, cmap='viridis', s=80, edgecolor='black')
    max_eta = max(max(eta_abs), max(eta_res)) * 1.1
    ax.plot([0, max_eta], [0, max_eta], 'k--', alpha=0.5, label='1:1')
    ax.set_xlabel(r'$\eta_{absolute}$', fontsize=12)
    ax.set_ylabel(r'$\eta_{residual}$', fontsize=12)
    ax.set_title('Absolute vs Residual Sensitivity', fontsize=12)
    ax.set_xlim([0, max_eta])
    ax.set_ylim([0, max_eta])
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Magnitude')
    ax.legend()
    
    # Panel 5: η reduction vs R²
    ax = axes[1, 1]
    eta_reduction = [(a - r) / a if a > 0 else 0 for a, r in zip(eta_abs, eta_res)]
    ax.scatter(r2_x, eta_reduction, c=magnitudes, cmap='viridis', s=80, edgecolor='black')
    ax.set_xlabel('R² (along-strike)', fontsize=12)
    ax.set_ylabel(r'$\eta$ reduction fraction', fontsize=12)
    ax.set_title('How Much Does Tracking Explain?', fontsize=12)
    ax.set_ylim([-0.1, 1.1])
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Magnitude')
    
    # Panel 6: Example trajectory with fit
    ax = axes[1, 2]
    if ensemble_data is not None:
        # Pick an event with moderate R²
        r2_values = [r.r_squared_x for r in results]
        example_idx = np.argmin(np.abs(np.array(r2_values) - np.median(r2_values)))
        r = results[example_idx]
        
        x_pulse = np.array(ensemble_data['x_pulse'])
        sort_idx = np.argsort(x_pulse)
        x_pulse_sorted = x_pulse[sort_idx]
        
        x_hypo = np.array([ensemble_data['events'][sort_idx[i]][example_idx]['x_hypo'] 
                          for i in range(len(x_pulse))])
        
        ax.scatter(x_pulse_sorted, x_hypo, c=range(len(x_pulse)), cmap='viridis', 
                   s=60, edgecolor='black', zorder=5)
        
        # Regression line
        x_fit = np.array([x_pulse_sorted.min(), x_pulse_sorted.max()])
        y_fit = r.alpha_x + r.beta_x * x_fit
        ax.plot(x_fit, y_fit, 'r-', lw=2, 
                label=f'β={r.beta_x:.2f}, R²={r.r_squared_x:.2f}')
        
        ax.set_xlabel(r'$x_{pulse}$ (km)', fontsize=12)
        ax.set_ylabel(r'$x_{hypo}$ (km)', fontsize=12)
        ax.set_title(f'Example: Event {r.event_idx} (M{r.magnitude:.1f})', fontsize=12)
        ax.legend()
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Run index')
    else:
        ax.text(0.5, 0.5, 'Ensemble data\nnot provided', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Example usage / testing
# =============================================================================

if __name__ == "__main__":
    # Example with synthetic data matching user's results
    np.random.seed(42)
    
    n_runs = 25
    n_events = 50
    x_pulse = np.linspace(100, 124, n_runs)
    
    # Simulate ensemble data
    ensemble_data = {
        'x_pulse': x_pulse,
        'events': []
    }
    
    # Create events with varying tracking behavior
    for i in range(n_runs):
        events = []
        for k in range(n_events):
            # Some events track well, others don't
            tracking_strength = 0.3 + 0.5 * np.random.random()
            noise = 5 + 10 * np.random.random()
            
            x_hypo = 50 + tracking_strength * (x_pulse[i] - 100) + np.random.normal(0, noise)
            z_hypo = 15 + 0.1 * (x_pulse[i] - 100) + np.random.normal(0, 3)
            mag = 6.0 + np.random.exponential(0.3)
            
            events.append({
                'x_hypo': x_hypo,
                'z_hypo': z_hypo,
                'magnitude': mag
            })
        ensemble_data['events'].append(events)
    
    # Simulate entropy data (matching user's γ_max = 8 results)
    entropy_data = {
        'H': 8.0 + 0.3 * np.random.randn(n_events),
        'N_eff': 3000 + 500 * np.random.randn(n_events),
        'eta': 0.10 + 0.02 * np.random.randn(n_events),
        'n_transitions': np.random.randint(15, 24, n_events)
    }
    
    # Run analysis
    results = analyze_tracking(ensemble_data, entropy_data, delta_threshold=2.0)
    
    # Print summary
    print_tracking_summary(results)
    
    # Create plots
    fig = plot_tracking_analysis(results, ensemble_data)
    fig.savefig('tracking_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved: tracking_analysis.png")
    
    plt.show()
