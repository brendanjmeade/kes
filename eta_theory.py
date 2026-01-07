"""
Theoretical framework for η (entropy-normalized transition density)
with comparison to KES simulation results.

Theory:
    η = ρ_T / H

    where ρ_T is the transition density (transitions per km of pulse shift)
    and H is the conditional entropy of the MaxEnt distribution.

    The theoretical maximum occurs when transitions happen every pulse step:
        η_max = 1 / (Δx_pulse * H)

    A more complete model accounts for basin structure in probability space.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# =============================================================================
# Theoretical Models
# =============================================================================


def eta_maximum(delta_x: float, H: float) -> float:
    """
    Theoretical maximum η when transitions occur every pulse step.

    Parameters
    ----------
    delta_x : float
        Pulse step size (km)
    H : float
        Entropy (nats)

    Returns
    -------
    eta_max : float
    """
    return 1.0 / (delta_x * H)


def basin_size(L: float, N_eff: float, alpha: float = 1.0) -> float:
    """
    Estimate characteristic basin size.

    The basin size is the spatial scale over which one nucleation site
    dominates the probability distribution.

    Parameters
    ----------
    L : float
        Fault length (km)
    N_eff : float
        Effective number of sites = exp(H)
    alpha : float
        Scaling factor (default 1.0, can be fit to data)

    Returns
    -------
    ell_b : float
        Basin size (km)
    """
    return alpha * L / np.sqrt(N_eff)


def transition_density_model(delta_x: float, ell_b: float) -> float:
    """
    Theoretical transition density as function of perturbation scale.

    Model: ρ_T = (1/ℓ_b) * (1 - exp(-Δx/ℓ_b))

    Limits:
        Δx << ℓ_b: ρ_T ≈ Δx/ℓ_b² (rare transitions)
        Δx >> ℓ_b: ρ_T ≈ 1/ℓ_b (saturated)

    Parameters
    ----------
    delta_x : float
        Pulse step size (km)
    ell_b : float
        Basin size (km)

    Returns
    -------
    rho_T : float
        Transition density (transitions per km)
    """
    return (1.0 / ell_b) * (1.0 - np.exp(-delta_x / ell_b))


def eta_model(delta_x: float, H: float, ell_b: float) -> float:
    """
    Full theoretical model for η.

    Parameters
    ----------
    delta_x : float
        Pulse step size (km)
    H : float
        Entropy (nats)
    ell_b : float
        Basin size (km)

    Returns
    -------
    eta : float
    """
    rho_T = transition_density_model(delta_x, ell_b)
    return rho_T / H


def eta_vs_perturbation_scale(
    delta_x_array: np.ndarray, H: float, ell_b: float
) -> np.ndarray:
    """
    Compute η across a range of perturbation scales.
    """
    return np.array([eta_model(dx, H, ell_b) for dx in delta_x_array])


def eta_vs_entropy(
    H_array: np.ndarray, delta_x: float, L: float, alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute η and η_max across a range of entropies.

    Returns
    -------
    eta : array
        Model prediction
    eta_max : array
        Theoretical maximum
    """
    eta = np.zeros_like(H_array)
    eta_max = np.zeros_like(H_array)

    for i, H in enumerate(H_array):
        N_eff = np.exp(H)
        ell_b = basin_size(L, N_eff, alpha)
        eta[i] = eta_model(delta_x, H, ell_b)
        eta_max[i] = eta_maximum(delta_x, H)

    return eta, eta_max


# =============================================================================
# Visualization
# =============================================================================


def plot_eta_theory_comparison(
    simulation_results: Optional[dict] = None,
    L: float = 150.0,  # Fault length (km)
    delta_x: float = 1.0,  # Pulse step (km)
    H_range: Tuple[float, float] = (2.0, 10.0),
    alpha: float = 1.0,  # Basin size scaling factor
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Create comparison plots of theoretical η vs simulation results.

    Parameters
    ----------
    simulation_results : dict, optional
        Dictionary with keys: 'H', 'eta', 'magnitude', 'N_eff'
        Each value is an array of per-event results.
    L : float
        Fault length (km)
    delta_x : float
        Pulse step size used in simulations (km)
    H_range : tuple
        Range of entropy values for theoretical curves
    alpha : float
        Basin size scaling factor (fit to data)
    """

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    H_theory = np.linspace(H_range[0], H_range[1], 100)
    N_eff_theory = np.exp(H_theory)

    # =========================================================================
    # Panel 1: η vs Entropy
    # =========================================================================
    ax = axes[0, 0]

    # Theoretical curves
    eta_theory, eta_max_theory = eta_vs_entropy(H_theory, delta_x, L, alpha)

    ax.plot(
        H_theory,
        eta_max_theory,
        "k-0",
        lw=1,
        label=r"$\eta_{max}$",
    )
    # ax.plot(H_theory, eta_theory, "b-", lw=2, label=f"Model (L={L} km, α={alpha})")
    # ax.fill_between(H_theory, eta_theory, eta_max_theory, alpha=0.2, color="blue")

    # Simulation results
    if simulation_results is not None:
        ax.scatter(
            simulation_results["H"],
            simulation_results["eta"],
            c=simulation_results["magnitude"],
            cmap="plasma_r",
            s=5e-5 * 10 ** simulation_results["magnitude"],
            edgecolor=None,
            linewidths=0.5,
            zorder=5,
            alpha=0.5,
            label="numerical realization",
        )
        # cbar = plt.colorbar(ax.collections[0], ax=ax)
        # cbar.set_label("Magnitude")

    ax.set_xlabel("$H$ (nats)", fontsize=12)
    ax.set_ylabel(r"$\eta$ (transitions / km / nat)", fontsize=12)
    # ax.set_title("η vs Entropy", fontsize=14)
    # ax.legend(loc="upper right")
    ax.set_xlim([3, 9])
    ax.set_ylim([0, 0.4])
    ax.set_box_aspect(1)
    ax.grid(False)

    # =========================================================================
    # Panel 2: η vs Perturbation Scale
    # =========================================================================
    ax = axes[0, 1]

    # Use mean H from simulations or default
    if simulation_results is not None:
        H_mean = np.mean(simulation_results["H"])
        N_eff_mean = np.mean(simulation_results["N_eff"])
    else:
        H_mean = 8.3
        N_eff_mean = np.exp(H_mean)

    ell_b = basin_size(L, N_eff_mean, alpha)

    delta_x_array = np.logspace(-1, 2, 100)  # 0.1 to 100 km
    eta_vs_dx = eta_vs_perturbation_scale(delta_x_array, H_mean, ell_b)
    eta_max_vs_dx = 1.0 / (delta_x_array * H_mean)

    ax.loglog(delta_x_array, eta_max_vs_dx, "k--", lw=2, label=r"$\eta_{max}$")
    ax.loglog(
        delta_x_array, eta_vs_dx, "b-", lw=2, label=f"Model (ℓ_b = {ell_b:.2f} km)"
    )

    # Mark the simulation point
    if simulation_results is not None:
        eta_mean = np.mean(simulation_results["eta"])
        ax.scatter(
            [delta_x],
            [eta_mean],
            c="red",
            s=150,
            marker="*",
            zorder=5,
            label=f"Simulation (Δx={delta_x} km)",
        )

    # Mark characteristic scales
    ax.axvline(ell_b, color="green", ls=":", lw=2, alpha=0.7, label=f"Basin size ℓ_b")

    ax.set_xlabel(r"Perturbation scale $\Delta x$ (km)", fontsize=12)
    ax.set_ylabel(r"$\eta$ (transitions/km/nat)", fontsize=12)
    ax.set_title(f"η vs Perturbation Scale (H = {H_mean:.1f})", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim([0.1, 100])

    # =========================================================================
    # Panel 3: Efficiency η/η_max vs Entropy
    # =========================================================================
    ax = axes[1, 0]

    efficiency_theory = eta_theory / eta_max_theory

    ax.plot(H_theory, efficiency_theory, "b-", lw=2, label="Theory")
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)

    if simulation_results is not None:
        eta_max_sim = 1.0 / (delta_x * simulation_results["H"])
        efficiency_sim = simulation_results["eta"] / eta_max_sim
        ax.scatter(
            simulation_results["H"],
            efficiency_sim,
            c=simulation_results["magnitude"],
            cmap="viridis",
            s=100,
            edgecolor="black",
            zorder=5,
            label="Simulation",
        )

    ax.set_xlabel("Entropy H (nats)", fontsize=12)
    ax.set_ylabel(r"Efficiency $\eta / \eta_{max}$", fontsize=12)
    ax.set_title("How Close to Maximum Sensitivity?", fontsize=14)
    ax.set_ylim([0, 1.1])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4: Basin Size vs N_eff
    # =========================================================================
    ax = axes[1, 1]

    ell_b_theory = basin_size(L, N_eff_theory, alpha)

    ax.loglog(
        N_eff_theory,
        ell_b_theory,
        "b-",
        lw=2,
        label=rf"$\ell_b = \alpha L / \sqrt{{N_{{eff}}}}$",
    )

    # Physical reference lines
    ax.axhline(1.0, color="gray", ls=":", alpha=0.7, label="1 km (pulse step)")
    ax.axhline(L, color="gray", ls="--", alpha=0.7, label=f"{L} km (fault length)")

    if simulation_results is not None:
        ell_b_sim = basin_size(L, simulation_results["N_eff"], alpha)
        ax.scatter(
            simulation_results["N_eff"],
            ell_b_sim,
            c=simulation_results["magnitude"],
            cmap="viridis",
            s=100,
            edgecolor="black",
            zorder=5,
            label="Simulation",
        )

    ax.set_xlabel(r"$N_{eff}$ (effective sites)", fontsize=12)
    ax.set_ylabel(r"Basin size $\ell_b$ (km)", fontsize=12)
    ax.set_title("Characteristic Basin Size", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def plot_regime_diagram(
    L: float = 150.0, figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create a regime diagram showing where different η behaviors occur.

    Axes: Perturbation scale (Δx) vs Entropy (H)
    Coloring: η value
    """
    fig, ax = plt.subplots(figsize=figsize)

    H_array = np.linspace(2, 10, 50)
    delta_x_array = np.logspace(-1, 2, 50)

    H_grid, dx_grid = np.meshgrid(H_array, delta_x_array)
    eta_grid = np.zeros_like(H_grid)

    for i in range(len(delta_x_array)):
        for j in range(len(H_array)):
            N_eff = np.exp(H_array[j])
            ell_b = basin_size(L, N_eff)
            eta_grid[i, j] = eta_model(delta_x_array[i], H_array[j], ell_b)

    # Plot
    pcm = ax.pcolormesh(
        H_array, delta_x_array, eta_grid, shading="auto", cmap="viridis"
    )
    ax.set_yscale("log")

    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(r"$\eta$ (transitions/km/nat)", fontsize=12)

    # Contours
    contours = ax.contour(
        H_array,
        delta_x_array,
        eta_grid,
        levels=[0.01, 0.05, 0.1, 0.2],
        colors="white",
        linewidths=1.5,
    )
    ax.clabel(contours, inline=True, fontsize=10, fmt="%.2f")

    # Basin size = delta_x line (crossover)
    ell_b_line = basin_size(L, np.exp(H_array))
    ax.plot(H_array, ell_b_line, "r--", lw=2, label=r"$\Delta x = \ell_b$ (crossover)")

    ax.set_xlabel("Entropy H (nats)", fontsize=12)
    ax.set_ylabel(r"Perturbation scale $\Delta x$ (km)", fontsize=12)
    ax.set_title(f"η Regime Diagram (L = {L} km)", fontsize=14)
    ax.legend(loc="upper left")

    # Annotate regions
    ax.text(
        3,
        0.3,
        "Saturated\n(every step\nis a transition)",
        fontsize=11,
        ha="center",
        color="white",
    )
    ax.text(
        8, 30, "Sparse\n(rare transitions)", fontsize=11, ha="center", color="white"
    )

    plt.tight_layout()
    return fig


# =============================================================================
# Main: Example usage with synthetic or real data
# =============================================================================

if __name__ == "__main__":
    # Example simulation results (replace with actual data)
    # These match the M7+ results from the user's simulation
    simulation_results = {
        "H": np.array([8.396, 8.316, 7.972, 8.377, 8.320]),
        "eta": np.array([0.1191, 0.1052, 0.1098, 0.1144, 0.1152]),
        "magnitude": np.array([7.27, 7.43, 7.56, 7.66, 7.05]),
        "N_eff": np.array([4431.8, 4090.2, 2900.8, 4344.6, 4106.8]),
    }

    # Fault parameters (adjust to match your model)
    L = 150.0  # Fault length (km) - examine your geometry.py
    delta_x = 1.0  # Pulse step size (km)

    # Fit alpha to match observations
    # At saturation: eta ≈ 1/(ell_b * H) ≈ eta_obs
    # So: ell_b ≈ 1/(eta_obs * H)
    # And: alpha ≈ ell_b * sqrt(N_eff) / L
    H_mean = np.mean(simulation_results["H"])
    N_eff_mean = np.mean(simulation_results["N_eff"])
    eta_mean = np.mean(simulation_results["eta"])

    ell_b_empirical = 1.0 / (eta_mean * H_mean)
    alpha_fit = ell_b_empirical * np.sqrt(N_eff_mean) / L

    print(f"Empirical basin size: {ell_b_empirical:.2f} km")
    print(f"Fitted alpha: {alpha_fit:.3f}")
    print(f"Expected ell_b with alpha=1: {L / np.sqrt(N_eff_mean):.2f} km")

    # Create plots
    fig1 = plot_eta_theory_comparison(
        simulation_results=simulation_results, L=L, delta_x=delta_x, alpha=alpha_fit
    )
    fig1.savefig("eta_theory_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: eta_theory_comparison.png")

    fig2 = plot_regime_diagram(L=L)
    fig2.savefig("eta_regime_diagram.png", dpi=150, bbox_inches="tight")
    print("Saved: eta_regime_diagram.png")

    plt.show()
