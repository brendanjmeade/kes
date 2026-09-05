"""
Prescribed vs. recovered Gutenberg-Richter and Omori statistics.

Reads a seed-varied ensemble and builds a two-panel validation figure:
(a) normalized cumulative magnitude distribution of the simulated
    catalogs (min-max envelope across runs, maximum-likelihood b-value
    fit) against the prescribed doubly truncated Gutenberg-Richter law,
    and
(b) the model event rate lambda(t) stacked over M >= m_threshold
    mainshocks (run min-max envelope, best-fit Omori model) against the
    prescribed Omori-Utsu kernel plus the estimated reference rate.

Prescribed parameters (b_value, M_min, M_max, omori_*) are read from the
first run's HDF5 config group (files written before full config persistence
fall back to the live Config class for class-level attributes).

The rate history and its grid come from the per-step 'step/' group when it
exists (lambda_total on the simulation timestep, so the Omori stack resolves
every lag even when snapshots are sparse); older files fall back to the
snapshot-grid 'times' / 'lambda_history'.

When the run used the deficit-tilted magnitude law (magnitude_tilt_mu != 0,
see magnitude_law.py) panel (a) also shows the prescribed law averaged over
the stored per-step tilt history, weighted by the expected event count of
each step (dashed, 'prescribed, time-averaged'); its Aki b-value on the
magnitude grid is reported in the caption. --origin loading restricts the
recovered catalog to loading-origin events (events['origin'] == 0) when the
Bath-split origin field exists.

Outputs:
    gr_omori.png                  Two-panel comparison figure (500 dpi)
    gr_omori_caption.txt          Figure caption with computed numbers
    gr_omori_classic.png          Variant with classic (untruncated) G-R
    gr_omori_classic_caption.txt  lines in panel (a), via --classic_gr

Usage:
    python plot_gr_omori.py [--input_dir results/ensemble] [--output PATH]
                            [--m_threshold 6.0] [--classic_gr]
                            [--origin all|loading]

Defaults:
    input_dir   = results/ensemble
    output      = <input_dir>/gr_omori.png (gr_omori_classic.png with
                  --classic_gr)
    m_threshold = 6.0
    origin      = all
"""

import argparse
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from hdf5_io import read_config
from magnitude_law import TiltedGR
from temporal_prob import omori_step_rate

FONTSIZE = 8


def load_ensemble_events(input_dir, origin="all"):
    """Load event times/magnitudes and the rate history (with its grid step) per run.

    The rate comes from 'step/lambda_total' on the per-step grid 'step/times'
    when the group exists, else from the snapshot-grid 'lambda_history'.
    origin='loading' keeps only loading-origin events when events['origin']
    exists (Bath split); the per-step tilt history (tilt_theta,
    expected_count) is carried along for the time-averaged prescribed law.
    """
    files = sorted(Path(input_dir).glob("ensemble_run_*.h5"))
    if not files:
        raise SystemExit(f"No ensemble_run_*.h5 files found in {input_dir}")

    runs = []
    duration = None
    dt_grid = None
    cfg = None
    for f in files:
        with h5py.File(f, "r") as h5:
            if cfg is None:
                cfg = read_config(h5)
            events = h5["events"][...]
            n_total = int(events.size)
            has_origin = "origin" in events.dtype.names
            if origin == "loading" and has_origin:
                events = events[events["origin"] == 0]
            theta = weight = None
            if "step" in h5:
                st = h5["step"]
                times = st["times"][:2]
                lam = st["lambda_total"][:]
                if "tilt_theta" in st:
                    theta = st["tilt_theta"][:]
                    weight = st["expected_count"][:]
            else:
                times = h5["times"][:2]
                lam = h5["lambda_history"][:]
            runs.append(
                {
                    "name": f.stem,
                    "time": events["time"].copy(),
                    "magnitude": events["magnitude"].copy(),
                    "lam": lam,
                    "theta": theta,
                    "weight": weight,
                    "n_total": n_total,
                    "has_origin": has_origin,
                }
            )
            duration = float(h5["config"].attrs["duration_years"])
            dt_grid = float(times[1] - times[0])
    return runs, duration, dt_grid, cfg


def prescribed_gr_survival(m, b, M_min, M_max):
    """Doubly truncated G-R survival function N(>=M)/N_tot on [M_min, M_max]."""
    denom = 10 ** (-b * M_min) - 10 ** (-b * M_max)
    return (10 ** (-b * m) - 10 ** (-b * M_max)) / denom


def classic_gr_survival(m, b, M_min):
    """Classic (untruncated) G-R survival function N(>=M)/N_tot."""
    return 10 ** (-b * (m - M_min))


def _truncated_b_from_mean(target, dR):
    """b of the doubly truncated G-R law whose mean excess magnitude is target."""
    lo, hi = 0.1, 5.0
    for _ in range(80):
        b = 0.5 * (lo + hi)
        beta = b * np.log(10)
        mean_model = 1 / beta - dR / np.expm1(beta * dR)
        if mean_model > target:
            lo = b
        else:
            hi = b
    return 0.5 * (lo + hi)


def fit_gr_b(mags, M_min, M_max):
    """Maximum-likelihood b-value for the doubly truncated G-R law."""
    m = mags[mags >= M_min] - M_min
    return _truncated_b_from_mean(m.mean(), M_max - M_min)


def compute_tilted_marginal(runs, cfg):
    """Time average of the prescribed deficit-tilted law (magnitude_law.TiltedGR).

    The per-step tilt multipliers theta(t) ('step/tilt_theta') are pooled
    over runs and averaged with weight lambda(t) dt ('step/expected_count'),
    giving the magnitude distribution the loading-origin draws follow over
    the whole ensemble; returned as the survival function N(>= M)/N on the
    law's magnitude grid together with its Aki and doubly truncated
    maximum-likelihood b-values. None when the tilt is off (mu = 0) or no
    per-step tilt history is stored.
    """
    if float(getattr(cfg, "magnitude_tilt_mu", 0.0)) == 0.0:
        return None
    with_theta = [r for r in runs if r["theta"] is not None]
    if not with_theta:
        return None
    theta = np.concatenate([r["theta"] for r in with_theta])
    weight = np.concatenate([r["weight"] for r in with_theta])
    ok = np.isfinite(theta) & np.isfinite(weight)
    if not ok.any() or weight[ok].sum() <= 0:
        return None
    if not hasattr(cfg, "element_area_m2"):
        cfg.compute_derived_parameters()
    D_ref = getattr(cfg, "deficit_reference", None)
    if D_ref is None:
        D_ref = float(cfg.deficit_reference_years) * float(cfg.geom_loading_rate_total)
    law = TiltedGR(cfg, D_ref)
    M, p = law.marginal(theta[ok], weight[ok])
    # Survival function by cumulative trapezoid from the top of the grid
    inc = 0.5 * (p[1:] + p[:-1]) * law.dM
    surv = np.empty_like(p)
    surv[-1] = 0.0
    surv[:-1] = np.cumsum(inc[::-1])[::-1]
    surv /= surv[0]
    mass = np.trapezoid(p, dx=law.dM)
    mean_M = np.trapezoid(M * p, dx=law.dM) / mass
    return {
        "grid": M,
        "surv": surv,
        "b_aki": np.log10(np.e) / (mean_M - law.M_min),
        "b_trunc": _truncated_b_from_mean(mean_M - law.M_min, law.M_max - law.M_min),
        "theta_mean": float(np.sum(theta[ok] * weight[ok]) / weight[ok].sum()),
        "n_steps": int(ok.sum()),
        "law": law,
    }


def compute_gr(runs, cfg, mag_grid):
    """Per-run normalized cumulative counts N(>=M)/N and ensemble envelope."""
    surv = np.zeros((len(runs), mag_grid.size))
    for i, run in enumerate(runs):
        m_sorted = np.sort(run["magnitude"])
        surv[i] = (m_sorted.size - np.searchsorted(m_sorted, mag_grid)) / m_sorted.size

    mags_pooled = np.concatenate([r["magnitude"] for r in runs])
    # Classic (untruncated) maximum-likelihood b-value (Aki 1965)
    m_above = mags_pooled[mags_pooled >= cfg.M_min]
    b_fit_classic = np.log10(np.e) / (m_above.mean() - cfg.M_min)
    return {
        "grid": mag_grid,
        "lo": surv.min(axis=0),
        "hi": surv.max(axis=0),
        "b_fit": fit_gr_b(mags_pooled, cfg.M_min, cfg.M_max),
        "b_fit_classic": b_fit_classic,
        "n_pooled": mags_pooled.size,
        "n_below_mmin": int(np.sum(mags_pooled < cfg.M_min)),
    }


def compute_omori_stack(runs, dt_grid, cfg, m_threshold):
    """Superposed-epoch stack of the model's internal rate lambda(t).

    Event times are quantized to multiples of dt_grid, so the observable
    lags after each mainshock are k*dt_grid; per run, lambda is averaged
    over all M >= m_threshold mainshocks at each lag, and the run-to-run
    spread provides the uncertainty envelope.
    """
    k_max = int(np.floor(cfg.omori_duration_years / dt_grid))
    centers = np.arange(1, k_max + 1) * dt_grid

    lam_stacks = np.zeros((len(runs), k_max))
    K_mainshocks = []
    n_mainshocks = 0
    for i, run in enumerate(runs):
        t, m = run["time"], run["magnitude"]
        mains = t[m >= m_threshold]
        K_mainshocks.append(
            cfg.omori_K_ref
            * 10 ** (cfg.omori_alpha * (m[m >= m_threshold] - cfg.omori_M_ref))
        )
        n_mainshocks += mains.size
        idx_ms = np.round(mains / dt_grid).astype(int)
        idx = idx_ms[:, None] + np.arange(1, k_max + 1)[None, :]
        for j in range(k_max):
            valid = idx[:, j][idx[:, j] < run["lam"].size]
            lam_stacks[i, j] = run["lam"][valid].mean() if valid.size else np.nan

    K_all = np.concatenate(K_mainshocks)
    K_mean = K_all.mean() if K_all.size else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN lags (no mainshocks)
        lam_mean = np.nanmean(lam_stacks, axis=0)
        lam_lo = np.nanmin(lam_stacks, axis=0)
        lam_hi = np.nanmax(lam_stacks, axis=0)

    # Prescribed kernel on the grid in the run's own Omori mode (legacy point
    # sample or step-integrated), using the simulation timestep
    dt_step = float(getattr(cfg, "time_step_years", dt_grid))
    kernel = np.array(
        [omori_step_rate(K_mean, k, dt_step, cfg) for k in range(1, k_max + 1)]
    )

    # Baseline from the long-lag stack, corrected for the prescribed tail
    tail = centers >= 15.0
    baseline = lam_mean[tail].mean() - kernel[tail].mean()

    # Branching-ratio diagnostic (all events trigger in the model)
    mags_all = np.concatenate([r["magnitude"] for r in runs])
    K_all_mean = np.mean(
        cfg.omori_K_ref * 10 ** (cfg.omori_alpha * (mags_all - cfg.omori_M_ref))
    )
    n_branch = np.sum(
        [omori_step_rate(K_all_mean, k, dt_step, cfg) * dt_step for k in range(1, k_max + 1)]
    )

    t_smooth = np.geomspace(dt_grid, cfg.omori_duration_years, 200)
    if getattr(cfg, "omori_integrate_over_step", False):
        # Step-integrated mode has no smooth instantaneous curve on the grid;
        # show the prescribed per-step expectation at the grid lags instead
        t_smooth = centers
        prescribed_smooth = baseline + kernel
    else:
        prescribed_smooth = (
            baseline + K_mean / (t_smooth + cfg.omori_c_years) ** cfg.omori_p
        )

    # Best-fit Omori model to the ensemble-mean stack (c unresolvable, fixed)
    c0 = cfg.omori_c_years

    def omori_model(t, K, p, B):
        return B + K / (t + c0) ** p

    fit_ok = np.isfinite(lam_mean)
    if fit_ok.sum() >= 3:
        try:
            (K_fit, p_fit, B_fit), _ = curve_fit(
                omori_model, centers[fit_ok], lam_mean[fit_ok], p0=[0.3, 1.0, 0.6],
                bounds=([0.0, 0.1, 0.0], [10.0, 3.0, 5.0]),
            )
        except (RuntimeError, ValueError):
            K_fit = p_fit = B_fit = np.nan
    else:
        # Too few mainshocks (short smoke runs): no fit
        K_fit = p_fit = B_fit = np.nan
    return {
        "centers": centers,
        "lam_mean": lam_mean,
        "lam_lo": lam_lo,
        "lam_hi": lam_hi,
        "baseline": baseline,
        "t_smooth": t_smooth,
        "prescribed_smooth": prescribed_smooth,
        "fit_smooth": omori_model(t_smooth, K_fit, p_fit, B_fit),
        "K_fit": K_fit,
        "p_fit": p_fit,
        "B_fit": B_fit,
        "n_mainshocks": n_mainshocks,
        "K_mean": K_mean,
        "n_branch": n_branch,
    }


def _nice_limits(vmin, vmax, n_intervals=3):
    """Round axis limits and ticks (about n_intervals steps) enclosing [vmin, vmax]."""
    span = max(vmax - vmin, 1e-6)
    raw = span / n_intervals
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    if hi <= lo:
        hi = lo + step
    ticks = np.arange(lo, hi + 0.5 * step, step)
    return float(lo), float(hi), [float(t) for t in ticks]


def make_figure(gr, om, cfg, output, classic_gr=False, tilt=None):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8, 4))
    cool = "#1f77b4"
    warm = "#ff7f0e"

    # Panel a: Gutenberg-Richter
    floor = 1e-3
    ax_a.fill_between(
        gr["grid"],
        np.maximum(gr["lo"], floor),
        np.maximum(gr["hi"], floor),
        color=cool,
        alpha=0.25,
        linewidth=0,
        label="min–max envelope",
    )
    m_dense = np.arange(cfg.M_min, cfg.M_max + 1e-9, 0.005)
    if classic_gr:
        fit_line = classic_gr_survival(m_dense, gr["b_fit_classic"], cfg.M_min)
        prescribed_line = classic_gr_survival(m_dense, cfg.b_value, cfg.M_min)
        b_fit_label = gr["b_fit_classic"]
    else:
        fit_line = prescribed_gr_survival(m_dense, gr["b_fit"], cfg.M_min,
                                          cfg.M_max)
        prescribed_line = prescribed_gr_survival(m_dense, cfg.b_value,
                                                 cfg.M_min, cfg.M_max)
        b_fit_label = gr["b_fit"]
    ax_a.plot(m_dense, fit_line, "-", color=cool, linewidth=1.0,
              label=f"best fit ($b$ = {b_fit_label:.2f})")
    ax_a.plot(m_dense, prescribed_line, "-", color=warm, linewidth=1.0,
              label=f"prescribed G-R ($b$ = {cfg.b_value:.2f})")
    if tilt is not None:
        ax_a.plot(tilt["grid"], np.maximum(tilt["surv"], floor), "--", color=warm,
                  linewidth=1.0, label="prescribed, time-averaged")
    ax_a.set_yscale("log")
    ax_a.set_xlim(5.0, 8.0)
    ax_a.set_ylim(floor, 1.0)
    ax_a.set_xticks([5, 6, 7, 8])
    ax_a.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
    ax_a.set_xlabel("$M$", fontsize=FONTSIZE)
    ax_a.set_ylabel("normalized count", fontsize=FONTSIZE)
    ax_a.legend(loc="lower left", fontsize=FONTSIZE - 2, frameon=False)

    # Panel b: Omori stack
    ax_b.fill_between(
        om["centers"],
        om["lam_lo"],
        om["lam_hi"],
        color=cool,
        alpha=0.25,
        linewidth=0,
        label="min–max envelope",
    )
    ax_b.plot(om["t_smooth"], om["fit_smooth"], "-", color=cool, linewidth=1.0,
              label=f"best fit ($p$ = {om['p_fit']:.2f})")
    ax_b.plot(om["t_smooth"], om["prescribed_smooth"], "-", color=warm,
              linewidth=1.0, label=f"prescribed ($p$ = {cfg.omori_p:.2f})")
    ax_b.set_xscale("log")
    ax_b.set_xlim(1.0, 20.0)
    ax_b.set_xticks([1, 3, 10, 20])
    ax_b.set_xticklabels(["1", "3", "10", "20"])
    # Round y-limits enclosing the stack and the prescribed curve
    shown = om["centers"] <= 20.0
    y_all = np.concatenate([om["lam_lo"][shown], om["lam_hi"][shown],
                            np.asarray(om["prescribed_smooth"])[np.asarray(om["t_smooth"]) <= 20.0]])
    y_all = y_all[np.isfinite(y_all)]
    y_lo, y_hi, y_ticks = _nice_limits(y_all.min(), y_all.max()) if y_all.size else (0.3, 1.2, [0.3, 0.6, 0.9, 1.2])
    ax_b.set_ylim(y_lo, y_hi)
    ax_b.set_yticks(y_ticks)
    ax_b.set_xlabel("$t$ (years)", fontsize=FONTSIZE)
    ax_b.set_ylabel("events per year", fontsize=FONTSIZE)
    ax_b.legend(loc="lower left", fontsize=FONTSIZE - 2, frameon=False)

    for ax, letter in [(ax_a, "a"), (ax_b, "b")]:
        ax.set_box_aspect(1)
        ax.tick_params(axis="both", labelsize=FONTSIZE, direction="out",
                       length=3, width=0.8)
        ax.text(0.95, 0.97, letter, transform=ax.transAxes, ha="right",
                va="top", fontsize=FONTSIZE + 1)

    plt.savefig(output, dpi=500, bbox_inches="tight")
    print(f"Saved: {output}")
    return fig


def tilt_caption(tilt, cfg, origin="all"):
    """Caption sentence for the time-averaged prescribed tilted law."""
    if tilt is None:
        return ""
    law = tilt["law"]
    text = (
        "The dashed orange curve is the prescribed deficit-tilted law "
        f"(magnitude_law.TiltedGR: shape {law.shape}, target {law.target}, "
        f"mu = {law.mu:g}, D_w = {law.D_w / law.D_ref:.2f} D_ref) averaged over "
        "the stored per-step tilt multipliers theta(t) weighted by the expected "
        f"event count lambda(t) dt of each step (pooled over {tilt['n_steps']} "
        f"steps; mean theta = {tilt['theta_mean']:.3f}), i.e. the time-averaged "
        "magnitude distribution the loading-origin draws follow; its Aki "
        f"b-value on the magnitude grid is b = {tilt['b_aki']:.3f} (doubly "
        f"truncated maximum-likelihood b = {tilt['b_trunc']:.3f}). "
    )
    if getattr(cfg, "omori_split_enabled", False):
        text += (
            "Omori-origin events draw from the plain G-R law capped at "
            f"M_parent - {cfg.omori_bath_dM:g} (Bath split)"
            + (", which the overlay does not include. " if origin == "all"
               else " and are excluded from the recovered catalog here. ")
        )
    return text


def build_caption(gr, om, cfg, m_threshold, n_runs, duration, dt_grid,
                  classic=False, tilt=None, origin="all"):
    c_days = cfg.omori_c_years * 365.25
    pooled_rate = gr["n_pooled"] / (n_runs * duration)
    return (
        "Prescribed versus recovered earthquake statistics for a "
        f"{n_runs}-member simulation ensemble (identical configuration, "
        f"varied random seed; {duration:.0f} yr each; "
        f"{gr['n_pooled']} events pooled). "
        + (
            "(a) Normalized cumulative magnitude distribution N(>= M)/N. "
            f"The shaded band is the min-max envelope across the {n_runs} "
            "runs, the blue line is the classic (untruncated) "
            "Gutenberg-Richter maximum-likelihood fit to the pooled "
            f"recovered catalog (b = {gr['b_fit_classic']:.2f}; Aki 1965), "
            "and the orange line is the prescribed classic relation "
            f"N(>= M)/N = 10^(-b(M - {cfg.M_min:g})) with b = "
            f"{cfg.b_value:.2f}, both straight on these axes. The "
            "recovered distributions fall below the straight lines "
            f"approaching M = {cfg.M_max:g} because magnitudes are drawn "
            f"from a distribution truncated at M_max = {cfg.M_max:g} and "
            "because catalog magnitudes, recomputed from the moment "
            "actually released, deplete where accumulated moment is "
            "scarce. "
            if classic
            else
            "(a) Normalized cumulative magnitude distribution N(>= M)/N. "
            f"The shaded band is the min-max envelope across the {n_runs} "
            "runs, the blue curve is the maximum-likelihood doubly "
            "truncated Gutenberg-Richter fit to the pooled recovered "
            f"catalog (b = {gr['b_fit']:.2f}), and the orange curve is "
            f"the prescribed doubly truncated distribution (b = "
            f"{cfg.b_value:.2f}, {cfg.M_min:g} <= M <= {cfg.M_max:g}); "
            "the downward curvature approaching M_max is the signature of "
            "the upper truncation. Catalog magnitudes are recomputed from "
            "the moment actually released, so where accumulated moment is "
            "scarce the largest magnitudes deplete before M_max. "
        )
        + tilt_caption(tilt, cfg, origin)
        + (
            "Only loading-origin events (events['origin'] == 0) enter the "
            "recovered catalog. "
            if origin == "loading" else ""
        )
        +
        "(b) Superposed-epoch aftershock rate: the model event rate "
        "lambda(t) per mainshock, averaged over the observable lags "
        f"following each of the {om['n_mainshocks']} M >= {m_threshold:g} "
        "mainshocks and stacked per run (lags of 1-20 yr shown). The "
        f"shaded band is the min-max envelope across the {n_runs} runs, "
        "the blue curve is the best-fitting Omori-Utsu model "
        f"B + K/(t + c)^p to the ensemble-mean stack (K = "
        f"{om['K_fit']:.2f} yr^-1, p = {om['p_fit']:.2f}, B = "
        f"{om['B_fit']:.2f} yr^-1, with c fixed at {c_days:.0f} day), and "
        "the orange curve is the prescribed direct Omori-Utsu kernel "
        f"K/(t + c)^p (p = {cfg.omori_p:.2f}) averaged over the actual "
        f"mainshock magnitudes (mean K = {om['K_mean']:.2f} yr^-1, with "
        f"K = K_ref 10^(alpha(M - M_ref)), K_ref = {cfg.omori_K_ref:g} "
        f"yr^-1, M_ref = {cfg.omori_M_ref:g}, alpha = {cfg.omori_alpha:g}, "
        f"c = {c_days:.0f} day, {cfg.omori_duration_years:g}-yr window), "
        f"added to a reference rate ({om['baseline']:.2f} yr^-1) estimated "
        "from the 15-30 yr stack "
        "after removing the prescribed tail. The reference rate exceeds "
        f"the unconditional mean rate ({pooled_rate:.2f} yr^-1) because "
        "mainshocks occur preferentially within multi-decade high-activity "
        "epochs and because secondary triggering (branching ratio n ~ "
        f"{om['n_branch']:.2f}) elevates the stack; the plotted prescribed "
        "kernel is the direct (first-generation) term only. Event times "
        f"are quantized to the {dt_grid:.2f}-yr output grid, so sub-annual "
        f"decay and the c = {c_days:.0f} day rolloff are unresolvable; the "
        "first plotted lag is the shortest observable."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prescribed vs. recovered G-R and Omori statistics"
    )
    parser.add_argument("--input_dir", default="results/ensemble",
                        help="Directory containing ensemble_run_*.h5 files")
    parser.add_argument("--output", default=None,
                        help="Output figure path (default <input_dir>/gr_omori.png)")
    parser.add_argument("--m_threshold", type=float, default=6.0,
                        help="Mainshock magnitude threshold for the Omori stack")
    parser.add_argument("--classic_gr", action="store_true",
                        help="Plot classic (untruncated) G-R lines in panel (a)")
    parser.add_argument("--origin", choices=["all", "loading"], default="all",
                        help="Recovered catalog: all events or loading-origin only "
                             "(needs events['origin'], Bath split)")
    args = parser.parse_args()

    default_name = "gr_omori_classic.png" if args.classic_gr else "gr_omori.png"
    output = Path(args.output) if args.output else Path(args.input_dir) / default_name
    runs, duration, dt_grid, cfg = load_ensemble_events(args.input_dir, args.origin)
    mag_grid = np.arange(4.8, 8.0 + 1e-9, 0.02)
    gr = compute_gr(runs, cfg, mag_grid)
    om = compute_omori_stack(runs, dt_grid, cfg, args.m_threshold)
    tilt = compute_tilted_marginal(runs, cfg)

    print(f"Runs: {len(runs)}, duration {duration:.0f} yr, grid dt {dt_grid:.4f} yr "
          f"({runs[0]['lam'].size} rate samples per run)")
    n_total = sum(r["n_total"] for r in runs)
    print(f"Pooled events: {gr['n_pooled']} ({gr['n_below_mmin']} below M_min)"
          + (f"; origin = {args.origin}: {gr['n_pooled']} of {n_total} events kept"
             if args.origin != "all" else ""))
    if args.origin == "loading" and not any(r["has_origin"] for r in runs):
        print("  (no events['origin'] field: all events kept)")
    print(f"G-R fit: b = {gr['b_fit']:.3f} truncated, "
          f"{gr['b_fit_classic']:.3f} classic (prescribed {cfg.b_value:g})")
    if tilt is not None:
        print(f"Prescribed time-averaged tilted law: b(Aki) = {tilt['b_aki']:.3f}, "
              f"b(truncated ML) = {tilt['b_trunc']:.3f}; mean theta = {tilt['theta_mean']:.3f} "
              f"over {tilt['n_steps']} steps (mu = {tilt['law'].mu:g}, shape = {tilt['law'].shape}, "
              f"target = {tilt['law'].target})")
    print(f"Mainshocks M>={args.m_threshold:g}: {om['n_mainshocks']}")
    print(f"Omori fit: K = {om['K_fit']:.3f} /yr, p = {om['p_fit']:.3f}, "
          f"B = {om['B_fit']:.3f} /yr "
          f"(prescribed mean K = {om['K_mean']:.3f}, p = {cfg.omori_p:g}, "
          f"baseline = {om['baseline']:.3f}); "
          f"branching ratio n: {om['n_branch']:.3f}")

    make_figure(gr, om, cfg, output, classic_gr=args.classic_gr, tilt=tilt)

    caption = build_caption(gr, om, cfg, args.m_threshold, len(runs),
                            duration, dt_grid, classic=args.classic_gr,
                            tilt=tilt, origin=args.origin)
    caption_path = output.with_name(output.stem + "_caption.txt")
    caption_path.write_text(caption + "\n")
    print(f"Saved: {caption_path}")
    print("\nCaption:\n" + caption)


if __name__ == "__main__":
    main()
