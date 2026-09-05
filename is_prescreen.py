"""
Static importance-sampling pre-screen of deficit-weighted magnitude laws

Re-weights an EXISTING ensemble to preview what a different magnitude law
(magnitude_law.TiltedGR: shape / target / mu / pivot / power) would do to the
pre-mainshock magnitude-composition statistics, without running the simulator.

Every event e keeps its time, hypocenter and released magnitude and receives
the importance weight

    w_e = p_target(M_nom,e | l_e) / p_run(M_nom,e | l_e),

where l_e = ln(D_e / D_w) is the reservoir coordinate the draw actually saw
(the step's start-of-step reservoir step/moment_deficit minus the geometric
moment released by earlier events of the same step; this reproduces the
stored `deficit_ratio` of new files to machine precision), p_run is the law
the run was generated with (plain truncated G-R for legacy files, so the
denominator is 1; for a tilted run the stored `size_logweight`), and p_target
is the law under test.  Under the Bath split (files with an `origin` field)
only loading-origin draws are re-weighted; Omori children keep w = 1.

THIS IS STATIC.  The reservoir path D(t), the event times, lambda(t) and the
nucleation locations are those of the original run: there is no rate-size
coupling (kappa), no reservoir feedback from the changed release, no
controller response and no re-clipping.  The pre-screen answers "given this
D(t), how would the magnitude composition of these events look under the new
law", the leading-order effect on the ordering statistic; a dynamic run of
the same law can differ (its D(t) is stiffer or softer).  Weights compound
(mainshock x target) in the superposed-epoch statistics, so watch the
effective sample size (ESS) columns: steep tilts leave few effective
mainshocks.

Per (shape, target, mu) row -- the run's own law (mu = 0 for legacy files)
reproduces the plain-catalog numbers of plot_foreshocks.py, repeated in the
first row labelled "catalog":
  ord_pre    share of M >= m_hi among M >= m_lo non-mainshock targets in the
             [-10, -1] step window before M >= m_mainshock mainshocks
             (weighted numerator and denominator, each term times the
             mainshock's own weight) over the weighted global share; 16-84%
             band from a mainshock-resampling bootstrap.  ord_post: [11, 30].
  b5.0..b6.5 pre-window share of the bands [5,5.5) [5.5,6) [6,6.5) [6.5,7)
             relative to the global share (same weighting)
  rho        Spearman rho(D_pre, M | M >= m_mainshock), D_pre the reservoir
             the draw saw; mean and sd over a weighted bootstrap (mainshocks
             resampled with probability proportional to w)
  Dpct       mean percentile of D_pre within the ensemble's pooled step-D
             distribution at M >= m_mainshock times (same bootstrap)
  P>=6.5,P>=7  tail fractions of the target law averaged uniformly over the
             run's D(t) path (its time-averaged magnitude distribution; the
             catalog row shows the run's own law)
  Em_law     time-averaged E[m | D] / E_GR of the target law over the D path
  Em_cat     weighted mean released geometric moment / E_GR: the
             realized-to-calibrated moment ratio under the static
             reweighting (> 1: the dynamic run would drain the reservoir
             until the controller / kappa responds)
  ESS        (sum w)^2 / sum w^2 over all events, and over the mainshocks

Usage:
    python is_prescreen.py --input_dir results/ensemble_ref3000 \
        --shape moment --shape linear --shape gamma [--target loglinear|power] \
        [--mu 0,0.5,1,2,3,5] [--pivot 5.0] [--power 1.0] [--reference_years 0] \
        [--m_mainshock 7.0] [--m_hi 6.5] [--m_lo 5.0] [--seed 0]
Prints the table and writes it to <input_dir>/is_prescreen.txt (or --output).
Unfinished runs (no events, short step/ group, unreadable file) are skipped.
"""

import argparse
import contextlib
import copy
import io
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

from hdf5_io import read_config
from magnitude_law import TiltedGR

BANDS = ((5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.0))
PRE_WINDOW = (-10, -1)
POST_WINDOW = (11, 30)
TILT_KEYS = ("magnitude_tilt_shape", "magnitude_tilt_target", "magnitude_tilt_mu",
             "magnitude_tilt_pivot", "magnitude_tilt_power", "magnitude_tilt_reference_years")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def run_is_complete(h5, cfg):
    """A finished run: events written, a step/ group of the full length."""
    if "events" not in h5 or "step" not in h5 or int(h5.attrs.get("n_events", 0)) <= 0:
        return False
    n_expected = int(round(float(cfg.duration_years) / float(cfg.time_step_years)))
    return h5["step"]["times"].shape[0] >= n_expected


def reservoir_seen_by_draw(k, geom_moment, D_step):
    """
    Reservoir each draw saw: the step's start-of-step D minus the geometric
    moment released by earlier events of the same step (array order is the
    processing order within a step).
    """
    order = np.argsort(k, kind="stable")
    k_s, gm_s = k[order], geom_moment[order]
    cum_before = np.cumsum(gm_s) - gm_s
    new_group = np.r_[True, k_s[1:] != k_s[:-1]] if k_s.size else np.array([], bool)
    base = np.maximum.accumulate(np.where(new_group, cum_before, 0.0))
    D_seen = np.empty(k.size)
    D_seen[order] = D_step[k_s] - (cum_before - base)
    return D_seen


def load_runs(input_dir):
    files = sorted(Path(input_dir).glob("ensemble_run_*.h5"))
    if not files:
        raise SystemExit(f"No ensemble_run_*.h5 files found in {input_dir}")
    runs, skipped = [], []
    for f in files:
        try:
            with h5py.File(f, "r") as h5:
                cfg = read_config(h5)
                if not run_is_complete(h5, cfg):
                    skipped.append(f"{f.name} (unfinished)")
                    continue
                events = h5["events"][...]
                st = h5["step"]
                times = st["times"][:]
                D_step = st["moment_deficit"][:]
                attrs = {key: float(h5.attrs[key]) for key in
                         ("initial_reservoir", "cumulative_loading", "cumulative_release") if key in h5.attrs}
        except OSError as exc:
            skipped.append(f"{f.name} ({exc})")
            continue

        n_steps = times.size
        dt_grid = float(times[1] - times[0]) if n_steps > 1 else float(cfg.time_step_years)
        duration = float(cfg.duration_years)
        L_tot = float(getattr(cfg, "geom_loading_rate_total", 0.0) or 0.0)
        if L_tot <= 0:
            L_tot = attrs.get("cumulative_loading", 0.0) / duration
        D_ref = float(getattr(cfg, "deficit_reference", 0.0) or 0.0)
        if D_ref <= 0:
            ref_years = float(getattr(cfg, "deficit_reference_years", 0.0) or 0.0)
            D_ref = ref_years * L_tot if (ref_years > 0 and L_tot > 0) else attrs.get("initial_reservoir", 0.0)
        if D_ref <= 0:
            raise SystemExit(f"{f.name}: cannot determine the reference reservoir D_ref")

        names = events.dtype.names
        t = events["time"].astype(float)
        k = np.clip(np.round(t / dt_grid).astype(int), 0, n_steps - 1)
        m = events["magnitude"].astype(float)
        if "magnitude_nominal" in names and np.isfinite(events["magnitude_nominal"]).all():
            m_nom = events["magnitude_nominal"].astype(float)
        else:
            m_nom = m.copy()
        gm = np.maximum(events["geom_moment"].astype(float), 0.0)
        D_seen = np.maximum(reservoir_seen_by_draw(k, gm, D_step), 1e-6 * D_ref)
        is_loading = (events["origin"] == 0) if "origin" in names else np.ones(events.size, bool)

        runs.append({
            "name": f.stem, "cfg": cfg, "time": t, "k": k, "magnitude": m, "m_nom": m_nom,
            "geom_moment": gm, "D_seen": D_seen, "D_step": D_step,
            "n_steps": n_steps, "dt_grid": dt_grid, "duration": duration, "L_tot": L_tot,
            "D_ref": D_ref, "is_loading": is_loading,
            "deficit_ratio": events["deficit_ratio"].astype(float) if "deficit_ratio" in names else None,
            "size_logweight": events["size_logweight"].astype(float) if "size_logweight" in names else None,
        })
    return runs, skipped


def precompute_windows(run, m_mainshock, t_min):
    """Mainshock indices and the target indices of their pre/post windows (integer grid lags)."""
    m, k = run["magnitude"], run["k"]
    is_main = m >= m_mainshock
    tgt = ~is_main
    main_idx = np.flatnonzero(is_main & (run["time"] >= t_min))
    pre_lists, post_lists = [], []
    for i in main_idx:
        lag = k - k[i]
        pre_lists.append(np.flatnonzero(tgt & (lag >= PRE_WINDOW[0]) & (lag <= PRE_WINDOW[1])))
        post_lists.append(np.flatnonzero(tgt & (lag >= POST_WINDOW[0]) & (lag <= POST_WINDOW[1])))
    run.update({"is_main": is_main, "tgt": tgt, "main_idx": main_idx,
                "pre_lists": pre_lists, "post_lists": post_lists})


# ----------------------------------------------------------------------------
# Laws and weights
# ----------------------------------------------------------------------------
def build_law(run, overrides):
    """TiltedGR from the run's config with the CLI overrides (reservoir coordinate)."""
    cfg = copy.copy(run["cfg"])
    for key, val in overrides.items():
        if val is not None:
            setattr(cfg, key, val)
    cfg.magnitude_tilt_deficit = "reservoir"  # the pre-screen coordinate is the reservoir path
    if not hasattr(cfg, "element_area_m2"):
        with contextlib.redirect_stdout(io.StringIO()):
            cfg.compute_derived_parameters()
    if not float(getattr(cfg, "geom_loading_rate_total", 0.0) or 0.0) > 0:
        cfg.geom_loading_rate_total = run["L_tot"]
    return TiltedGR(cfg, run["D_ref"])


def law_key(run):
    cfg = run["cfg"]
    return tuple(float(getattr(cfg, key)) for key in
                 ("b_value", "M_min", "M_max", "gamma_min", "gamma_max", "alpha_spatial",
                  "shear_modulus_Pa", "element_area_m2")) + (run["D_ref"], run["L_tot"])


class LawCache:
    def __init__(self):
        self._laws = {}

    def get(self, run, overrides):
        key = (law_key(run), tuple(sorted((k, v) for k, v in overrides.items() if v is not None)))
        if key not in self._laws:
            self._laws[key] = build_law(run, overrides)
        return self._laws[key]


def target_log_weight(run, law):
    """ln p_law(M_nom | l) - ln p_GR(M_nom) for loading-origin draws (0 for Omori children)."""
    lw = np.zeros(run["magnitude"].size)
    L = run["is_loading"]
    if L.any():
        ell = np.log(run["D_seen"][L] / law.D_w)
        lw[L] = law.log_weight(run["m_nom"][L], ell)
    return lw


def proposal_log_weight(run, cache):
    """ln p_run(M_nom | l) - ln p_GR(M_nom): the law the run itself drew from."""
    cfg = run["cfg"]
    mu_run = float(getattr(cfg, "magnitude_tilt_mu", 0.0) or 0.0)
    lw = np.zeros(run["magnitude"].size)
    if mu_run == 0.0:
        return lw, "truncated G-R (run mu = 0)"
    own = ", ".join(f"{key.replace('magnitude_tilt_', '')}={getattr(cfg, key)}" for key in TILT_KEYS)
    stored = run["size_logweight"]
    L = run["is_loading"]
    if stored is not None and np.isfinite(stored[L]).all():
        lw[L] = stored[L]
        return lw, f"run's own tilted law from stored size_logweight ({own}, deficit={cfg.magnitude_tilt_deficit})"
    lw = target_log_weight(run, cache.get(run, {}))
    return lw, f"run's own tilted law recomputed on the reservoir coordinate ({own})"


def validate_coordinates(runs, cache):
    """Consistency of the reconstructed draw coordinate with the stored per-event fields."""
    lines = []
    for run in runs:
        if run["deficit_ratio"] is None:
            continue
        L = run["is_loading"] & np.isfinite(run["deficit_ratio"])
        if not L.any():
            continue
        law0 = cache.get(run, {})
        ratio = run["D_seen"][L] / law0.D_w
        rel = np.abs(ratio - run["deficit_ratio"][L]) / run["deficit_ratio"][L]
        msg = f"  {run['name']}: D_seen/D_w vs stored deficit_ratio, max rel diff {rel.max():.2e} ({L.sum()} loading draws)"
        if run["size_logweight"] is not None and float(getattr(run["cfg"], "magnitude_tilt_mu", 0.0) or 0.0) != 0.0 \
                and getattr(run["cfg"], "magnitude_tilt_deficit", "reservoir") == "reservoir":
            lw = target_log_weight(run, law0)
            msg += f"; recomputed vs stored size_logweight max abs diff {np.max(np.abs(lw[L] - run['size_logweight'][L])):.2e}"
        lines.append(msg)
    return lines


# ----------------------------------------------------------------------------
# Weighted statistics
# ----------------------------------------------------------------------------
def weighted_ordering(runs, weights, m_hi, m_lo, n_boot, rng):
    """Weighted share of M>=m_hi among M>=m_lo targets in the pre/post windows vs global."""
    n_bands = len(BANDS)
    pre_hi, pre_lo, post_hi, post_lo, band_pre = [], [], [], [], []
    g_hi = g_lo = 0.0
    g_band = np.zeros(n_bands)
    for run, w in zip(runs, weights):
        m, tgt = run["magnitude"], run["tgt"]
        g_hi += w[tgt & (m >= m_hi)].sum()
        g_lo += w[tgt & (m >= m_lo)].sum()
        for b, (lo, hi) in enumerate(BANDS):
            g_band[b] += w[tgt & (m >= lo) & (m < hi)].sum()
        for i, pre, post in zip(run["main_idx"], run["pre_lists"], run["post_lists"]):
            wm, mp, wp, mq, wq = w[i], m[pre], w[pre], m[post], w[post]
            pre_hi.append(wm * wp[mp >= m_hi].sum())
            pre_lo.append(wm * wp[mp >= m_lo].sum())
            post_hi.append(wm * wq[mq >= m_hi].sum())
            post_lo.append(wm * wq[mq >= m_lo].sum())
            band_pre.append([wm * wp[(mp >= lo) & (mp < hi)].sum() for lo, hi in BANDS])
    pre_hi, pre_lo, post_hi, post_lo = (np.array(v, dtype=float) for v in (pre_hi, pre_lo, post_hi, post_lo))
    band_pre = np.array(band_pre, dtype=float).reshape(-1, n_bands)
    n = pre_hi.size
    glob = g_hi / g_lo if g_lo > 0 else np.nan
    share_pre = pre_hi.sum() / pre_lo.sum() if pre_lo.sum() > 0 else np.nan
    share_post = post_hi.sum() / post_lo.sum() if post_lo.sum() > 0 else np.nan
    boots = []
    for _ in range(n_boot if n else 0):
        idx = rng.integers(0, n, n)
        if pre_lo[idx].sum() > 0 and glob > 0:
            boots.append(pre_hi[idx].sum() / pre_lo[idx].sum() / glob)
    lo, hi = (np.percentile(boots, [16, 84]) if boots else (np.nan, np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        band_ratio = (band_pre.sum(axis=0) / pre_lo.sum()) / (g_band / g_lo) if (n and g_lo > 0) else np.full(n_bands, np.nan)
    return {"pre": share_pre / glob if glob > 0 else np.nan, "lo": lo, "hi": hi,
            "post": share_post / glob if glob > 0 else np.nan, "bands": band_ratio,
            "share_pre": share_pre, "global": glob, "n_main": int(n)}


def weighted_dm_stats(runs, weights, D_pooled, n_boot, rng):
    """
    Spearman rho(D_pre, M) and the mean D-percentile over the mainshocks
    (weighted bootstrap); percentiles against the pooled, sorted step-D of
    the ensemble.
    """
    D, M, pct, w = [], [], [], []
    for run, wr in zip(runs, weights):
        idx = run["main_idx"]
        D.append(run["D_seen"][idx])
        M.append(run["magnitude"][idx])
        w.append(wr[idx])
        pct.append(np.searchsorted(D_pooled, run["D_seen"][idx], side="left") / D_pooled.size)
    D, M, pct, w = (np.concatenate(v) if v else np.array([]) for v in (D, M, pct, w))
    out = {"n": D.size, "rho": np.nan, "rho_sd": np.nan, "pct": np.nan, "pct_sd": np.nan,
           "rho_plain": np.nan, "pct_plain": np.nan, "pct_w": np.nan, "ess": 0.0}
    if D.size == 0:
        return out
    out["ess"] = w.sum() ** 2 / (w ** 2).sum()
    out["pct_plain"] = float(pct.mean())
    out["pct_w"] = float((w * pct).sum() / w.sum())
    if D.size < 3:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["rho_plain"] = float(spearmanr(D, M).correlation)
        if n_boot <= 0:
            return out
        p = w / w.sum()
        rhos, pcts = [], []
        for _ in range(n_boot):
            idx = rng.choice(D.size, D.size, p=p)
            rhos.append(spearmanr(D[idx], M[idx]).correlation)
            pcts.append(pct[idx].mean())
    out.update({"rho": float(np.nanmean(rhos)), "rho_sd": float(np.nanstd(rhos)),
                "pct": float(np.mean(pcts)), "pct_sd": float(np.std(pcts))})
    return out


def law_time_averages(runs, law):
    """Tail fractions and E[m]/E_GR of the law averaged uniformly over the runs' D(t) paths."""
    ell = np.concatenate([np.log(np.maximum(r["D_step"], 1e-6 * r["D_ref"]) / law.D_w) for r in runs])
    thetas = np.asarray(law.theta(ell), dtype=float)
    M, pbar = law.marginal(thetas)

    def tail(thr):
        return float(np.trapezoid(np.where(M >= thr, pbar, 0.0), dx=law.dM))

    return tail(6.5), tail(7.0), float(np.trapezoid(law.m * pbar, dx=law.dM)) / law.E_gr


def law_summary(law):
    parts = []
    for r in (0.6, 0.8, 1.2, 1.5):
        l = np.log(r)
        parts.append(f"{r:.1f}: th={float(law.theta(l)):+.2f} E={law.expected_moment(l) / law.E_gr:.2f} "
                     f"P7={law.tail(l, 7.0) / max(law.tail(0.0, 7.0), 1e-300):.2f}")
    return f"c_phi={law.c_phi:.4f}  D/D_w -> " + " | ".join(parts)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
HEADER = ("law                   mu | ord_pre [16%,84%]  ord_post | b5.0  b5.5  b6.0  b6.5 | "
          "rho    sd   Dpct   sd  | P>=6.5 P>=7   | Em_law Em_cat |    ESS  ESS_ms")


def format_row(label, mu, od, dm, p65, p7, em_law, em_cat, ess):
    mu_s = f"{mu:>4g}" if mu is not None else "   -"
    b = od["bands"]
    return (f"{label:<21s} {mu_s} | {od['pre']:5.2f} [{od['lo']:4.2f},{od['hi']:4.2f}] {od['post']:6.2f}   | "
            f"{b[0]:5.2f} {b[1]:5.2f} {b[2]:5.2f} {b[3]:5.2f} | "
            f"{dm['rho']:+5.2f} {dm['rho_sd']:4.2f}  {dm['pct']:5.2f} {dm['pct_sd']:4.2f} | "
            f"{p65:6.4f} {p7:6.4f} | {em_law:6.3f} {em_cat:6.3f} | {ess:6.0f} {dm['ess']:7.1f}")


def main():
    parser = argparse.ArgumentParser(description="Static importance-sampling pre-screen of magnitude laws")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--shape", action="append", choices=["moment", "gamma", "linear"],
                        help="Tilt shape (repeatable; default moment)")
    parser.add_argument("--target", choices=["loglinear", "power"], default="loglinear")
    parser.add_argument("--mu", type=str, default="0,0.5,1,2,3,5", help="Comma list of mu values")
    parser.add_argument("--pivot", type=float, default=None, help="magnitude_tilt_pivot (default: the run's)")
    parser.add_argument("--power", type=float, default=None, help="magnitude_tilt_power (default: the run's)")
    parser.add_argument("--reference_years", type=float, default=None,
                        help="magnitude_tilt_reference_years, the neutral point D_w (default: the run's; 0 = D_ref)")
    parser.add_argument("--m_mainshock", type=float, default=7.0)
    parser.add_argument("--m_hi", type=float, default=6.5)
    parser.add_argument("--m_lo", type=float, default=5.0)
    parser.add_argument("--t_min", type=float, default=0.0, help="Ignore mainshocks before this time (yr)")
    parser.add_argument("--n_boot", type=int, default=1000, help="Mainshock bootstrap draws for the ordering band")
    parser.add_argument("--n_boot_is", type=int, default=200, help="Weighted bootstrap draws for rho / Dpct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Table file (default <input_dir>/is_prescreen.txt)")
    args = parser.parse_args()

    shapes = args.shape or ["moment"]
    mus = [float(v) for v in args.mu.split(",")]
    rng = np.random.default_rng(args.seed)
    runs, skipped = load_runs(args.input_dir)
    if not runs:
        raise SystemExit(f"No finished runs in {args.input_dir} (skipped: {skipped})")
    for run in runs:
        precompute_windows(run, args.m_mainshock, args.t_min)
    cache = LawCache()

    lines = [f"Importance-sampling pre-screen (STATIC: original D(t), times, locations) -- {args.input_dir}",
             f"  args: shapes={shapes} target={args.target} mu={mus} pivot={args.pivot} power={args.power} "
             f"reference_years={args.reference_years} m_mainshock={args.m_mainshock} m_hi={args.m_hi} "
             f"m_lo={args.m_lo} t_min={args.t_min} seed={args.seed}"]
    if skipped:
        lines.append(f"  skipped: {', '.join(skipped)}")
    n_events = sum(r["magnitude"].size for r in runs)
    n_children = sum(int((~r["is_loading"]).sum()) for r in runs)
    n_main = sum(r["main_idx"].size for r in runs)
    D_years = np.concatenate([r["D_step"] / r["L_tot"] for r in runs])
    lines.append(f"  runs: {len(runs)} x {runs[0]['duration']:g} yr, {n_events} events "
                 f"({n_children} Omori children kept at w = 1), {n_main} M>={args.m_mainshock:g} mainshocks; "
                 f"D_ref = {runs[0]['D_ref'] / runs[0]['L_tot']:.0f} yr, D path {D_years.mean():.0f} +/- {D_years.std():.0f} yr")

    # Proposal (the run's own law)
    proposals, descs = [], set()
    for run in runs:
        lw, desc = proposal_log_weight(run, cache)
        proposals.append(lw)
        descs.add(desc)
    lines.append("  proposal law: " + "; ".join(sorted(descs)))
    lines.extend(validate_coordinates(runs, cache))

    ones = [np.ones(r["magnitude"].size) for r in runs]
    D_pooled = np.sort(np.concatenate([r["D_step"] for r in runs]))
    dm0 = weighted_dm_stats(runs, ones, D_pooled, 0, rng)
    lines.append(f"  plain catalog: rho(D_pre, M | M>={args.m_mainshock:g}) = {dm0['rho_plain']:+.3f}, "
                 f"mean D-percentile = {dm0['pct_plain']:.3f} (n = {dm0['n']})")

    # Rows; the catalog row carries the run's own law (plain G-R for mu = 0
    # files) in its law columns, E_GR is the same for every law
    rows = []
    law0 = cache.get(runs[0], {"magnitude_tilt_mu": 0.0})
    p65, p7, em = law_time_averages(runs, cache.get(runs[0], {}))
    od = weighted_ordering(runs, ones, args.m_hi, args.m_lo, args.n_boot, rng)
    dm = weighted_dm_stats(runs, ones, D_pooled, args.n_boot_is, rng)
    em_cat = sum((w * r["geom_moment"]).sum() for r, w in zip(runs, ones)) / n_events / law0.E_gr
    rows.append(format_row("catalog", None, od, dm, p65, p7, em, em_cat, float(n_events)))
    law_lines = []
    for shape in shapes:
        for mu in mus:
            overrides = {"magnitude_tilt_shape": shape, "magnitude_tilt_target": args.target,
                         "magnitude_tilt_mu": mu, "magnitude_tilt_pivot": args.pivot,
                         "magnitude_tilt_power": args.power,
                         "magnitude_tilt_reference_years": args.reference_years}
            weights, law = [], None
            for run, lw_prop in zip(runs, proposals):
                law = cache.get(run, overrides)
                weights.append(np.exp(target_log_weight(run, law) - lw_prop))
            w_all = np.concatenate(weights)
            ess = w_all.sum() ** 2 / (w_all ** 2).sum()
            od = weighted_ordering(runs, weights, args.m_hi, args.m_lo, args.n_boot, rng)
            dm = weighted_dm_stats(runs, weights, D_pooled, args.n_boot_is, rng)
            p65, p7, em = law_time_averages(runs, law)
            em_cat = sum((w * r["geom_moment"]).sum() for r, w in zip(runs, weights)) / w_all.sum() / law.E_gr
            label = f"{shape}/{args.target}"
            rows.append(format_row(label, mu, od, dm, p65, p7, em, em_cat, ess))
            if mu != 0.0:
                law_lines.append(f"  {label} mu={mu:g}: {law_summary(law)}")

    lines.append("")
    lines.append(HEADER)
    lines.extend(rows)
    lines.append("")
    lines.append(f"  E_GR[m] = {law0.E_gr:.4e} m^3; G-R tails P(M>=6.5) = {law0.tail(0.0, 6.5):.4f}, "
                 f"P(M>=7) = {law0.tail(0.0, 7.0):.4f}; ord = share(M>={args.m_hi:g} | M>={args.m_lo:g}) in "
                 f"[{PRE_WINDOW[0]},{PRE_WINDOW[1]}] / global (post: [{POST_WINDOW[0]},{POST_WINDOW[1]}]); "
                 f"bands = pre-window share / global; rho, Dpct: weighted bootstrap of {args.n_boot_is} draws")
    lines.append("  law summaries (theta, E[m]/E_GR, P(M>=7)/GR at D/D_w):")
    lines.extend(law_lines)

    text = "\n".join(lines)
    print(text)
    out = Path(args.output) if args.output else Path(args.input_dir) / "is_prescreen.txt"
    out.write_text(text + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
