"""
Pre-mainshock seismicity diagnostics for simulation ensembles.

Quantifies how much seismicity precedes large mainshocks, and why, so that
changes to the event sampling, Omori integration, or rate law can be judged
against the same yardstick. Several ensembles can be overlaid (one colour
per input directory).

Panels (all lags in integer steps of the output grid; events sit on the
grid, so the literal [-1, 0) year window is empty and same-step co-events
are reported separately):
(a) Superposed-epoch rate ratio of M >= m_target events around M >=
    m_mainshock mainshocks: observed count / expected count from the
    long-term rate, stacked over mainshocks, with a bootstrap band
    (resampling mainshocks). Solid: whole fault; dashed: within
    +/- local_km along strike of the mainshock hypocenter (normalized by
    that strip's own long-term rate, which removes the loading-pulse
    heterogeneity).
(b) Stacked rate decomposition around the same mainshocks: the loading
    (deficit-driven) and aftershock (Omori) components of lambda(t),
    normalized by the run-mean total rate. Read from the 'step/' group;
    for older files the Omori term is reconstructed from the catalog with
    the legacy point-sampled kernel.
(c) Fano factor (variance/mean of counts) versus window length. The
    deterministic accumulator gives F -> 0 at short windows; Poisson
    sampling gives F ~ 1 at short windows (over-dispersion at long windows
    reflects the slowly varying rate and is expected in both).
(d) Counting noise: standard deviation of N_obs - int(lambda dt) over
    non-overlapping windows. The accumulator is bounded at ~0.4 events at
    every window; Poisson sampling tracks sqrt(<N>) (dotted).

Outputs:
    foreshocks.png          Four-panel figure (500 dpi)
    foreshocks_summary.txt  Acceptance numbers per ensemble

Usage:
    python plot_foreshocks.py [--input_dir DIR [DIR ...]] [--m_mainshock 7.5]
                              [--m_targets 5,6] [--window 50] [--bin 5]
                              [--local_km 25] [--n_boot 1000] [--output PATH]
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from hdf5_io import read_config

FONTSIZE = 8
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_runs(input_dir):
    """Load the per-run catalog and rate history needed for the diagnostics."""
    files = sorted(Path(input_dir).glob("ensemble_run_*.h5"))
    if not files:
        raise SystemExit(f"No ensemble_run_*.h5 files found in {input_dir}")

    runs = []
    for f in files:
        with h5py.File(f, "r") as h5:
            cfg = read_config(h5)
            events = h5["events"][...]
            if "step" in h5:
                # Per-step history is written every step (snapshots may be sparser)
                times = h5["step"]["times"][:]
                lam = h5["step"]["lambda_total"][:]
            else:
                times = h5["times"][:]
                lam = h5["lambda_history"][:]
            dt_grid = float(times[1] - times[0]) if times.size > 1 else 1.0
            dt_step = float(getattr(cfg, "time_step_years", dt_grid))
            run = {
                "name": f.stem,
                "time": events["time"].astype(float),
                "offset": (events["time_offset"].astype(float)
                           if "time_offset" in events.dtype.names
                           else np.zeros(events.size)),
                "magnitude": events["magnitude"].astype(float),
                "x": events["hypocenter_x_km"].astype(float),
                "dt_grid": dt_grid,
                "dt_step": dt_step,
                "duration": float(cfg.duration_years),
                "n_steps": lam.size,
                "lam": lam,
                "expected_count": None,
                "lam_loading": None,
                "lam_aftershock": None,
                "correction": None,
                "coupling_final": float(h5.attrs["cumulative_release"] / h5.attrs["cumulative_loading"]),
                "cfg": cfg,
            }
            run["reservoir"] = None
            run["L_tot"] = float(h5.attrs["cumulative_loading"]) / float(cfg.duration_years)
            if "magnitude_nominal" in events.dtype.names and np.isfinite(events["magnitude_nominal"]).all():
                run["m_nom"] = events["magnitude_nominal"].astype(float)
            else:
                # Fallback: nominal magnitude from the rupture element count
                # (Allen & Hayes area), saturates at the whole-fault cap
                ea = h5["event_arrays"]
                n_el = np.array([ea[f"event_{i:06d}"]["ruptured_elements"].shape[0] for i in range(events.size)])
                run["m_nom"] = np.log10(np.maximum(n_el, 1) * cfg.element_size_km**2) + 3.99
            if "step" in h5:
                st = h5["step"]
                run["expected_count"] = st["expected_count"][:]
                run["lam_loading"] = st["lambda_loading"][:]
                run["lam_aftershock"] = st["lambda_aftershock"][:]
                run["correction"] = st["correction_factor"][:]
                run["n_steps"] = st["times"].shape[0]
                if "reservoir" in st:
                    run["reservoir"] = st["reservoir"][:]
            else:
                run["expected_count"] = lam * dt_step
                run["lam_aftershock"] = reconstruct_legacy_omori(run, cfg)
                run["lam_loading"] = lam - run["lam_aftershock"]
            run["step_index"] = np.round(run["time"] / dt_grid).astype(int)
            runs.append(run)
    return runs


def reconstruct_legacy_omori(run, cfg):
    """Omori term of lambda on the grid with the legacy point-sampled kernel."""
    n_steps = run["n_steps"]
    dt_grid = run["dt_grid"]
    lam_omori = np.zeros(n_steps)
    if not getattr(cfg, "omori_enabled", False):
        return lam_omori
    K = cfg.omori_K_ref * 10 ** (cfg.omori_alpha * (run["magnitude"] - cfg.omori_M_ref))
    grid = np.arange(n_steps) * dt_grid
    for t_e, K_e in zip(run["time"], K):
        dt = grid - t_e
        mask = (dt > 0) & (dt <= cfg.omori_duration_years)
        lam_omori[mask] += K_e / (dt[mask] + cfg.omori_c_years) ** cfg.omori_p
    return lam_omori


# ----------------------------------------------------------------------------
# Superposed-epoch statistics
# ----------------------------------------------------------------------------
def bin_edges(window, width):
    """Integer-lag bin edges in [-window, window] excluding lag 0."""
    neg = np.arange(-window, 0, width)
    if neg[-1] != 0:
        neg = np.append(neg, 0)
    pos = np.arange(1, window + 1, width)
    if pos[-1] != window + 1:
        pos = np.append(pos, window + 1)
    return neg, pos


def mainshock_mask(run, m_mainshock):
    """Mainshock selector: actual magnitude (default) or nominal draw (--key nominal)."""
    if run.get("key", "actual") == "nominal":
        return run["m_nom"] >= m_mainshock
    return run["magnitude"] >= m_mainshock


def stack_counts(runs, m_mainshock, m_target, window, width, local_km=None):
    """
    Observed and expected target-event counts per lag bin for every mainshock

    Returns dict with bin centers (years), per-mainshock observed and expected
    arrays (n_mainshocks x n_bins), and the number of same-step co-events.
    """
    neg, pos = bin_edges(window, width)
    edges = np.concatenate([neg, pos])  # lag-0 bin is [0, 1) -> excluded below
    n_bins = edges.size - 1
    lag0_bin = np.searchsorted(edges, 0, side="right") - 1

    obs_rows, exp_rows, co_events = [], [], 0
    for run in runs:
        m, k, x = run["magnitude"], run["step_index"], run["x"]
        n_steps, dt_grid, T = run["n_steps"], run["dt_grid"], run["duration"]
        is_main = mainshock_mask(run, m_mainshock)
        is_target = (m >= m_target) & ~is_main
        t_min = run.get("t_min", 0.0)
        for i in np.flatnonzero(is_main & (run["time"] >= t_min)):
            sel = is_target.copy()
            if local_km is not None:
                sel &= np.abs(x - x[i]) <= local_km
            rate_long = sel.sum() / T  # events per year in this selection
            lags = k[sel] - k[i]
            counts, _ = np.histogram(lags, bins=edges)
            # exposure: steps of each bin that fall inside the catalog
            exposure = np.zeros(n_bins)
            for j in range(n_bins):
                lo, hi = edges[j], edges[j + 1]
                lo_c = max(lo, -k[i])
                hi_c = min(hi, n_steps - k[i])
                exposure[j] = max(0, hi_c - lo_c) * dt_grid
            co_events += counts[lag0_bin]
            obs_rows.append(np.delete(counts, lag0_bin).astype(float))
            exp_rows.append(np.delete(rate_long * exposure, lag0_bin))
    centers = 0.5 * (edges[:-1] + edges[1:])
    centers = np.delete(centers, lag0_bin) * runs[0]["dt_grid"]
    return {
        "centers": centers,
        "obs": np.array(obs_rows) if obs_rows else np.zeros((0, n_bins - 1)),
        "exp": np.array(exp_rows) if exp_rows else np.zeros((0, n_bins - 1)),
        "co_events": int(co_events),
        "n_mainshocks": len(obs_rows),
    }


def ratio_with_bootstrap(obs, exp, n_boot, rng, pre_mask=None):
    """Stacked ratio per bin and a bootstrap (16-84%) band over mainshocks."""
    if obs.shape[0] == 0:
        nan = np.full(obs.shape[1], np.nan)
        return nan, nan, nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = obs.sum(axis=0) / exp.sum(axis=0)
    boots = np.empty((n_boot, obs.shape[1]))
    n = obs.shape[0]
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            boots[b] = obs[idx].sum(axis=0) / exp[idx].sum(axis=0)
    lo, hi = np.nanpercentile(boots, [16, 84], axis=0)
    return ratio, lo, hi


def window_ratio(obs, exp, centers, lo_yr, hi_yr, n_boot, rng):
    """Scalar observed/expected over the bins with lo_yr <= center < hi_yr."""
    sel = (centers >= lo_yr) & (centers < hi_yr)
    if obs.shape[0] == 0 or not sel.any():
        return np.nan, np.nan, np.nan
    o, e = obs[:, sel].sum(axis=1), exp[:, sel].sum(axis=1)
    ratio = o.sum() / e.sum() if e.sum() > 0 else np.nan
    boots = []
    n = o.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if e[idx].sum() > 0:
            boots.append(o[idx].sum() / e[idx].sum())
    lo, hi = (np.percentile(boots, [16, 84]) if boots else (np.nan, np.nan))
    return ratio, lo, hi


def stack_rate_components(runs, m_mainshock, window):
    """Mean lambda_loading and lambda_aftershock (in units of the run-mean
    total rate) at integer lags around every mainshock, pooled over runs."""
    lags = np.arange(-window, window + 1)
    sums = {"loading": np.zeros(lags.size), "aftershock": np.zeros(lags.size),
            "reservoir": np.zeros(lags.size)}
    counts = np.zeros(lags.size)
    counts_res = np.zeros(lags.size)
    for run in runs:
        lam_mean = run["lam"].mean()
        if lam_mean <= 0:
            continue
        k_main = run["step_index"][mainshock_mask(run, m_mainshock) & (run["time"] >= run.get("t_min", 0.0))]
        for k in k_main:
            idx = k + lags
            ok = (idx >= 0) & (idx < run["n_steps"])
            sums["loading"][ok] += run["lam_loading"][idx[ok]] / lam_mean
            sums["aftershock"][ok] += run["lam_aftershock"][idx[ok]] / lam_mean
            counts[ok] += 1
            if run["reservoir"] is not None:
                sums["reservoir"][ok] += run["reservoir"][idx[ok]] / run["reservoir"].mean()
                counts_res[ok] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "lags_yr": lags * runs[0]["dt_grid"],
            "loading": sums["loading"] / counts,
            "aftershock": sums["aftershock"] / counts,
            "reservoir": sums["reservoir"] / counts_res,
        }


# ----------------------------------------------------------------------------
# Counting statistics
# ----------------------------------------------------------------------------
def counts_per_window(run, width, m_min):
    sel = run["magnitude"] >= m_min
    k = run["step_index"][sel]
    n_win = run["n_steps"] // width
    if n_win == 0:
        return np.array([])
    return np.bincount(np.minimum(k // width, n_win - 1), minlength=n_win)[:n_win]


def fano_factor(run, width, m_min):
    c = counts_per_window(run, width, m_min)
    return c.var() / c.mean() if c.size > 1 and c.mean() > 0 else np.nan


def counting_noise(run, width, m_min):
    """sd of N_obs - sum(mu) over non-overlapping windows, and mean N."""
    c = counts_per_window(run, width, m_min)
    n_win = c.size
    if n_win < 2:
        return np.nan, np.nan
    mu = run["expected_count"][: n_win * width].reshape(n_win, width).sum(axis=1)
    return float(np.std(c - mu)), float(c.mean())


def interevent_stats(run, m_min):
    sel = run["magnitude"] >= m_min
    t_grid = np.sort(run["time"][sel])
    t_cont = np.sort((run["time"] + run["offset"])[sel])
    out = {}
    for label, t in (("grid", t_grid), ("continuous", t_cont)):
        if t.size < 3:
            out[label] = (np.nan, np.nan)
            continue
        dt = np.diff(t)
        out[label] = (dt.std() / dt.mean() if dt.mean() > 0 else np.nan, dt.max())
    return out


def reservoir_stats(runs, d_floor):
    """Reservoir level (years of loading): run and second-half statistics."""
    out = []
    for run in runs:
        if run["reservoir"] is None:
            continue
        D = run["reservoir"] / run["L_tot"]
        half = D[D.size // 2:]
        out.append({"mean": D.mean(), "sd": D.std(), "half_mean": half.mean(), "half_sd": half.std(),
                    "min": D.min(), "max": D.max(), "frac_below": float(np.mean(half < d_floor))})
    return out


def clipping_stats(runs, m_mainshock, window, tol=0.05):
    """Fraction of events released below their nominal draw, overall and around mainshocks."""
    tot = {"all": [0, 0], "6-7": [0, 0], "7+": [0, 0], "pre": [0, 0], "post": [0, 0]}
    for run in runs:
        m, mn, k = run["magnitude"], run["m_nom"], run["step_index"]
        clipped = (mn - m) > tol
        for key, sel in (("all", np.ones(m.size, bool)), ("6-7", (mn >= 6) & (mn < 7)), ("7+", mn >= 7)):
            tot[key][0] += clipped[sel].sum(); tot[key][1] += sel.sum()
        k_main = k[mainshock_mask(run, m_mainshock) & (run["time"] >= run.get("t_min", 0.0))]
        for km in k_main:
            lag = k - km
            pre = (lag >= -10) & (lag <= -1)
            post = (lag >= 11) & (lag <= 30)
            tot["pre"][0] += clipped[pre].sum(); tot["pre"][1] += pre.sum()
            tot["post"][0] += clipped[post].sum(); tot["post"][1] += post.sum()
    return {key: (v[0] / v[1] if v[1] else np.nan) for key, v in tot.items()}


def hosting_by_cycle_age(runs, m_ref=7.0, bands=((6.0, 6.5), (6.5, 7.0), (7.0, 7.5)),
                         edges=(0, 20, 40, 70, 100, 150, np.inf), tol=0.05):
    """P(released within tol of nominal | nominal band) vs years since the last actual M>=m_ref."""
    table = {b: np.zeros((len(edges) - 1, 2)) for b in bands}
    for run in runs:
        t, m, mn = run["time"], run["magnitude"], run["m_nom"]
        t_big = t[m >= m_ref]
        for i in range(t.size):
            prev = t_big[t_big < t[i]]
            if prev.size == 0:
                continue
            age = t[i] - prev.max()
            j = np.searchsorted(edges, age, side="right") - 1
            for b in bands:
                if b[0] <= mn[i] < b[1]:
                    table[b][j, 0] += (mn[i] - m[i]) <= tol
                    table[b][j, 1] += 1
    return {b: v[:, 0] / np.maximum(v[:, 1], 1) for b, v in table.items()}, {b: v[:, 1] for b, v in table.items()}, edges


def magnitude_ordering(runs, m_mainshock, m_hi, m_lo, n_boot, rng):
    """Share of M>=m_hi among M>=m_lo targets in [-10,-1] and [11,30] vs the global share."""
    pre_hi, pre_lo, post_hi, post_lo = [], [], [], []
    g_hi = g_lo = 0
    for run in runs:
        m, k = run["magnitude"], run["step_index"]
        is_main = mainshock_mask(run, m_mainshock)
        tgt = ~is_main
        g_hi += ((m >= m_hi) & tgt).sum(); g_lo += ((m >= m_lo) & tgt).sum()
        for km in k[is_main & (run["time"] >= run.get("t_min", 0.0))]:
            lag = k - km
            pre = (lag >= -10) & (lag <= -1) & tgt
            post = (lag >= 11) & (lag <= 30) & tgt
            pre_hi.append(((m >= m_hi) & pre).sum()); pre_lo.append(((m >= m_lo) & pre).sum())
            post_hi.append(((m >= m_hi) & post).sum()); post_lo.append(((m >= m_lo) & post).sum())
    pre_hi, pre_lo, post_hi, post_lo = map(np.array, (pre_hi, pre_lo, post_hi, post_lo))
    glob = g_hi / g_lo if g_lo else np.nan
    share_pre = pre_hi.sum() / pre_lo.sum() if pre_lo.sum() else np.nan
    share_post = post_hi.sum() / post_lo.sum() if post_lo.sum() else np.nan
    boots = []
    n = pre_hi.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, n) if n else np.array([], int)
        if pre_lo[idx].sum() > 0 and glob > 0:
            boots.append(pre_hi[idx].sum() / pre_lo[idx].sum() / glob)
    lo, hi = (np.percentile(boots, [16, 84]) if boots else (np.nan, np.nan))
    return {"pre": share_pre, "post": share_post, "global": glob,
            "pre_over_global": share_pre / glob if glob else np.nan, "lo": lo, "hi": hi,
            "n_pre_hi": int(pre_hi.sum()), "n_pre_lo": int(pre_lo.sum())}


def aki_b(mags, m_min):
    m = mags[mags >= m_min]
    return np.log10(np.e) / (m.mean() - m_min) if m.size > 1 else np.nan


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def analyze(runs, args, rng):
    m_targets = [float(v) for v in args.m_targets.split(",")]
    W, width, L = args.window, args.bin, args.local_km
    res = {"m_targets": m_targets, "stacks": {}, "ratios": {}}

    for m_t in m_targets:
        for scope, local in (("whole", None), ("local", L)):
            st = stack_counts(runs, args.m_mainshock, m_t, W, width, local)
            r, lo, hi = ratio_with_bootstrap(st["obs"], st["exp"], args.n_boot, rng)
            st.update({"ratio": r, "lo": lo, "hi": hi})
            res["stacks"][(m_t, scope)] = st
            dt_grid = runs[0]["dt_grid"]
            res["ratios"][(m_t, scope, "pre")] = window_ratio(
                st["obs"], st["exp"], st["centers"], -W * dt_grid, 0, args.n_boot, rng)
            res["ratios"][(m_t, scope, "post")] = window_ratio(
                st["obs"], st["exp"], st["centers"], 0, (W + 1) * dt_grid, args.n_boot, rng)
            res["ratios"][(m_t, scope, "pre10")] = window_ratio(
                st["obs"], st["exp"], st["centers"], -10 * dt_grid, 0, args.n_boot, rng)
            for lo, hi in ((1, 10), (3, 20), (11, 30), (41, 50)):
                res["ratios"][(m_t, scope, f"post{lo}_{hi}")] = window_ratio(
                    st["obs"], st["exp"], st["centers"], lo * dt_grid, (hi + 1) * dt_grid, args.n_boot, rng)

    res["components"] = stack_rate_components(runs, args.m_mainshock, W)
    comp = res["components"]
    lags = comp["lags_yr"]
    early = comp["loading"][(lags >= -W) & (lags < -W + 10)]
    late = comp["loading"][(lags >= -10) & (lags < 0)]
    res["flatness"] = (np.nanmean(late) / np.nanmean(early)
                       if early.size and late.size and np.nanmean(early) > 0 else np.nan)

    widths = [1, 2, 5, 10, 20, 50]
    m0 = m_targets[0]
    res["fano_widths"] = np.array(widths) * runs[0]["dt_grid"]
    res["fano"] = np.array([[fano_factor(r, w, m0) for w in widths] for r in runs])
    noise = np.array([[counting_noise(r, w, m0) for w in widths] for r in runs])
    res["noise_sd"] = noise[:, :, 0]
    res["noise_meanN"] = noise[:, :, 1]

    res["reservoir"] = reservoir_stats(runs, args.d_floor)
    res["clipping"] = clipping_stats(runs, args.m_mainshock, W)
    res["hosting"] = hosting_by_cycle_age(runs)
    res["ordering"] = {m: magnitude_ordering(runs, args.m_mainshock, m, m0, args.n_boot, rng)
                       for m in m_targets[1:]}

    # Catalog-level scalars
    N = sum(int((r["magnitude"] >= m0).sum()) for r in runs)
    T = sum(r["duration"] for r in runs)
    mu_total = sum(float(r["expected_count"].sum()) for r in runs)
    mags = np.concatenate([r["magnitude"] for r in runs])
    ie = [interevent_stats(r, m0) for r in runs]
    res["scalars"] = {
        "n_runs": len(runs),
        "duration": runs[0]["duration"],
        "N": N,
        "rate": N / T,
        "n_mainshocks": res["stacks"][(m0, "whole")]["n_mainshocks"],
        "co_events": res["stacks"][(m0, "whole")]["co_events"],
        "N_over_int_lambda": N / mu_total if mu_total > 0 else np.nan,
        "b_aki": aki_b(mags, m0),
        "coupling": np.mean([r["coupling_final"] for r in runs]),
        "cv_grid": np.nanmean([s["grid"][0] for s in ie]),
        "cv_cont": np.nanmean([s["continuous"][0] for s in ie]),
        "max_gap": np.nanmax([s["grid"][1] for s in ie]),
        "fano_1": np.nanmean(res["fano"][:, 0]),
        "corr_range": (
            (min(float(r["correction"].min()) for r in runs if r["correction"] is not None),
             max(float(r["correction"].max()) for r in runs if r["correction"] is not None))
            if any(r["correction"] is not None for r in runs) else None),
        "sampling": getattr(runs[0]["cfg"], "event_sampling", "deterministic"),
        "omori_integrated": bool(getattr(runs[0]["cfg"], "omori_integrate_over_step", False)),
        "nu": float(getattr(runs[0]["cfg"], "deficit_exponent", 1.0)),
        "rate_mode": getattr(runs[0]["cfg"], "rate_mode", "legacy"),
    }
    return res


def summarize(label, res, args):
    s = res["scalars"]
    m_t = res["m_targets"]
    lines = [f"[{label}]",
             f"  runs: {s['n_runs']} x {s['duration']:.0f} yr; sampling={s['sampling']}, "
             f"omori_integrated={s['omori_integrated']}, rate_mode={s['rate_mode']}, nu={s['nu']:g}",
             f"  events M>={m_t[0]:g}: {s['N']} ({s['rate']:.3f}/yr); mainshocks M>={args.m_mainshock:g}: {s['n_mainshocks']}",
             f"  b (Aki): {s['b_aki']:.3f}; geometric coupling: {s['coupling']:.3f}; N_obs/int(lambda dt): {s['N_over_int_lambda']:.3f}",
             f"  interevent CV (grid): {s['cv_grid']:.2f}, (with sub-step offsets): {s['cv_cont']:.2f}; "
             f"max gap: {s['max_gap']:.1f} yr; Fano(1 step): {s['fano_1']:.2f}; same-step co-events: {s['co_events']}"]
    if s["corr_range"] is not None:
        lines.append(f"  correction factor range: [{s['corr_range'][0]:.3f}, {s['corr_range'][1]:.3f}]")
    lines.append(f"  loading-term flatness (stacked lambda_load [-10,-1] / [-{args.window},-{args.window - 10}]): {res['flatness']:.3f}")
    if res["reservoir"]:
        rs = res["reservoir"]
        lines.append("  reservoir (yr of loading): run mean "
                     f"{np.mean([r['mean'] for r in rs]):.0f} (sd {np.mean([r['sd'] for r in rs]):.0f}); "
                     f"2nd-half mean per run [{', '.join(f'{r['half_mean']:.0f}' for r in rs)}]; "
                     f"min {min(r['min'] for r in rs):.0f}, max {max(r['max'] for r in rs):.0f}; "
                     f"frac(D < {args.d_floor:g}) 2nd half {np.mean([r['frac_below'] for r in rs]):.2f}")
        comp = res["components"]
        lags = comp["lags_yr"]
        dpre = np.nanmean(comp["reservoir"][(lags >= -10) & (lags <= -1)])
        dpost = np.nanmean(comp["reservoir"][(lags >= 11) & (lags <= 30)])
        lines.append(f"  stacked D/D_mean: [-10,-1]={dpre:.2f}, [11,30]={dpost:.2f}")
    cl = res["clipping"]
    lines.append(f"  clipped fraction (M_nom - M > 0.05): all={cl['all']:.3f}, M_nom in [6,7)={cl['6-7']:.3f}, "
                 f"M_nom>=7={cl['7+']:.3f}; pre[-10,-1]={cl['pre']:.3f}, post[11,30]={cl['post']:.3f}")
    host, nhost, edges = res["hosting"]
    labels = [f"{int(edges[i])}-{int(edges[i + 1]) if np.isfinite(edges[i + 1]) else 'inf'}" for i in range(len(edges) - 1)]
    lines.append("  hosting P(M within 0.05 of nominal | band) vs yr since last M>=7: " + ", ".join(labels))
    for b, v in host.items():
        lines.append(f"    M_nom [{b[0]:g},{b[1]:g}): " + ", ".join(f"{x:.2f}" for x in v)
                     + "  (n=" + ",".join(str(int(x)) for x in nhost[b]) + ")")
    for m, od in res["ordering"].items():
        lines.append(f"  ordering share(M>={m:g} | M>={m_t[0]:g}): pre[-10,-1]={od['pre']:.3f} ({od['n_pre_hi']}/{od['n_pre_lo']}), "
                     f"post[11,30]={od['post']:.3f}, global={od['global']:.3f}; pre/global={od['pre_over_global']:.2f} ({od['lo']:.2f}-{od['hi']:.2f})")
    for m in m_t:
        for scope in ("whole", "local"):
            pre = res["ratios"][(m, scope, "pre")]
            pre10 = res["ratios"][(m, scope, "pre10")]
            post = res["ratios"][(m, scope, "post")]
            lines.append(
                f"  M>={m:g} {scope:5s}: R_pre[-{args.window},-1]={pre[0]:.2f} ({pre[1]:.2f}-{pre[2]:.2f}); "
                f"R_pre[-10,-1]={pre10[0]:.2f} ({pre10[1]:.2f}-{pre10[2]:.2f}); "
                f"R_post[1,{args.window}]={post[0]:.2f} ({post[1]:.2f}-{post[2]:.2f})")
            parts = []
            for lo, hi in ((1, 10), (3, 20), (11, 30), (41, 50)):
                r = res["ratios"].get((m, scope, f"post{lo}_{hi}"))
                if r is not None and not np.isnan(r[0]):
                    parts.append(f"[{lo},{hi}]={r[0]:.2f}")
            if parts:
                lines.append(f"      R_post windows: " + ", ".join(parts))
    return "\n".join(lines)


def make_figure(results, labels, args, output):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    m0 = results[0]["m_targets"][0]

    for i, (res, label) in enumerate(zip(results, labels)):
        color = COLORS[i % len(COLORS)]
        # (a) rate ratio
        st = res["stacks"][(m0, "whole")]
        ax_a.fill_between(st["centers"], st["lo"], st["hi"], color=color, alpha=0.2, linewidth=0)
        ax_a.plot(st["centers"], st["ratio"], "-", color=color, linewidth=1.0, label=label)
        stl = res["stacks"][(m0, "local")]
        ax_a.plot(stl["centers"], stl["ratio"], "--", color=color, linewidth=0.8)
        # (b) components
        comp = res["components"]
        ax_b.plot(comp["lags_yr"], comp["loading"], "-", color=color, linewidth=1.0, label=label)
        ax_b.plot(comp["lags_yr"], comp["aftershock"], "--", color=color, linewidth=0.8)
        if np.isfinite(comp["reservoir"]).any():
            ax_b.plot(comp["lags_yr"], comp["reservoir"], ":", color=color, linewidth=0.8)
        # (c) Fano
        fano = res["fano"]
        ax_c.fill_between(res["fano_widths"], np.nanmin(fano, axis=0), np.nanmax(fano, axis=0),
                          color=color, alpha=0.2, linewidth=0)
        ax_c.plot(res["fano_widths"], np.nanmean(fano, axis=0), "-", color=color, linewidth=1.0, label=label)
        # (d) counting noise
        sd = np.nanmean(res["noise_sd"], axis=0)
        ref = np.sqrt(np.nanmean(res["noise_meanN"], axis=0))
        ax_d.plot(res["fano_widths"], sd, "-", color=color, linewidth=1.0, label=label)
        ax_d.plot(res["fano_widths"], ref, ":", color=color, linewidth=0.8)

    W = float(args.window)
    ax_a.axhline(1.0, color="k", linewidth=0.5)
    ax_a.axvline(0.0, color="k", linewidth=0.5)
    ax_a.set_xlim(-W, W)
    ax_a.set_xticks([-W, -W / 2, 0, W / 2, W])
    ax_a.set_xlabel("$t - t_{\\mathrm{mainshock}}$ (years)", fontsize=FONTSIZE)
    ax_a.set_ylabel(f"rate ratio, $M \\geq {m0:g}$", fontsize=FONTSIZE)
    ax_a.legend(loc="upper left", fontsize=FONTSIZE - 2, frameon=False,
                title="solid: fault, dashed: local", title_fontsize=FONTSIZE - 2)

    ax_b.axvline(0.0, color="k", linewidth=0.5)
    ax_b.set_xlim(-W, W)
    ax_b.set_xticks([-W, -W / 2, 0, W / 2, W])
    ax_b.set_xlabel("$t - t_{\\mathrm{mainshock}}$ (years)", fontsize=FONTSIZE)
    ax_b.set_ylabel("$\\lambda$ component / $\\bar{\\lambda}$", fontsize=FONTSIZE)
    ax_b.legend(loc="upper left", fontsize=FONTSIZE - 2, frameon=False,
                title="solid: loading, dashed: Omori, dotted: D/D_mean", title_fontsize=FONTSIZE - 2)

    ax_c.axhline(1.0, color="k", linewidth=0.5)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("window (years)", fontsize=FONTSIZE)
    ax_c.set_ylabel("Fano factor", fontsize=FONTSIZE)
    ax_c.legend(loc="upper left", fontsize=FONTSIZE - 2, frameon=False)

    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("window (years)", fontsize=FONTSIZE)
    ax_d.set_ylabel("sd($N - \\int \\lambda\\,dt$), events", fontsize=FONTSIZE)
    ax_d.legend(loc="upper left", fontsize=FONTSIZE - 2, frameon=False,
                title="dotted: Poisson $\\sqrt{N}$", title_fontsize=FONTSIZE - 2)

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        ax.set_box_aspect(1)
        ax.tick_params(axis="both", labelsize=FONTSIZE, direction="out", length=3, width=0.8)
        ax.text(0.95, 0.97, letter, transform=ax.transAxes, ha="right", va="top", fontsize=FONTSIZE + 1)

    plt.savefig(output, dpi=500, bbox_inches="tight")
    print(f"Saved: {output}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Pre-mainshock seismicity diagnostics")
    parser.add_argument("--input_dir", nargs="+", default=["results/ensemble"],
                        help="One or more directories with ensemble_run_*.h5 files")
    parser.add_argument("--m_mainshock", type=float, default=7.5)
    parser.add_argument("--m_targets", type=str, default="5,6",
                        help="Comma-separated target magnitude thresholds")
    parser.add_argument("--window", type=int, default=50, help="Half-width of the stack (steps)")
    parser.add_argument("--bin", type=int, default=5, help="Lag bin width (steps)")
    parser.add_argument("--local_km", type=float, default=25.0)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed")
    parser.add_argument("--t_min", type=float, default=0.0,
                        help="Ignore mainshocks before this time (spin-up exclusion, years)")
    parser.add_argument("--key", choices=["actual", "nominal"], default="actual",
                        help="Stack on actual mainshock magnitudes or on nominal-draw times")
    parser.add_argument("--d_floor", type=float, default=134.0,
                        help="Reservoir level (yr of loading) below which M7.5 capacity binds")
    parser.add_argument("--output", default=None,
                        help="Figure path (default <first input_dir>/foreshocks.png)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output = Path(args.output) if args.output else Path(args.input_dir[0]) / "foreshocks.png"

    results, labels, summaries = [], [], []
    for d in args.input_dir:
        runs = load_runs(d)
        for r in runs:
            r["t_min"] = args.t_min
            r["key"] = args.key
        res = analyze(runs, args, rng)
        label = Path(d).name
        results.append(res)
        labels.append(label)
        text = summarize(label, res, args)
        summaries.append(text)
        print(text)

    make_figure(results, labels, args, output)
    summary_path = output.with_name(output.stem + "_summary.txt")
    summary_path.write_text("\n\n".join(summaries) + "\n")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
