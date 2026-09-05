"""
Bath-split experiment: split the rate into a moment-deficit part and an Omori
part, and limit the sizes of Omori-generated events by Bath's law.

The split now lives in the library (config keys omori_split_enabled,
omori_bath_dM, omori_bath_K_scale; see CLAUDE.md), so `run` is a thin wrapper
around run_ensemble with those overrides and the origin / parent of every
event is stored in the HDF5 `events` dataset (fields `origin`, `parent_idx`).
`analyze` reads those fields and falls back to the legacy <run>_origins.npz
sidecar for ensembles produced by the old monkey-patched version.

The monkey-patch machinery (install / patched_*) is kept only because
experiment_characteristic.py builds on it; it reproduces the pre-library
split (uncorrected moment budget) and should not be used for new runs.

Usage:
    python experiment_bath_split.py run  --tag split_bath12 --dM 1.2 [--K_scale 1.0 | --restore_branching]
                                         [--n_runs 5] [--duration 3000] [--set key=value ...]
    python experiment_bath_split.py analyze --tags split_ref tilt_mu1_k0 [...]

Output: results/ensemble_<tag>/ (standard HDF5 with origin / parent_idx).
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

import run_ensemble
import simulator
from temporal_prob import omori_lag_steps, omori_productivity, omori_step_rate

_orig_rate = simulator.earthquake_rate
_orig_draw = simulator.draw_magnitude
_orig_spatial = simulator.spatial_probability
_orig_weights = simulator.compute_aftershock_spatial_weights
_orig_append = simulator.append_event
_orig_run = simulator.run_simulation


# ----------------------------------------------------------------------------
# Patched pieces
# ----------------------------------------------------------------------------
def _draw_truncated(config, M_lo, M_hi):
    b = config.b_value
    denom = 10 ** (-b * M_lo) - 10 ** (-b * M_hi)
    u = np.random.random()
    return -np.log10(10 ** (-b * M_lo) - u * denom) / b


def patched_rate(m_current, event_history, current_time, config,
                 cumulative_loading, cumulative_release, dt_years=None, memory_h=None):
    lam, comp = _orig_rate(m_current, event_history, current_time, config,
                           cumulative_loading, cumulative_release,
                           dt_years=dt_years, memory_h=memory_h)
    dt = dt_years if dt_years is not None else config.time_step_years
    parents, rates = [], []
    if config.omori_enabled and event_history:
        k_max = omori_lag_steps(config, dt)
        cap_on, dM = config._bath_cap, config._bath_dM
        for idx, ev in enumerate(event_history):
            lag = int(round((current_time - ev["time"]) / dt))
            if not (1 <= lag <= k_max):
                continue
            if cap_on and ev["magnitude"] - dM < config.M_min:
                continue  # cannot produce an event above M_min
            K = omori_productivity(ev["magnitude"], config) * config._bath_K_scale
            rates.append(omori_step_rate(K, lag, dt, config))
            parents.append(idx)
    om = float(np.sum(rates)) if rates else 0.0
    lam_new = max(0.0, lam - comp["aftershock"] + om)
    comp = dict(comp)
    comp["aftershock"] = om
    comp["n_active_sequences"] = len(parents)
    load = comp["loading"] + comp["background"] + comp["perturbation"]
    config._split = (load, np.array(parents, dtype=int), np.array(rates), om, event_history)
    return lam_new, comp


def patched_draw(config):
    load, parents, rates, om, hist = config._split
    total = load + om
    if om > 0 and total > 0 and np.random.random() < om / total:
        j = int(np.random.choice(parents.size, p=rates / om))
        parent = hist[parents[j]]
        M_cap = config.M_max
        if config._bath_cap:
            M_cap = min(config.M_max, parent["magnitude"] - config._bath_dM)
        config._current = ("omori", int(parents[j]), parent)
        return _draw_truncated(config, config.M_min, M_cap)
    config._current = ("loading", -1, None)
    return _orig_draw(config)


def patched_spatial(m_current, magnitude, config, aftershock_weights=None):
    kind, idx, parent = getattr(config, "_current", ("loading", -1, None))
    if kind == "omori" and parent is not None and parent.get("spatial_activation") is not None:
        phi = np.asarray(parent["spatial_activation"], dtype=float)
        w = phi if aftershock_weights is None else aftershock_weights * phi
        return _orig_spatial(m_current, magnitude, config, w)
    return _orig_spatial(m_current, magnitude, config, aftershock_weights)


def patched_weights(event_history, current_time, config):
    # Loading events carry no aftershock weighting; Omori events get Phi_parent
    return np.ones(config.n_elements), 0


def patched_append(h5file, event):
    kind, idx, _ = getattr(event_config, "_current", ("loading", -1, None))
    event_config._origins.append((0 if kind == "loading" else 1, idx))
    return _orig_append(h5file, event)


event_config = None


def patched_run(config):
    global event_config
    event_config = config
    config._origins = []
    config._current = ("loading", -1, None)
    config._split = (0.0, np.zeros(0, int), np.zeros(0), 0.0, [])
    results = _orig_run(config)
    origins = np.array(config._origins, dtype=int).reshape(-1, 2)
    out = Path(config.output_dir) / (Path(config.output_hdf5).stem + "_origins.npz")
    np.savez(out, origin=origins[:, 0], parent=origins[:, 1])
    return results


def install(dM, cap, K_scale):
    simulator.earthquake_rate = patched_rate
    simulator.draw_magnitude = patched_draw
    simulator.spatial_probability = patched_spatial
    simulator.compute_aftershock_spatial_weights = patched_weights
    simulator.append_event = patched_append
    run_ensemble.run_simulation = patched_run
    from config import Config
    Config._bath_dM = dM
    Config._bath_cap = cap
    Config._bath_K_scale = K_scale


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def eligible_productivity_share(config_cls, dM):
    """Share of G-R-averaged Omori productivity from parents with M - dM >= M_min."""
    M = np.linspace(config_cls.M_min, config_cls.M_max, 2000)
    p = 10 ** (-config_cls.b_value * M)
    K = 10 ** (config_cls.omori_alpha * (M - config_cls.omori_M_ref))
    return float(np.sum(p * K * (M - dM >= config_cls.M_min)) / np.sum(p * K))


def load_origins(h5file, filepath):
    """(origin, parent) arrays from the events dataset, or the legacy npz sidecar."""
    ev = h5file["events"]
    if "origin" in ev.dtype.names:
        rec = ev[...]
        return rec["origin"].astype(int), rec["parent_idx"].astype(int)
    orig_file = Path(filepath).with_name(Path(filepath).stem + "_origins.npz")
    if not orig_file.exists():
        return None, None
    o = np.load(orig_file)
    return o["origin"], o["parent"]


def analyze(tags, m_main=7.0):
    for tag in tags:
        files = sorted(glob.glob(f"results/ensemble_{tag}/ensemble_run_*.h5"))
        n_tot = n_om = n_big = n_big_om = 0
        bath_gaps = []
        for f in files:
            h = h5py.File(f, "r")
            ev = h["events"][...]
            m = ev["magnitude"]
            origin, parent = load_origins(h, f)
            if origin is None:
                continue
            n_tot += m.size
            n_om += int((origin == 1).sum())
            big = m >= m_main
            n_big += int(big.sum())
            n_big_om += int((big & (origin == 1)).sum())
            # Bath check: largest direct child per parent
            children = {}
            for i in np.flatnonzero(origin == 1):
                children.setdefault(int(parent[i]), []).append(m[i])
            for p_idx, kids in children.items():
                if 0 <= p_idx < m.size:
                    bath_gaps.append(m[p_idx] - max(kids))
        if n_tot == 0:
            print(f"[{tag}] no origin information (not a split run)")
            continue
        gaps = np.array(bath_gaps)
        print(f"[{tag}] events {n_tot}; Omori-origin {n_om / n_tot:.3f}; "
              f"M>={m_main:g} events {n_big}, of which Omori-origin {n_big_om} ({n_big_om / max(n_big, 1):.2f}); "
              f"parent - largest child: median {np.median(gaps):.2f}, 10th pct {np.percentile(gaps, 10):.2f}, "
              f"negative (child > parent) {np.mean(gaps < 0):.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--dM", type=float, default=1.2)
    r.add_argument("--K_scale", type=float, default=1.0,
                   help="Multiplier on the Omori productivity of eligible parents")
    r.add_argument("--restore_branching", action="store_true",
                   help="Scale K so eligible productivity equals the uncapped total")
    r.add_argument("--n_runs", type=int, default=5)
    r.add_argument("--duration", type=float, default=3000.0)
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = sub.add_parser("analyze")
    a.add_argument("--tags", nargs="+", required=True)
    a.add_argument("--m_main", type=float, default=7.0)
    args = ap.parse_args()

    if args.cmd == "analyze":
        analyze(args.tags, args.m_main)
        return

    from config import Config
    K_scale = args.K_scale
    if args.restore_branching:
        K_scale = 1.0 / eligible_productivity_share(Config, args.dM)
        print(f"K scaled by {K_scale:.3f} to restore the branching ratio")
    overrides = run_ensemble.parse_overrides(args.set)
    overrides.setdefault("snapshot_interval_years", 10.0)
    overrides["omori_split_enabled"] = True
    overrides["omori_bath_dM"] = float(args.dM)
    overrides["omori_bath_K_scale"] = float(K_scale)
    run_ensemble.run_ensemble(
        n_runs=args.n_runs, output_dir=f"results/ensemble_{args.tag}", vary_seed=True,
        base_seed=42, duration=args.duration, overrides=overrides,
    )


if __name__ == "__main__":
    main()
