"""
Experiment: KES with a characteristic magnitude distribution.

Loading-term events draw from a Youngs-Coppersmith-style mixture:
    with prob (1 - p_char): G-R (b = b_value) on [M_min, M_gr_max]
    with prob p_char:       uniform on [M_char_lo, M_char_hi]
so the catalog has a small-event band, an empty gap, and a characteristic
bump. The Omori part keeps the Bath split from experiment_bath_split (each
aftershock is capped at M_parent - dM), so aftershocks of characteristic
events fall below the gap. The moment-balance calibration E[m] is replaced
by the mixture expectation. All monkey-patched; no repository code changes.

Usage:
    python experiment_characteristic.py run --tag char_bath [--p_char 0.005]
        [--gr_max 6.3] [--char_lo 7.2] [--char_hi 7.5] [--n_runs 5] [--duration 3000]
"""

import argparse

import numpy as np

import run_ensemble
import simulator
import temporal_prob
import experiment_bath_split as bath
from config import Config

_orig_expected = temporal_prob.compute_expected_moment_per_event


def mixture_expected_moment(config):
    """E[geometric moment] under the characteristic mixture."""
    from moment import magnitude_to_seismic_moment
    def gr_mean(lo, hi):
        M = np.linspace(lo, hi, 2000)
        p = 10 ** (-config.b_value * M)
        p /= p.sum()
        return float(np.sum(magnitude_to_seismic_moment(M) / config.shear_modulus_Pa * p))
    def box_mean(lo, hi):
        M = np.linspace(lo, hi, 500)
        return float(np.mean(magnitude_to_seismic_moment(M) / config.shear_modulus_Pa))
    p = config._char_p
    return (1.0 - p) * gr_mean(config.M_min, config._char_gr_max) \
        + p * box_mean(config._char_lo, config._char_hi)


def char_draw(config):
    """Split-aware magnitude draw: Bath-capped G-R for Omori events,
    characteristic mixture for loading events."""
    load, parents, rates, om, hist = config._split
    total = load + om
    if om > 0 and total > 0 and np.random.random() < om / total:
        j = int(np.random.choice(parents.size, p=rates / om))
        parent = hist[parents[j]]
        M_cap = min(config.M_max, parent["magnitude"] - config._bath_dM)
        config._current = ("omori", int(parents[j]), parent)
        return bath._draw_truncated(config, config.M_min, M_cap)
    config._current = ("loading", -1, None)
    if np.random.random() < config._char_p:
        return config._char_lo + np.random.random() * (config._char_hi - config._char_lo)
    return bath._draw_truncated(config, config.M_min, config._char_gr_max)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--p_char", type=float, default=0.005)
    r.add_argument("--gr_max", type=float, default=6.3)
    r.add_argument("--char_lo", type=float, default=7.2)
    r.add_argument("--char_hi", type=float, default=7.5)
    r.add_argument("--dM", type=float, default=1.2)
    r.add_argument("--n_runs", type=int, default=5)
    r.add_argument("--duration", type=float, default=3000.0)
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    # Bath-split machinery (rate partition, spatial weighting, origin sidecars)
    bath.install(args.dM, True, 1.0)
    # Characteristic mixture on top
    Config._char_p = args.p_char
    Config._char_gr_max = args.gr_max
    Config._char_lo = args.char_lo
    Config._char_hi = args.char_hi
    simulator.draw_magnitude = char_draw
    temporal_prob.compute_expected_moment_per_event = mixture_expected_moment

    overrides = run_ensemble.parse_overrides(args.set)
    overrides.setdefault("snapshot_interval_years", 10.0)
    # E[m] must be the mixture expectation (capacity clipping is negligible at
    # D_ref = 250 for M <= 7.5, so use the unclipped mixture directly)
    overrides.setdefault("rate_expected_moment", "gr")
    run_ensemble.run_ensemble(
        n_runs=args.n_runs, output_dir=f"results/ensemble_{args.tag}", vary_seed=True,
        base_seed=42, duration=args.duration, overrides=overrides,
    )


if __name__ == "__main__":
    main()
