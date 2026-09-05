"""
Lollipop (stem) catalog plot around the largest event of one ensemble run,
plus the raw pre-/post-event counts around the largest events of every run.

Figure: the catalog of the chosen run (--run, or a single file via --input)
in a --window-yr wide window placed so that the run's largest event (by
released magnitude) sits at fraction --pos of the window (clipped to the
catalog; the window start is rounded down to 10 yr so the ticks are round).
Stems from --m_min to M, marker area growing geometrically with M, colour =
M (plasma). When the file carries the Bath-split origin field
(events['origin']: 0 = loading, 1 = Omori child) the Omori children are
drawn as squares with a red edge and a legend is added. The title reports the
in-window counts and the raw pre/post counts defined below.

Lollipop statistic (printed for every run and written to
<input_dir>/lollipop_counts.txt), two selections:

  block   Tile every run into consecutive non-overlapping --window-yr blocks
          [0, W), [W, 2W), ... (floor(duration / W) blocks; a run shorter
          than W is one block). In every block take the largest released
          magnitude; count M >= m events at integer grid lags
          k - k_0 in [-10, -1], [1, 10], [-50, -1] and [1, 50] relative to
          that event (k = round(t / dt_grid) on the step grid, so the event
          itself and same-step co-events at lag 0 are excluded; lags may
          cross the block edges; only the catalog edges truncate a window,
          flagged with '*'). The reported mean is over all blocks of all
          runs. This is the reference selection: for results/ensemble_ref3000
          (5 x 3000 yr, W = 1000, 15 blocks) it gives M >= 5 means of
          16.4 / 13.5 at 10 yr and 82 / 48 at 50 yr (pre / post).
  run     One window per run around the run's largest event (the event the
          figure is drawn around). Selecting the single largest event of a
          long catalog favours epochs of unusually full reservoir, so these
          counts are not comparable with the block statistic (ref3000:
          23.8 / 14.0 at 10 yr).

Counts are given for M >= m_min and M >= 6 and, when events['origin'] is
present, per origin (loading / Omori child). Earlier ad-hoc lollipop
figures (e.g. results/lollipop_method_comparison.png, "9/7") used a
different event selection and are not comparable with either table.

Outputs:
    <input_dir>/lollipop_run{NN}.png   Stem plot (300 dpi)
    <input_dir>/lollipop_counts.txt    Count tables

Usage:
    python plot_lollipop.py --input_dir results/ensemble_ref3000 [--run 0]
                            [--window 1000] [--pos 0.35] [--m_min 5]
                            [--output PATH]
    python plot_lollipop.py --input results/smoke_mu1/ensemble_run_00_seed042.h5
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FONTSIZE = 8
LAG_WINDOWS = ((-10, -1), (1, 10), (-50, -1), (1, 50))
ORIGIN_NAMES = {0: "loading", 1: "omori"}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _is_complete(h5):
    """A run whose simulation finished: events written and a full step grid."""
    if int(h5.attrs.get("n_events", 0)) <= 0:
        return False
    if "step" in h5 and "n_time_steps" in h5["config"].attrs:
        return h5["step"]["times"].shape[0] >= int(h5["config"].attrs["n_time_steps"])
    return True


def load_run(path):
    """Catalog and grid information of one run (None if the run is incomplete)."""
    with h5py.File(path, "r") as h5:
        if not _is_complete(h5):
            return None
        events = h5["events"][...]
        if "step" in h5:
            times = h5["step"]["times"][:]
        else:
            times = h5["times"][:]
        dt_grid = float(times[1] - times[0]) if times.size > 1 else 1.0
        run = {
            "name": Path(path).stem,
            "time": events["time"].astype(float),
            "magnitude": events["magnitude"].astype(float),
            "origin": (events["origin"].astype(int)
                       if "origin" in events.dtype.names else None),
            "dt_grid": dt_grid,
            "n_steps": int(times.size),
            "duration": float(h5["config"].attrs["duration_years"]),
        }
    run["step_index"] = np.round(run["time"] / dt_grid).astype(int)
    return run


def load_runs(input_dir=None, input_file=None):
    if input_file is not None:
        files = [Path(input_file)]
    else:
        files = sorted(Path(input_dir).glob("ensemble_run_*.h5"))
        if not files:
            raise SystemExit(f"No ensemble_run_*.h5 files found in {input_dir}")
    runs = []
    for f in files:
        try:
            run = load_run(f)
        except OSError as exc:
            print(f"Skipping {f.name}: cannot open ({exc})")
            continue
        if run is None:
            print(f"Skipping {f.name}: run incomplete")
            continue
        runs.append(run)
    if not runs:
        raise SystemExit("No complete runs found")
    return runs


# ----------------------------------------------------------------------------
# Counts
# ----------------------------------------------------------------------------
def lag_counts(run, i_ref, m_thr, origin=None):
    """Counts of M >= m_thr events at integer grid lags in LAG_WINDOWS around event i_ref."""
    lag = run["step_index"] - run["step_index"][i_ref]
    sel = run["magnitude"] >= m_thr
    if origin is not None and run["origin"] is not None:
        sel &= run["origin"] == origin
    return [int(((lag >= lo) & (lag <= hi) & sel).sum()) for lo, hi in LAG_WINDOWS]


def truncated(run, i_ref):
    """True when the +/-50-step windows around event i_ref cross a catalog edge."""
    k0 = run["step_index"][i_ref]
    return (k0 - 50 < 0) or (k0 + 50 > run["n_steps"] - 1)


def block_references(run, window):
    """Index of the largest event in each consecutive --window-yr block."""
    n_blocks = max(1, int(np.floor(run["duration"] / window + 1e-9)))
    width = window if run["duration"] >= window else run["duration"]
    refs = []
    for b in range(n_blocks):
        lo, hi = b * width, (b + 1) * width
        sel = (run["time"] >= lo) & (run["time"] < hi) if b < n_blocks - 1 else (run["time"] >= lo)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        refs.append((b, int(idx[np.argmax(run["magnitude"][idx])])))
    return refs


def count_rows(runs, window, m_thresholds, selection):
    """Rows (run, block, i_ref) with the count vectors per threshold and origin."""
    rows = []
    for run in runs:
        if selection == "run":
            refs = [(-1, int(np.argmax(run["magnitude"])))]
        else:
            refs = block_references(run, window)
        for b, i in refs:
            row = {"run": run["name"], "block": b, "t": run["time"][i], "M": run["magnitude"][i],
                   "trunc": truncated(run, i), "counts": {}}
            for m in m_thresholds:
                row["counts"][(m, "all")] = lag_counts(run, i, m)
                if run["origin"] is not None:
                    for o, name in ORIGIN_NAMES.items():
                        row["counts"][(m, name)] = lag_counts(run, i, m, origin=o)
            rows.append(row)
    return rows


def format_table(rows, m_thresholds, origins, selection, window):
    """Fixed-width text table of the counts plus their mean."""
    head = ("block: largest event of each consecutive "
            f"{window:g}-yr block" if selection == "block"
            else "run: largest event of the whole run")
    win = " ".join(f"[{lo},{hi}]" for lo, hi in LAG_WINDOWS)
    lines = [f"selection = {head}",
             "counts at integer grid lags " + win + " (pre10 post10 pre50 post50); '*' = window crosses a catalog edge"]
    for origin in origins:
        cols = "  ".join(f"{'M>=' + format(m, 'g') + ':':7s} {'pre10':>5} {'post10':>6} {'pre50':>6} {'post50':>6}"
                         for m in m_thresholds)
        lines.append(f"  [{origin}]")
        lines.append(f"  {'run':24s} {'blk':>3} {'t (yr)':>8} {'M':>5}   {cols}")
        acc = {m: np.zeros(len(LAG_WINDOWS)) for m in m_thresholds}
        for row in rows:
            parts = []
            for m in m_thresholds:
                c = row["counts"][(m, origin)]
                acc[m] += c
                parts.append(f"{'':7s} {c[0]:5d} {c[1]:6d} {c[2]:6d} {c[3]:6d}")
            blk = "run" if row["block"] < 0 else f"{row['block']:d}"
            flag = "*" if row["trunc"] else " "
            lines.append(f"  {row['run']:24s} {blk:>3} {row['t']:8.1f} {row['M']:5.2f}{flag}  "
                         + "  ".join(parts))
        n = max(len(rows), 1)
        parts = [f"{'':7s} {acc[m][0] / n:5.1f} {acc[m][1] / n:6.1f} {acc[m][2] / n:6.1f} {acc[m][3] / n:6.1f}"
                 for m in m_thresholds]
        lines.append(f"  {'mean over ' + str(len(rows)) + ' windows':24s} {'':>3} {'':>8} {'':>5}   "
                     + "  ".join(parts))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def plot_window(run, window, pos, m_min, output, counts_line):
    t, M = run["time"], run["magnitude"]
    i_max = int(np.argmax(M))
    t_max = t[i_max]
    W = min(window, run["duration"])
    t_lo = np.floor((t_max - pos * W) / 10.0) * 10.0
    t_lo = max(0.0, min(t_lo, run["duration"] - W))
    t_hi = t_lo + W
    sel = (t >= t_lo) & (t <= t_hi) & (M >= m_min)

    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    origin = run["origin"]
    is_child = np.zeros(t.size, bool) if origin is None else origin == 1
    ax.vlines(t[sel], m_min, M[sel], color="0.6", linewidth=0.5, zorder=1)
    size = 8 * 3.2 ** (M - 5)
    kw = dict(cmap="plasma", vmin=5.0, vmax=7.8, linewidths=0.3, zorder=2)
    sel_load = sel & ~is_child
    ax.scatter(t[sel_load], M[sel_load], s=size[sel_load], c=M[sel_load],
               edgecolors="black", marker="o", **kw)
    if is_child.any():
        sel_child = sel & is_child
        ax.scatter(t[sel_child], M[sel_child], s=size[sel_child], c=M[sel_child],
                   edgecolors="#d62728", marker="s", **kw)
        handles = [Line2D([], [], linestyle="", marker="o", markersize=4, markerfacecolor="0.7",
                          markeredgecolor="black", markeredgewidth=0.5, label="loading origin"),
                   Line2D([], [], linestyle="", marker="s", markersize=4, markerfacecolor="0.7",
                          markeredgecolor="#d62728", markeredgewidth=0.7, label="Omori child")]
        ax.legend(handles=handles, loc="upper left", fontsize=FONTSIZE - 2, frameon=False,
                  handletextpad=0.3, borderaxespad=0.3)

    ax.set_xlim(t_lo, t_hi)
    ax.set_xticks(t_lo + np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * W)
    y_hi = 8.0
    ax.set_ylim(m_min, y_hi)
    ax.set_yticks(np.arange(np.ceil(m_min), y_hi + 1e-9, 1.0))
    ax.set_xlabel("$t$ (years)", fontsize=FONTSIZE)
    ax.set_ylabel("$M$", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=FONTSIZE, direction="out", length=3, width=0.8)
    ax.text(0.99, 0.95, "a", transform=ax.transAxes, ha="right", va="top", fontsize=FONTSIZE + 1)

    n_win = int(sel.sum())
    n_7 = int((sel & (M >= 7.0)).sum())
    n_child = int((sel & is_child).sum())
    title = (f"{run['name']}: largest event $M$ = {M[i_max]:.2f} at $t$ = {t_max:.0f} yr; "
             f"window [{t_lo:.0f}, {t_hi:.0f}] yr, {n_win} events $M \\geq {m_min:g}$, {n_7} $M \\geq 7$"
             + (f", {n_child} Omori children" if origin is not None else ""))
    ax.set_title(title + "\n" + counts_line, fontsize=FONTSIZE - 1, loc="left")

    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved: {output}")
    return fig


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Lollipop catalog plot around the largest event")
    parser.add_argument("--input_dir", default="results/ensemble",
                        help="Directory with ensemble_run_*.h5 files")
    parser.add_argument("--input", default=None, help="Single HDF5 run (overrides --input_dir)")
    parser.add_argument("--run", type=int, default=0, help="Index of the run to draw (sorted file order)")
    parser.add_argument("--window", type=float, default=1000.0, help="Window length (yr)")
    parser.add_argument("--pos", type=float, default=0.35,
                        help="Fraction of the window at which the largest event sits")
    parser.add_argument("--m_min", type=float, default=5.0, help="Lower magnitude of the stems and counts")
    parser.add_argument("--output", default=None,
                        help="Figure path (default <input_dir>/lollipop_run{NN}.png)")
    args = parser.parse_args()

    runs = load_runs(args.input_dir, args.input)
    out_dir = Path(args.input).parent if args.input else Path(args.input_dir)
    if not 0 <= args.run < len(runs):
        raise SystemExit(f"--run {args.run} out of range (0-{len(runs) - 1})")
    run = runs[args.run]
    output = Path(args.output) if args.output else out_dir / f"lollipop_run{args.run:02d}.png"

    m_thresholds = [args.m_min, 6.0] if args.m_min != 6.0 else [6.0]
    origins = ["all"] + (list(ORIGIN_NAMES.values()) if runs[0]["origin"] is not None else [])
    tables = []
    for selection in ("block", "run"):
        rows = count_rows(runs, args.window, m_thresholds, selection)
        tables.append(format_table(rows, m_thresholds, origins, selection, args.window))
    text = "\n\n".join(tables)
    print(text)
    counts_path = out_dir / "lollipop_counts.txt"
    counts_path.write_text(text + "\n")
    print(f"Saved: {counts_path}")

    # Counts of the drawn run (run selection) for the title
    i_max = int(np.argmax(run["magnitude"]))
    c5 = lag_counts(run, i_max, args.m_min)
    c6 = lag_counts(run, i_max, 6.0)
    counts_line = (f"$M \\geq {args.m_min:g}$ counts [-10,-1] / [1,10]: {c5[0]} / {c5[1]}, "
                   f"[-50,-1] / [1,50]: {c5[2]} / {c5[3]}; "
                   f"$M \\geq 6$: {c6[0]} / {c6[1]}, {c6[2]} / {c6[3]}")
    if run["origin"] is not None:
        co = lag_counts(run, i_max, args.m_min, origin=1)
        counts_line += f" (Omori children $M \\geq {args.m_min:g}$: {co[0]} / {co[1]}, {co[2]} / {co[3]})"
    plot_window(run, args.window, args.pos, args.m_min, output, counts_line)


if __name__ == "__main__":
    main()
