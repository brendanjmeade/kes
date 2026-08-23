"""
Animate a single ensemble run in the style of moment_snapshot_tXXXX.png

Three-panel animation based on visualize.py:plot_moment_snapshots (MODE 2):
- Upper panel: moment deficit rate (change per snapshot interval), magma
- Middle panel: cumulative moment deficit (loading - release), coolwarm
- Lower panel: magnitude-time lollipop plot (plot_ensemble.py style) with a
  gray cursor marking the current time

Unlike the static snapshots (which autoscale each frame), the animation uses
fixed global color limits so colors are comparable across time.

Usage:
    python animate_ensemble_run.py --input results/ensemble/ensemble_run_00_seed042.h5
    python animate_ensemble_run.py --input ... --fps 30 --dpi 200 --stride 1
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

FONTSIZE = 8


def main():
    parser = argparse.ArgumentParser(
        description="Animate moment deficit evolution for one ensemble run"
    )
    parser.add_argument(
        "--input",
        default="results/ensemble/ensemble_run_00_seed042.h5",
        help="Path to ensemble run HDF5 file",
    )
    parser.add_argument("--output", default=None, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument(
        "--stride", type=int, default=1, help="Use every Nth snapshot"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_moment_evolution.mp4"
    else:
        output_path = Path(args.output)

    with h5py.File(input_path, "r") as f:
        times = f["times"][:]
        release_snapshots = f["release_snapshots"][:]
        afterslip_snapshots = (
            f["afterslip_snapshots"][:] if "afterslip_snapshots" in f else None
        )
        slip_rate = f["slip_rate"][:]
        events = f["events"][:]
        cfg = dict(f["config"].attrs)

    n_along_strike = int(cfg["n_along_strike"])
    n_down_dip = int(cfg["n_down_dip"])
    element_size_km = np.sqrt(float(cfg["element_area_m2"])) / 1000.0
    fault_length_km = n_along_strike * element_size_km
    fault_depth_km = n_down_dip * element_size_km

    length_vec = np.linspace(0, fault_length_km, n_along_strike)
    depth_vec = np.linspace(0, fault_depth_km, n_down_dip)
    length_grid, depth_grid = np.meshgrid(length_vec, depth_vec)

    # Total cumulative release (coseismic + afterslip), matching plot_moment_snapshots
    total_release = release_snapshots.copy()
    if afterslip_snapshots is not None:
        total_release += afterslip_snapshots

    # Deficit rate: change between consecutive snapshots (first frame = current value)
    delta_release = np.empty_like(total_release)
    delta_release[0] = total_release[0]
    delta_release[1:] = np.diff(total_release, axis=0)

    # Cumulative deficit: loading - release, compressed with a signed
    # cube root so large late-time deficits don't wash out early structure
    deficit = slip_rate[None, :] * times[:, None] - total_release
    deficit = np.sign(deficit) * np.abs(deficit) ** (1.0 / 3.0)

    # Fixed global color limits. The rate panel is scaled to the median
    # per-frame peak so typical events fill the colormap and the largest
    # events saturate; the deficit panel clips only the extreme tail.
    per_frame_peak = np.abs(delta_release[1:]).max(axis=1)
    delta_max = np.median(per_frame_peak[per_frame_peak > 0])
    deficit_max = np.max(np.abs(deficit))
    delta_levels = np.linspace(0, delta_max, 21)
    deficit_levels = np.linspace(-deficit_max, deficit_max, 21)

    frame_indices = np.arange(0, len(times), args.stride)
    print(f"Input: {input_path}")
    print(f"Frames: {len(frame_indices)}  ({args.fps} fps -> "
          f"{len(frame_indices) / args.fps:.0f} s movie)")
    print(f"Rate scale: +/-{delta_max:.3f} m^3/yr, "
          f"deficit scale: +/-{deficit_max**3:.2f} m^3 (cube-root compressed)")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 4.0))

    # Static magnitude-time lollipop panel (plot_ensemble.py style)
    event_times = events["time"]
    event_mags = events["magnitude"]
    for t, m in zip(event_times, event_mags):
        if m >= 6.0:
            ax3.plot([t, t], [5.0, m], "-k", linewidth=0.25, zorder=1)
    ax3.scatter(
        event_times,
        event_mags,
        c=event_mags,
        cmap="plasma",
        s=1e-8 * event_mags**12.0,
        alpha=1.0,
        edgecolors="black",
        linewidth=0.25,
        zorder=10,
        vmin=5,
        vmax=8,
    )
    ax3.set_xlabel("$t$ (years)", fontsize=FONTSIZE)
    ax3.set_ylabel("$M$", fontsize=FONTSIZE)
    ax3.set_xlim(0, float(cfg["duration_years"]))
    ax3.set_ylim([5, 8])
    ax3.set_yticks([5, 6, 7, 8])
    ax3.tick_params(axis="both", labelsize=FONTSIZE)
    time_cursor = ax3.axvline(
        times[frame_indices[0]], color="0.5", linewidth=0.75, alpha=0.7, zorder=20
    )

    def draw_frame(idx):
        actual_time = times[idx]
        time_cursor.set_xdata([actual_time, actual_time])

        ax1.clear()
        ax2.clear()

        delta_scaled = delta_release[idx].reshape(n_along_strike, n_down_dip).T
        cf1 = ax1.contourf(
            length_grid,
            depth_grid,
            delta_scaled,
            cmap="magma",
            levels=delta_levels,
            extend="max",
        )
        ax1.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
        ax1.set_title(
            f"$\\dot{{{{m}}}}_\\mathrm{{{{d}}}}$ ($t$ = {actual_time:.0f})",
            fontsize=FONTSIZE,
        )
        ax1.invert_yaxis()
        ax1.set_yticks([0, 25])
        ax1.set_xticklabels([])
        ax1.tick_params(axis="both", labelsize=FONTSIZE)
        ax1.set_aspect("equal", adjustable="box")

        deficit_scaled = deficit[idx].reshape(n_along_strike, n_down_dip).T
        cf2 = ax2.contourf(
            length_grid,
            depth_grid,
            deficit_scaled,
            cmap="coolwarm",
            levels=deficit_levels,
            extend="both",
        )
        ax2.set_xlabel("$x$ (km)", fontsize=FONTSIZE)
        ax2.set_ylabel("$d$ (km)", fontsize=FONTSIZE)
        ax2.set_title(
            f"$m_\\mathrm{{{{d}}}}$ ($t$ = {actual_time:.0f})", fontsize=FONTSIZE
        )
        ax2.invert_yaxis()
        ax2.set_yticks([0, 25])
        ax2.tick_params(axis="both", labelsize=FONTSIZE)
        ax2.set_aspect("equal", adjustable="box")

        return cf1, cf2

    # Draw first frame to attach colorbars and fix the layout
    cf1, cf2 = draw_frame(frame_indices[0])
    cbar1 = plt.colorbar(cf1, ax=ax1)
    cbar1.set_label("$\\dot{{{m}}}$ (m$^3$ / year)", fontsize=FONTSIZE - 2)
    cbar1.ax.tick_params(labelsize=FONTSIZE - 2)
    cbar1.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{x:.2f}")
    )
    # Colorbar axis is sign(m)|m|^(1/3): place ticks at round physical values
    # so their nonuniform spacing shows the compression
    cbar2 = plt.colorbar(cf2, ax=ax2)
    cbar2.set_label("$m$ (m$^{3}$, cube-root scale)", fontsize=FONTSIZE - 2)
    cbar2.ax.tick_params(labelsize=FONTSIZE - 2)
    tick_vals = np.array([-8.0, -1.0, 0.0, 1.0, 8.0])
    tick_vals = tick_vals[np.abs(tick_vals) ** (1.0 / 3.0) <= deficit_max]
    cbar2.set_ticks(np.sign(tick_vals) * np.abs(tick_vals) ** (1.0 / 3.0))
    cbar2.set_ticklabels([f"{v:g}" for v in tick_vals])
    plt.tight_layout()

    # Match the lollipop panel's size to the (aspect-adjusted, colorbar-
    # narrowed) field panels above it, anchored to the top of its slot
    fig.canvas.draw()
    pos1 = ax1.get_position()
    pos3 = ax3.get_position()
    ax3.set_position(
        [pos1.x0, pos3.y1 - pos1.height, pos1.width, pos1.height]
    )

    writer = FFMpegWriter(
        fps=args.fps,
        codec="h264",
        extra_args=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"],
    )
    with writer.saving(fig, str(output_path), dpi=args.dpi):
        for idx in tqdm(frame_indices, desc="Rendering"):
            draw_frame(idx)
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
