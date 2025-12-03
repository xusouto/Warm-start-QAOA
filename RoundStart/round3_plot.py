#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def collect_averages(
    data_dir: Path, depth_min: int, depth_max: int, iters: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    depths = list(range(depth_min, depth_max + 1))
    all_prob = np.full((len(depths), iters), np.nan, dtype=float)
    all_ener = np.full((len(depths), iters), np.nan, dtype=float)

    for idx, d in enumerate(depths):
        for i in range(iters):
            fname = data_dir / f"results_d{d}_i{i}.json"
            if not fname.exists():
                continue
            try:
                with open(fname, "r") as f:
                    obj = json.load(f)
                p = obj.get("prob", None)
                e = obj.get("ener_norm", None)
                if p is not None:
                    all_prob[idx, i] = float(p)
                if e is not None:
                    all_ener[idx, i] = float(e)
            except Exception as exc:
                print(f"[WARN] Skipping {fname}: {exc}")
                continue

    return np.array(depths, dtype=int), all_prob, all_ener


def make_plots(
    depths: np.ndarray,
    all_prob: np.ndarray,
    all_ener: np.ndarray,
    out_path: str,
    dpi: int = 150,
):
    # ----- Statistics for probability (in %) -----
    # Work directly in percent for plotting
    avg_prob = np.nanmean(all_prob, axis=1) * 100.0
    med_prob = np.nanmedian(all_prob, axis=1) * 100.0
    q25_prob = np.nanpercentile(all_prob, 25, axis=1) * 100.0
    q75_prob = np.nanpercentile(all_prob, 75, axis=1) * 100.0

    # ----- Statistics for energy -----
    avg_ener = np.nanmean(all_ener, axis=1)
    med_ener = np.nanmedian(all_ener, axis=1)
    q25_ener = np.nanpercentile(all_ener, 25, axis=1)
    q75_ener = np.nanpercentile(all_ener, 75, axis=1)

    # --- Figure & axes: stack vertically, share x-axis, no vertical gap ---
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
    )
    # Make the two axes touch with no whitespace in between
    fig.subplots_adjust(hspace=0.0)

    # Common font sizes
    label_fs = 14
    tick_fs = 12
    legend_fs = 11

    # ---------- Probability vs Depth (TOP) ----------
    ax = axes[0]

    # Small dots: individual runs, scaled to percent
    for idx, d in enumerate(depths):
        vals = all_prob[idx, :] * 100.0  # convert to %
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        ax.scatter(
            np.full_like(vals, d, dtype=float),
            vals,
            s=20,
            alpha=0.4,
            label="_nolegend_",
        )

    # Shaded 25–75% quantile region
    ax.fill_between(
        depths,
        q25_prob,
        q75_prob,
        alpha=0.2,
        label="25–75% quantile",
    )

    # Large dots: medians
    ax.plot(
        depths,
        med_prob,
        marker="o",
        markersize=8,
        linestyle="-",
        label="Median",
    )

    # Mean as dashed line
    ax.plot(
        depths,
        avg_prob,
        linestyle="--",
        linewidth=1.2,
        label="Mean",
    )

    ax.set_ylabel("Prob. Maximum cut (%)", fontsize=label_fs)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", fontsize=legend_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)

    # Remove x tick labels from top plot (shared x with bottom)
    ax.tick_params(labelbottom=False)

    # ---------- Energy vs Depth (BOTTOM) ----------
    ax = axes[1]

    # Small dots: individual runs
    for idx, d in enumerate(depths):
        vals = all_ener[idx, :]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        ax.scatter(
            np.full_like(vals, d, dtype=float),
            vals,
            s=20,
            alpha=0.4,
            label="_nolegend_",
        )

    # Shaded 25–75% quantile region
    ax.fill_between(
        depths,
        q25_ener,
        q75_ener,
        alpha=0.2,
        label="25–75% quantile",
    )

    # Large dots: medians
    ax.plot(
        depths,
        med_ener,
        marker="o",
        markersize=8,
        linestyle="-",
        label="Median",
    )

    # Mean as dashed line
    ax.plot(
        depths,
        avg_ener,
        linestyle="--",
        linewidth=1.2,
        label="Mean",
    )

    ax.set_xlabel("QAOA depth p", fontsize=label_fs)
    ax.set_ylabel("Energy ", fontsize=label_fs)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xticks(depths)
    ax.tick_params(axis="both", labelsize=tick_fs)

    # Reference lines
    ax.axhline(y=-14.5, color="r", linestyle="--", label="initial 001111 cut")
    ax.axhline(y=-18.5, color="g", linestyle="-", label="MAXCUT energy")

    # Legend in bottom-left corner
    ax.legend(loc="lower left", fontsize=legend_fs)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved combined figure to: {out_path}")




def main():
    parser = argparse.ArgumentParser(
        description="Plot probability and energy vs depth from JSON files with quantile shading."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Results3",
        help="Directory containing results_d{d}_i{i}.json files.",
    )
    parser.add_argument(
        "--depth-min", type=int, default=1, help="Minimum depth (inclusive)."
    )
    parser.add_argument(
        "--depth-max", type=int, default=6, help="Maximum depth (inclusive)."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=30,
        help="Number of iterations per depth (files i0..i{iters-1}).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Results3/depth_averages.png",
        help="Output filename for combined figure.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="DPI for saved figures."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    depths, all_prob, all_ener = collect_averages(
        data_dir, args.depth_min, args.depth_max, args.iters
    )

    if np.all(np.isnan(all_prob)) and np.all(np.isnan(all_ener)):
        print("No valid data found. Please check your directory and file pattern.")
        return

    make_plots(depths, all_prob, all_ener, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
