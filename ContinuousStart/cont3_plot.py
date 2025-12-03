import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

def load_grid(data_dir: Path, t_values: List[int], n_values: List[int]) -> Tuple[np.ndarray, np.ndarray, int]:

    e_ann = np.full((len(n_values), len(t_values)), np.nan, dtype=float)
    e_ann_es = np.full((len(n_values), len(t_values)), np.nan, dtype=float)
    count = 0

    for t_idx, T in enumerate(t_values):
        for n_idx, N in enumerate(n_values):
            fname = data_dir / f"results_T{T}_N{N}.json"
            if not fname.exists():
                continue
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    data = json.load(f)

                e1 = data.get("e_ann", None)
                e2 = data.get("e_ann_es", None)

                if e1 is not None:
                    e_ann[n_idx, t_idx] = float(e1)
                if e2 is not None:
                    e_ann_es[n_idx, t_idx] = float(e2)
                count += 1
            except Exception as exc:
                print(f"[WARN] Failed to parse {fname}: {exc}")
                continue
    return e_ann, e_ann_es, count

def normalize_shared_01(grid: np.ndarray, gmin: float, gmax: float) -> np.ndarray:

    norm = np.array(grid, dtype=float, copy=True)
    mask = np.isnan(norm)
    if np.isfinite(gmin) and np.isfinite(gmax) and gmax > gmin:
        norm = (norm - gmin) / (gmax - gmin)
    else:
        norm[:] = 0.5
    norm = np.clip(norm, 0.0, 1.0)
    norm[mask] = np.nan
    return norm

def plot_heatmap(ax,
                 grid: np.ndarray,
                 title: str,  # kept for compatibility but unused
                 t_values: List[int],
                 n_values: List[int],
                 vmin=0.0,
                 vmax=1.0,
                 gamma: float = 0.8,
                 colorbar_label="Normalized energy"):

    masked = np.ma.array(grid, mask=np.isnan(grid))

    # colormap with explicit color for NaNs
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    # map indices to actual T/N values
    xmin, xmax = t_values[0], t_values[-1]
    ymin, ymax = n_values[0], n_values[-1]

    imshow_kwargs = dict(
        origin="lower",
        aspect="auto",
        extent=[xmin, xmax, ymin, ymax],
        interpolation="nearest",
        cmap=cmap,
    )

    if gamma is not None and gamma != 1.0:
        imshow_kwargs["norm"] = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    else:
        imshow_kwargs["vmin"] = vmin
        imshow_kwargs["vmax"] = vmax

    im = ax.imshow(masked, **imshow_kwargs)

    # Axis labels renamed
    ax.set_xlabel("Total anneal time T")
    ax.set_ylabel("Number of steps N")

    # No per-axes title anymore
    # ax.set_title(title)

    # Make ticks readable
    def sparse_ticks(vals, max_ticks=8):
        if len(vals) <= max_ticks:
            return vals
        step = max(1, len(vals) // max_ticks)
        return [vals[i] for i in range(0, len(vals), step)]

    ax.set_xticks(sparse_ticks(t_values))
    ax.set_yticks(sparse_ticks(n_values))

    # No colorbar in this function; caller handles it.
    return im

def main():
    parser = argparse.ArgumentParser(description="Plot e_ann and e_ann_es heatmaps from JSON results.")
    parser.add_argument("--data-dir", type=str,
                        default="/mnt/netapp1/Store_CESGA/home/cesga/jsouto/WS_GitHub/ContinuousStart/Results3",
                        help="Directory containing results_T*_N*.json files.")
    parser.add_argument("--t-max", type=int, default=58, help="Maximum T value (inclusive).")
    parser.add_argument("--n-max", type=int, default=97, help="Maximum N value (inclusive).")
    parser.add_argument("--out", type=str, default="Results3/Heatmaps.png",
                        help="Output filename for combined figure (PNG).")
    parser.add_argument("--individual", type=str, nargs="?", const="PNG",
                        help="Also save individual figures for each map. Optional arg chooses format (e.g., PNG, PDF, SVG).")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figures.")
    parser.add_argument("--gamma", type=float, default=0.8,
                        help="Gamma for contrast (PowerNorm). <1 brightens lows / darkens highs; >1 does the opposite.")
    args = parser.parse_args()

    # The job script sweeps T = 1..61 step 3, N = 1..81 step 4
    t_values = list(range(1, args.t_max + 1, 3))
    n_values = list(range(1, args.n_max + 1, 6))

    data_dir = Path(args.data_dir)
    e_ann, e_ann_es, count = load_grid(data_dir, t_values, n_values)

    if count == 0:
        print("No JSON files found that match the expected pattern. Nothing to plot.")
        return

    # Global min/max across both grids (ignoring NaNs)
    all_vals = np.concatenate([
        e_ann[~np.isnan(e_ann)],
        e_ann_es[~np.isnan(e_ann_es)]
    ]) if (np.any(~np.isnan(e_ann)) or np.any(~np.isnan(e_ann_es))) else np.array([])

    if all_vals.size == 0:
        print("All values are NaN. Nothing to plot.")
        return

    gmin = float(np.min(all_vals))
    gmax = float(np.max(all_vals))

    if not np.isfinite(gmin) or not np.isfinite(gmax):
        print("Global min/max are not finite. Nothing to plot.")
        return

    print(f"[INFO] Shared normalization range: min={gmin:.6g}, max={gmax:.6g}")

    # Normalize both grids to [0,1] with the shared range
    e_ann_norm = normalize_shared_01(e_ann, gmin, gmax)
    e_ann_es_norm = normalize_shared_01(e_ann_es, gmin, gmax)

    # Combined figure (normalized) with a SINGLE shared colorbar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = plot_heatmap(axes[0], e_ann_norm, "e_ann (normalized)",
                       t_values, n_values, vmin=0.0, vmax=1.0, gamma=args.gamma)
    im1 = plot_heatmap(axes[1], e_ann_es_norm, "e_ann_es (normalized)",
                       t_values, n_values, vmin=0.0, vmax=1.0, gamma=args.gamma)

    # Single colorbar for both subplots
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Normalized energy")

    # No suptitle
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined figure to: {args.out}")

    # Optionally save individual normalized figures, with their own colorbar
    if args.individual:
        fmt = args.individual.lower().lstrip(".")
        for name, grid in [("e_ann", e_ann_norm), ("e_ann_es", e_ann_es_norm)]:
            fig_i, ax_i = plt.subplots(figsize=(6, 5), constrained_layout=True)
            im_i = plot_heatmap(ax_i, grid, f"{name} (normalized)",
                                t_values, n_values, vmin=0.0, vmax=1.0, gamma=args.gamma)
            cbar_i = fig_i.colorbar(im_i, ax=ax_i, fraction=0.046, pad=0.04)
            cbar_i.set_label("Normalized energy")
            out_name = f"{name}.{fmt}"
            fig_i.savefig(out_name, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig_i)
            print(f"Saved {name} (normalized) to: {out_name}")

if __name__ == "__main__":
    main()
