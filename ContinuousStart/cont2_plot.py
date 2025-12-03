import argparse
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

def to_num(v):

    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    if isinstance(v, (list, tuple)) and v:
        return v[0]
    return v

def load_results(dir_path: str, pattern: str):
    paths = sorted(glob.glob(os.path.join(dir_path, pattern)))
    if not paths:
        raise SystemExit(f"No files found matching: {os.path.join(dir_path, pattern)}")

    iterations, c_prods, e_warms, e_colds, deltas = [], [], [], [], []
    bad = []

    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            it  = int(to_num(data["iteration"]))
            cp  = float(to_num(data["cprod"]))
            ew  = float(to_num(data["Ew/E0"]))
            ec  = float(to_num(data["Ec/E0"]))
            dwc = float(to_num(data["Delta_frac"]))
        except Exception as e:
            bad.append((p, str(e)))
            continue

        iterations.append(it)
        c_prods.append(cp)
        e_warms.append(ew)
        e_colds.append(ec)
        deltas.append(dwc)

    if bad:
        print("⚠️ Skipped files:")
        for p, err in bad:
            print(f"  - {p}: {err}")

    order = sorted(range(len(iterations)), key=lambda k: iterations[k])
    iterations = [iterations[i] for i in order]
    c_prods    = [c_prods[i]    for i in order]
    e_warms    = [e_warms[i]    for i in order]
    e_colds    = [e_colds[i]    for i in order]
    deltas     = [deltas[i]     for i in order]

    print(f"Loaded {len(iterations)} files.")
    return iterations, c_prods, e_warms, e_colds, deltas

def main():
    ap = argparse.ArgumentParser(
        description="Plot histogram of e_warm/e_cold and delta_w_c vs c_prod."
    )
    ap.add_argument("--dir", default="Results2",
                    help="Directory containing results files.")
    ap.add_argument("--pattern", default="results_i*.json",
                    help="Glob pattern for results files.")
    ap.add_argument("--bins", type=int, default=20,
                    help="Number of histogram bins.")
    ap.add_argument("--out-hist", default="Results2/hist_e_warm_cold.png",
                    help="Output PNG for full+zoom histogram.")
    ap.add_argument("--out-zoom", default="Results2/hist_e_warm_cold_zoom.png",
                    help="Output PNG for zoomed-only histogram.")
    ap.add_argument("--out-delta", default="Results2/delta_vs_cprod.png",
                    help="Output PNG for delta vs c_prod.")
    ap.add_argument("--out-combo", default="Results2/Histogram_scatter_zoomed.png",
                    help="Output PNG for zoomed histogram (top) + scatter (bottom).")
    ap.add_argument("--zoom-bins", type=int, default=None,
                    help="Bins for the zoomed histogram (0.7–1.0). If omitted, uses ~2× --bins, capped.")
    args = ap.parse_args()

    iters, c_prods, e_warms, e_colds, deltas = load_results(args.dir, args.pattern)

    # Common styling
    warm_color = "#ff7f0e"   # vivid orange
    cold_color = "#1f77b4"   # bright blue
    zoom_bins = args.zoom_bins if args.zoom_bins is not None else min(60, max(30, args.bins * 2))
    zoom_edges = np.linspace(0.7, 1.0, zoom_bins + 1)

    # ----------------------------------------------------------------------
    # 1) Histogram: e_warm & e_cold  (full range + zoomed 0.7–1.0) in one figure
    # ----------------------------------------------------------------------
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    # Full-range histogram
    ax_full.hist(
        [e_warms, e_colds],
        bins=args.bins,
        label=["Warm-start", "Cold-start"],
        alpha=0.65,
        color=[warm_color, cold_color],
        edgecolor="none",          # <-- fixed
        linewidth=0.8
    )
    ax_full.set_xlabel(r"$E_{QAOA}/E_0$")
    ax_full.set_ylabel("Counts")
    ax_full.legend()
    ax_full.set_yscale("log", base=10)
    ax_full.yaxis.set_major_locator(LogLocator(base=10))
    ax_full.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_full.grid(True, alpha=0.3)

    # Zoomed histogram (in the same figure, right panel)
    ax_zoom.hist(
        [e_warms, e_colds],
        bins=zoom_edges,
        label=["Warm-start (zoom)", "Cold-start (zoom)"],
        alpha=0.65,
        color=[warm_color, cold_color],
        edgecolor="none",          # <-- fixed
        linewidth=0.8
    )
    ax_zoom.set_xlabel(r"$E_{QAOA}/E_0$")
    ax_zoom.set_ylabel("Count")
    ax_zoom.set_xlim(0.7, 1.0)
    ax_zoom.set_yscale("log", base=10)
    ax_zoom.yaxis.set_major_locator(LogLocator(base=10))
    ax_zoom.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_zoom.grid(True, alpha=0.3)

    #fig.savefig(args.out_hist, dpi=150)
    #print("Saved:", args.out_hist)

    # ----------------------------------------------------------------------
    # 1b) Zoomed-only histogram in its own figure
    # ----------------------------------------------------------------------
    fig_zoom, ax_zoom_only = plt.subplots(figsize=(5.5, 4.0))  # slightly wider

    ax_zoom_only.hist(
        [e_warms, e_colds],
        bins=zoom_edges,
        label=["Warm-start", "Cold-start"],
        alpha=0.65,
        color=[warm_color, cold_color],
        edgecolor="none",          # <-- fixed
        linewidth=0.8
    )
    ax_zoom_only.set_xlabel(r"$E_{QAOA}/E_0$")
    ax_zoom_only.set_ylabel("Count")
    ax_zoom_only.set_xlim(0.7, 1.0)
    ax_zoom_only.set_yscale("log", base=10)
    ax_zoom_only.yaxis.set_major_locator(LogLocator(base=10))
    ax_zoom_only.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_zoom_only.grid(True, alpha=0.3)
    ax_zoom_only.legend()

    fig_zoom.tight_layout()
    #fig_zoom.savefig(args.out_zoom, dpi=150)
    #print("Saved:", args.out_zoom)

    # ----------------------------------------------------------------------
    # Prepare sorted data for scatter
    # ----------------------------------------------------------------------
    order = sorted(range(len(c_prods)), key=lambda k: c_prods[k])
    x = [c_prods[i] for i in order]
    y = [deltas[i]  for i in order]

    # ----------------------------------------------------------------------
    # 2) delta_w_c vs c_prod (scatter) alone
    # ----------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))  # also a bit wider

    ax2.scatter(
        x, y,
        marker="o",
        color="green",
        alpha=0.6,
        label=r"$\Delta_\mathrm{warm}/\Delta_\mathrm{cold}$"
    )

    ax2.axhline(1.0, linestyle="--", linewidth=1.2, color="gray", label="y=1")

    ax2.set_xlabel(r"$d^{*T} c^*/B$")
    ax2.set_ylabel(r"$\Delta_\mathrm{warm}/\Delta_\mathrm{cold}$")
    ax2.grid(True)
    ax2.legend()

    fig2.tight_layout()
    #fig2.savefig(args.out_delta, dpi=150)
    #print("Saved:", args.out_delta)

    # ----------------------------------------------------------------------
    # 3) Combined figure: zoomed histogram (top) + scatter (bottom)
    #    both a bit elongated in x
    # ----------------------------------------------------------------------
    fig3, (ax3_top, ax3_bottom) = plt.subplots(
        2, 1, figsize=(7.0, 6.5), sharex=False, constrained_layout=True
    )

    # Top: zoomed histogram
    ax3_top.hist(
        [e_warms, e_colds],
        bins=zoom_edges,
        label=["Warm-start", "Cold-start"],
        alpha=0.65,
        color=[warm_color, cold_color],
        edgecolor="none",          # <-- fixed
        linewidth=0.8
    )
    ax3_top.set_ylabel("Count")
    ax3_top.set_xlim(0.7, 1.0)
    ax3_top.set_yscale("log", base=10)
    ax3_top.set_xlabel(r"$E_{QAOA}/E_0$")
    ax3_top.yaxis.set_major_locator(LogLocator(base=10))
    ax3_top.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax3_top.grid(True, alpha=0.3)
    ax3_top.legend()

    # Bottom: scatter
    ax3_bottom.scatter(
        x, y,
        marker="o",
        color="green",
        alpha=0.6,
        label=r"$\Delta_\mathrm{warm}/\Delta_\mathrm{cold}$"
    )
    ax3_bottom.axhline(1.0, linestyle="--", linewidth=1.2, color="gray", label="y=1")
    ax3_bottom.set_xlabel(r"$d^{*T} c^*/B$")
    ax3_bottom.set_ylabel(r"$\Delta_\mathrm{warm}/\Delta_\mathrm{cold}$")
    ax3_bottom.grid(True)
    ax3_bottom.legend()

    fig3.savefig(args.out_combo, dpi=150)
    print("Saved:", args.out_combo)

if __name__ == "__main__":
    main()
