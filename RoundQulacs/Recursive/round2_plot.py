import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_ratios(results_path: Path, cases=None):

    if cases is None:
        cases = {"WSQAOA", "CSQAOA", "GW", "RGW"}

    with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    out = {}
    for c in cases:
        entry = data.get(c, None)
        if not isinstance(entry, dict):
            continue
        r = entry.get("approx_ratio", None)
        if r is None:
            continue
        try:
            out[c] = float(r)
        except (TypeError, ValueError):
            continue
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot histogram of approx_ratio over iterations for 4 cases")
    parser.add_argument("--n", type=int, required=True, help="Problem size n")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations (default 100, iter=0..iters-1)")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing results_{n}_{iter}.json")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    parser.add_argument("--out", type=str, default=None, help="If set, save figure to this path (png/pdf/etc)")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    cases = ["WSQAOA", "CSQAOA", "GW", "RGW"]

    ratios = defaultdict(list)

    missing_files = 0
    for it in range(args.iters):
        p = base_dir / f"Results2/results_{args.n}_{it}.json"
        one = load_ratios(p, set(cases))
        if not one:
            if not p.exists():
                missing_files += 1
            continue
        for c, r in one.items():
            ratios[c].append(r)

    # Print a quick summary
    print("Collected approx_ratio counts:", flush=True)
    for c in cases:
        print(f"  {c}: {len(ratios[c])}", flush=True)
    if missing_files:
        print(f"Missing results files: {missing_files}", flush=True)

    # Plot
    plt.figure()
    all_vals = [v for c in cases for v in ratios[c]]
    if len(all_vals) == 0:
        raise RuntimeError("No approx_ratio values found. Check paths / JSON contents.")

    vmin = min(all_vals)
    vmax = max(all_vals)
    # Expand a touch to avoid edge effects
    pad = 1e-6
    bin_edges = np.linspace(vmin - pad, vmax + pad, args.bins + 1)

    for c in cases:
        if len(ratios[c]) == 0:
            continue
        plt.hist(ratios[c], bins=bin_edges, alpha=0.5, label=f"{c} (n={len(ratios[c])})")

    plt.xlabel("approx_ratio = cut_value / maxcut_value")
    plt.ylabel("count (iterations)")
    plt.title(f"Approximation ratios over iterations (n={args.n}, iters=0..{args.iters-1})")
    plt.legend()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path}", flush=True)
    else:
        plt.show()


if __name__ == "__main__":
    main()
