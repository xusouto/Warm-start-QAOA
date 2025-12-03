import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Average Energy vs eps: first within-graph (over cuts), then across graphs.")
    p.add_argument("--folder", type=Path, required=True,
                   help="Folder containing result JSON files.")
    p.add_argument("--pattern", type=str, default="results_g*_c*.json",
                   help="Glob pattern to find files (default: results_g*_c*.json).")
    p.add_argument("--png", type=Path, default=Path("Results1/energy_vs_eps.png"),
                   help="Output PNG path (default: energy_vs_eps.png).")
    p.add_argument("--csv", type=Path, default=Path("Results1/averaged_energy.csv"),
                   help="Output CSV path (default: averaged_energy.csv).")
    p.add_argument("--with-error", choices=["none", "std", "stderr"], default="none",
                   help="If set to 'std', shade ±1σ (across graphs); if 'stderr', shade standard error across graphs; default: none.")
    p.add_argument("--tolerance", type=float, default=1e-12,
                   help="Numerical tolerance for comparing eps arrays across files.")
    return p.parse_args()

def find_files(folder: Path, pattern: str) -> List[Path]:
    files = sorted(folder.glob(pattern))
    return [f for f in files if f.is_file()]

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def equal_eps(eps_ref: np.ndarray, eps_other: np.ndarray, tol: float) -> bool:
    if len(eps_ref) != len(eps_other):
        return False
    return np.allclose(np.asarray(eps_ref, dtype=float), np.asarray(eps_other, dtype=float), atol=tol, rtol=0)

def collect_grouped(files: List[Path], tol: float) -> Tuple[np.ndarray, Dict[int, List[np.ndarray]], Dict[int, List[int]]]:

    eps_ref = None
    by_graph: Dict[int, List[np.ndarray]] = {}
    cuts_by_graph: Dict[int, List[int]] = {}

    # Parse "g" and "c" from filenames like results_g12_c3.json
    fname_re = re.compile(r"g(?P<g>\d+)_c(?P<c>\d+)\.json$", re.IGNORECASE)

    for f in files:
        data = read_json(f)
        eps = np.asarray(data.get("eps", []), dtype=float)
        e = np.asarray(data.get("Energies", []), dtype=float)
        if eps.size == 0 or e.size == 0:
            print(f"[WARN] Skipping {f.name}: missing 'eps' or 'Energies'.")
            continue
        if eps.shape != e.shape:
            print(f"[WARN] Skipping {f.name}: len(eps) != len(Energies).")
            continue

        if eps_ref is None:
            eps_ref = eps
        else:
            if not equal_eps(eps_ref, eps, tol):
                print(f"[ERROR] eps grid mismatch in {f.name}. Aborting.")
                raise ValueError(f"eps arrays differ (try adjusting --tolerance?): {f.name}")

        # Determine graph id and cut id
        g_id = None
        c_id = None

        m = fname_re.search(f.name)
        if m:
            g_id = int(m.group("g"))
            c_id = int(m.group("c"))

        # Fallbacks from JSON if needed
        if g_id is None:
            g_val = data.get("graph")
            if isinstance(g_val, int):
                g_id = g_val
        if c_id is None:
            c_val = data.get("cut")
            if isinstance(c_val, int):
                c_id = c_val

        if g_id is None:
            print(f"[WARN] Skipping {f.name}: could not determine graph id (need 'g' in filename or 'graph' in JSON).")
            continue

        by_graph.setdefault(g_id, []).append(e)
        if c_id is not None:
            cuts_by_graph.setdefault(g_id, []).append(c_id)

    if eps_ref is None or not by_graph:
        raise RuntimeError("No valid files found with matching eps/Energies and identifiable graph ids.")

    # Convert lists to arrays for each graph
    for g, energies_list in by_graph.items():
        by_graph[g] = [np.asarray(arr, dtype=float) for arr in energies_list]

    return eps_ref, by_graph, cuts_by_graph

def compute_two_stage_stats(by_graph: Dict[int, List[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Per-graph mean over cuts
    per_graph_means = []
    for g, e_list in by_graph.items():
        stack = np.vstack(e_list)  # shape (num_cuts_g, N)
        g_mean = np.mean(stack, axis=0)
        per_graph_means.append(g_mean)

    G = len(per_graph_means)
    per_graph_means_stack = np.vstack(per_graph_means)  # (G, N)

    mean_across_graphs = np.mean(per_graph_means_stack, axis=0)
    std_across_graphs = np.std(per_graph_means_stack, axis=0, ddof=1) if G > 1 else np.zeros_like(mean_across_graphs)
    stderr_across_graphs = std_across_graphs / math.sqrt(G) if G > 0 else np.zeros_like(mean_across_graphs)

    return mean_across_graphs, std_across_graphs, stderr_across_graphs, per_graph_means_stack

def save_csv(path: Path, eps: np.ndarray, mean: np.ndarray, std: np.ndarray):
    import csv
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["eps", "energy_mean", "energy_std_across_graphs"])
        for x, m, s in zip(eps, mean, std):
            writer.writerow([f"{x:.16g}", f"{m:.16g}", f"{s:.16g}"])

def plot(eps: np.ndarray,
         mean: np.ndarray,
         err: np.ndarray,
         err_mode: str,
         per_graph_means_stack: np.ndarray,
         all_runs: List[np.ndarray],
         png_path: Path):

    plt.figure()

    # 1) Small dots for each WS-QAOA run
    #    (plotting each file's energy array as tiny points along eps)
    for run in all_runs:
        plt.scatter(eps, run, s=8, alpha=0.35, linewidths=0)

    # 2+3) IQR band & median across graphs (of per-graph means)
    p25 = np.percentile(per_graph_means_stack, 25, axis=0)
    med = np.percentile(per_graph_means_stack, 50, axis=0)
    p75 = np.percentile(per_graph_means_stack, 75, axis=0)

    plt.fill_between(eps, p25, p75, alpha=0.25, label="IQR (25–75%) across graphs")
    plt.plot(eps, med, label="Median (across graphs)")

    # --- NEW: annotate the two maximum points of the median curve ---
    if med.size >= 2:
        # indices of the two largest median values
        top2_idx = np.argsort(med)[-2:]
        for i in top2_idx:
            x_val = eps[i]
            y_val = med[i]
            # optional marker to highlight
            plt.scatter([x_val], [y_val], s=30, zorder=5)
            # text with value next to the point
            plt.annotate(
                f"{y_val:.3g}",
                xy=(x_val, y_val),
                xytext=(0, 8),           # slight offset above the point
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    # ---------------------------------------------------------------

    # 4) Original mean ± error shading (if requested), added on top for continuity
    plt.plot(eps, mean, linestyle="--", label="Mean (across graphs)")
    if err_mode != "none" and np.any(err > 0):
        lower = mean - err
        upper = mean + err
        plt.fill_between(eps, lower, upper, alpha=0.15, label=f"± {err_mode} (across graphs)")

    plt.xlabel("eps")
    plt.ylabel("Energy")
    plt.title("Energy vs eps\nDots: WS-QAOA runs · Band: 25–75% across graphs · Line: median (across graphs)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print(f"[OK] Saved plot -> {png_path}")

def main():
    args = parse_args()

    files = find_files(args.folder, args.pattern)
    if not files:
        raise FileNotFoundError(f"No files match {args.pattern!r} in {str(args.folder)!r}")

    print(f"[INFO] Found {len(files)} files. Loading/grouping...")

    eps, by_graph, cuts_by_graph = collect_grouped(files, args.tolerance)

    # Log grouping summary
    total_files_used = sum(len(v) for v in by_graph.values())
    print(f"[INFO] Using {total_files_used} files across {len(by_graph)} graphs.")
    for g in sorted(by_graph):
        ncuts = len(by_graph[g])
        cuts_list = cuts_by_graph.get(g)
        if cuts_list:
            print(f"  - Graph {g}: {ncuts} cuts -> {sorted(set(cuts_list))}")
        else:
            print(f"  - Graph {g}: {ncuts} cuts")

    # Two-stage stats (and per-graph means stack for quantiles/median)
    mean, std_graphs, stderr_graphs, per_graph_means_stack = compute_two_stage_stats(by_graph)

    # Choose error shading (across graphs)
    err = {"none": np.zeros_like(mean), "std": std_graphs, "stderr": stderr_graphs}[args.with_error]

    # Save CSV (mean + std across graphs) – unchanged
    save_csv(args.csv, eps, mean, std_graphs)
    print(f"[OK] Saved CSV -> {args.csv}")

    # Collect all individual WS-QAOA runs (each file) for dot layer
    all_runs = []
    for e_list in by_graph.values():
        all_runs.extend(e_list)  # each is shape (N,)

    # Plot with dots + IQR + median (+ optional mean±error)
    plot(eps, mean, err, args.with_error, per_graph_means_stack, all_runs, args.png)

if __name__ == "__main__":
    main()
