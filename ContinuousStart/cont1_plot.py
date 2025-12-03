#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

FNAME_RE = re.compile(r"results_d(?P<depth>\d+)_i(?P<iter>\d+)\.json$")

def load_data(indir: Path):

    by_depth = defaultdict(lambda: defaultdict(list))  

    for p in indir.glob("results_d*_i*.json"):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        depth = int(m.group("depth"))

        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"[WARN] Skipping {p} (read error: {e})")
            continue

        metrics = [
            "temp_pw", "temp_pc", "temp_plw",
            "temp_ew", "temp_ec", "temp_elw",
            "E0",
        ]
        for key in metrics:
            if key in payload and isinstance(payload[key], (int, float)):
                by_depth[depth][key].append(float(payload[key]))
            else:
                print(f"[WARN] {p} missing '{key}', skipping that value.")

    return by_depth

def aggregate_quantiles(by_depth):

    depths_sorted = sorted(by_depth.keys())
    metrics = [
        "temp_pw", "temp_pc", "temp_plw",
        "temp_ew", "temp_ec", "temp_elw",
        "E0",
    ]
    q50 = {m: [] for m in metrics}
    q25 = {m: [] for m in metrics}
    q75 = {m: [] for m in metrics}
    counts = {m: [] for m in metrics}

    for d in depths_sorted:
        bucket = by_depth[d]
        for m in metrics:
            vals = bucket.get(m, [])
            if len(vals) == 0:
                q50[m].append(np.nan)
                q25[m].append(np.nan)
                q75[m].append(np.nan)
                counts[m].append(0)
            else:
                arr = np.asarray(vals, dtype=float)
                q50[m].append(float(np.nanmedian(arr)))
                q25[m].append(float(np.nanpercentile(arr, 25)))
                q75[m].append(float(np.nanpercentile(arr, 75)))
                counts[m].append(len(arr))

    for m in metrics:
        q50[m] = np.array(q50[m], dtype=float)
        q25[m] = np.array(q25[m], dtype=float)
        q75[m] = np.array(q75[m], dtype=float)

    return depths_sorted, q50, q25, q75, counts

def _mask_valid(arr):
    return ~np.isnan(arr)

def plot_combined(depths, q50, q25, q75, out_path):

    plt.rcParams.update({"font.size": 11})
    x = np.array(depths, dtype=float)

    fig, (ax_probs, ax_energies) = plt.subplots(
    2, 1, sharex=True, figsize=(8, 6),
    gridspec_kw={"hspace": 0.0})

    prob_style = {
        "temp_pw":  ("green",  "Warm-start"),
        "temp_pc":  ("orange", "Equal superposition"),
        "temp_plw": ("blue",   "Warm-start - normal mixer"),
    }
    energy_style = {
        "temp_ew":  ("green",  "Warm-start"),
        "temp_ec":  ("orange", "Equal superposition"),
        "temp_elw": ("blue",   "Warm-start - normal mixer"),
    }

    handles = []
    labels = []

    def prep_prob(a):
        return a * 100.0

    for key, marker in [
        ("temp_pc", "s"),   # Equal superposition
        ("temp_pw", "o"),   # Warm-start
        ("temp_plw", "^"),  # Warm-start - normal mixer
    ]:
        color, label = prob_style[key]
        med = prep_prob(q50[key])
        lo  = prep_prob(q25[key])
        hi  = prep_prob(q75[key])
        m = _mask_valid(med)
        if m.any():
            line = ax_probs.plot(
                x[m], med[m], marker=marker, linewidth=2.5,
                label=label, color=color
            )[0]
            m_iqr = _mask_valid(lo) & _mask_valid(hi) & m
            if m_iqr.any():
                ax_probs.fill_between(
                    x[m_iqr], lo[m_iqr], hi[m_iqr],
                    alpha=0.15, color=color
                )
            handles.append(line)
            labels.append(label)

    ax_probs.set_ylabel(r'Probability $|\langle d^* \mid \psi^* \rangle|^2$')
    ax_probs.grid(True, alpha=0.3)

    for key, marker in [
        ("temp_ec", "s"),   # Equal superposition
        ("temp_ew", "o"),   # Warm-start
        ("temp_elw", "^"),  # Warm-start - normal mixer
    ]:
        color, label = energy_style[key]
        med = q50[key]
        lo  = q25[key]
        hi  = q75[key]
        m = _mask_valid(med)
        if m.any():
            line = ax_energies.plot(
                x[m], med[m], marker=marker, linewidth=2.5,
                label=label, color=color
            )[0]
            m_iqr = _mask_valid(lo) & _mask_valid(hi) & m
            if m_iqr.any():
                ax_energies.fill_between(
                    x[m_iqr], lo[m_iqr], hi[m_iqr],
                    alpha=0.15, color=color
                )
            handles.append(line)
            labels.append(label)

    e0 = q50.get("E0", None)
    e0_line = None
    if e0 is not None:
        m_e0 = _mask_valid(e0)
        if m_e0.any():
            e0_vals = e0[m_e0]
            if np.nanmax(np.abs(e0_vals - np.nanmedian(e0_vals))) < 1e-9:
                e0_line = ax_energies.axhline(
                    np.nanmedian(e0_vals),
                    linestyle="--", linewidth=1.2,
                    color="gray", label="E0",
                )
            else:
                e0_line = ax_energies.plot(
                    x[m_e0], e0[m_e0],
                    linestyle="--", linewidth=1.2,
                    color="gray", label="E0",
                )[0]

    if e0_line is not None:
        handles.append(e0_line)
        labels.append("E0")

    ax_energies.set_xlabel("QAOA depth p")
    ax_energies.set_ylabel("Energy")
    ax_energies.grid(True, alpha=0.3)

    for ax in (ax_probs, ax_energies):
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.setp(ax_probs.get_xticklabels(), visible=False)
    ax_probs.tick_params(axis="x", which="both", length=0)

    ax_probs.margins(x=0.02)
    ax_energies.margins(x=0.02)

    by_label = {}
    for h, lab in zip(handles, labels):
        by_label[lab] = h  

    ax_energies.legend(
        by_label.values(), by_label.keys(),
        loc="upper right",
        framealpha=0.9
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved combined plot to {out_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate results_* JSON and plot median & IQR vs depth (combined probs+energies)."
    )
    ap.add_argument(
        "--indir", type=Path, default=Path("."),
        help="Directory containing results_d*_i*.json"
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG for combined probabilities+energies plot"
    )
    ap.add_argument(
        "--out-probs", type=Path, default=None,
        help="(Legacy) Output PNG for probabilities plot; now used as combined output if --out is not given"
    )
    ap.add_argument(
        "--out-energies", type=Path, default=None,
        help="(Legacy, ignored) Output PNG for energies plot (no longer used)"
    )

    args = ap.parse_args()

    out_path = args.out or args.out_probs or Path("probs_and_energies_vs_depth.png")

    by_depth = load_data(args.indir)
    if not by_depth:
        print("[ERROR] No matching files found. Ensure names like results_d3_i9.json exist in --indir.")
        return

    depths, q50, q25, q75, counts = aggregate_quantiles(by_depth)

    expected = max((max(v) for v in counts.values() if v), default=None)
    for m, arr in counts.items():
        for d, c in zip(depths, arr):
            if c and expected and c < expected:
                print(f"[WARN] depth={d} metric={m}: only {c} samples (expected up to ~{expected}).")

    plot_combined(depths, q50, q25, q75, out_path)


if __name__ == "__main__":
    main()
