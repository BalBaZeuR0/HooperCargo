"""Result figures for the README / 1002 proposal from evaluate.py CSVs.

    python scripts/make_plots.py --eval results/eval.csv [--baseline results/eval_fixed.csv]
                                 [--progress runs/main/progress.csv] --out plots/
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dataviz reference palette (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")   # categorical slots 1-2

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "lines.linewidth": 2,
})


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def read_rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def binned_success(rows, key, bins=5):
    xs = [float(r[key]) for r in rows]
    lo, hi = min(xs), max(xs)
    width = (hi - lo) / bins or 1.0
    out = []
    for b in range(bins):
        a, z = lo + b * width, lo + (b + 1) * width
        sel = [r for r, x in zip(rows, xs) if a <= x < z or (b == bins - 1 and x == z)]
        k = sum(int(r["success"]) for r in sel)
        ci = wilson_interval(k, len(sel))
        out.append(((a + z) / 2, k / len(sel) if sel else float("nan"), ci, len(sel)))
    return out


def _success_axes(ax, xlabel):
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Safe-landing rate")
    ax.set_xlabel(xlabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def _plot_binned(ax, rows, key, color, label=None, scale=1.0):
    pts = [p for p in binned_success(rows, key) if p[3] > 0]
    x = [p[0] * scale for p in pts]
    y = [p[1] for p in pts]
    err = [[p[1] - p[2][0] for p in pts], [p[2][1] - p[1] for p in pts]]
    ax.errorbar(x, y, yerr=err, color=color, marker="o", markersize=6, capsize=3,
                elinewidth=1.2, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    return x, y


FIGS = [
    ("success_vs_payload.png", "payload_mass", "Payload mass (kg)", 1.0,
     "Safe landings vs payload mass"),
    ("success_vs_com_shift.png", "com_shift", "Centre-of-mass shift from payload (cm)", 100.0,
     "Safe landings vs centre-of-mass shift"),
    ("success_vs_wind.png", "wind_mean", "Mean wind speed (m/s)", 1.0,
     "Safe landings vs wind"),
]


def make_plots(eval_csv, out_dir, baseline=None, progress=None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(eval_csv)
    n = len(rows)
    written = []
    for name, key, xlabel, scale, title in FIGS:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        _plot_binned(ax, rows, key, SERIES[0], scale=scale)
        _success_axes(ax, xlabel)
        ax.set_title(f"{title}  (n = {n}, 95% CI)", loc="left")
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)
        written.append(out_dir / name)

    if baseline is not None:
        base = read_rows(baseline)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for r, color, label in ((rows, SERIES[0], "Trained with random payload"),
                                (base, SERIES[1], "Trained with fixed payload")):
            x, y = _plot_binned(ax, r, "payload_mass", color, label=label)
        _success_axes(ax, "Payload mass (kg)")
        ax.set_title("Payload randomisation vs fixed-payload training", loc="left")
        ax.legend(frameon=False, loc="lower left", labelcolor=INK_2)
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_comparison.png", dpi=160)
        plt.close(fig)
        written.append(out_dir / "baseline_comparison.png")

    if progress is not None:
        prog = [r for r in read_rows(progress) if r.get("eval/episode_success") not in (None, "")]
        steps = [float(r["global_step"]) / 1e6 for r in prog]
        succ = [float(r["eval/episode_success"]) for r in prog]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(steps, succ, color=SERIES[0], marker="o", markersize=4)
        prev = None
        for s, r in zip(steps, prog):
            if prev is not None and r["stage"] != prev:
                ax.axvline(s, color=AXIS, linestyle="--", linewidth=1)
                ax.text(s, 1.02, f" stage {r['stage']}", color=MUTED, fontsize=8, va="bottom")
            prev = r["stage"]
        _success_axes(ax, "Environment steps (millions)")
        ax.set_title("Training curve: evaluation safe-landing rate", loc="left", pad=14)
        fig.tight_layout()
        fig.savefig(out_dir / "training_curve.png", dpi=160)
        plt.close(fig)
        written.append(out_dir / "training_curve.png")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--progress", type=Path)
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args(argv)
    for p in make_plots(args.eval, args.out, args.baseline, args.progress):
        print(p)


if __name__ == "__main__":
    main()
