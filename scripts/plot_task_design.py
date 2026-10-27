"""Figures that describe the task itself (no trained policy needed).

    python scripts/plot_task_design.py --out plots/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hoopercargo.config import G0, STAGES, HopperParams  # noqa: E402
from hoopercargo.payload import Cargo, loaded_com, sample_cargo  # noqa: E402
from make_plots import INK_2, MUTED, SERIES  # noqa: E402  (also applies the plot style)


def thrust_to_weight(p: HopperParams, payload: float, fuel: float) -> float:
    return p.max_thrust / ((p.dry_mass + payload + fuel) * G0)


def plot_task_design(out_dir, samples: int = 2000, scale: float = 1.0) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = HopperParams.for_scale(scale)
    stage = STAGES[2].scaled(p)

    cargos = jax.vmap(lambda k: sample_cargo(k, p, stage))(
        jax.random.split(jax.random.PRNGKey(0), samples))
    empty = Cargo(mass=jp.array(0.0), offset=jp.zeros(3))
    base = loaded_com(p, empty, p.fuel_mass0)
    shift = jax.vmap(lambda c: jp.linalg.norm(loaded_com(p, c, p.fuel_mass0) - base))(cargos)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(np.asarray(cargos.mass), 100 * np.asarray(shift), s=8, color=SERIES[0],
               alpha=0.5, linewidths=0)
    ax.set_xlabel("Payload mass (kg)")
    ax.set_ylabel("CoM shift vs. empty hopper (cm)")
    ax.set_title(f"Random cargo: mass and centre-of-mass shift ({samples} samples)", loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "task_randomization.png", dpi=160)
    plt.close(fig)

    m = np.linspace(0.0, p.payload_mass_max, 50)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for fuel, color, label in ((p.fuel_mass0, SERIES[0], "full tank"), (0.0, SERIES[1], "empty tank")):
        twr = [thrust_to_weight(p, x, fuel) for x in m]
        ax.plot(m, twr, color=color, label=label)
        ax.annotate(label, (m[-1], twr[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK_2, va="center", fontsize=9)
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.0, 1.02, "thrust = weight", color=MUTED, fontsize=8, va="bottom")
    ax.set_ylim(0.8, None)
    ax.set_xlim(0, p.payload_mass_max * 1.2)
    ax.set_xlabel("Payload mass (kg)")
    ax.set_ylabel("Max thrust / weight")
    ax.set_title("Thrust-to-weight ratio across the payload range", loc="left")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_dir / "thrust_to_weight.png", dpi=160)
    plt.close(fig)
    return [out_dir / "task_randomization.png", out_dir / "thrust_to_weight.png"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("plots"))
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args(argv)
    for f in plot_task_design(args.out, args.samples):
        print(f)


if __name__ == "__main__":
    main()
