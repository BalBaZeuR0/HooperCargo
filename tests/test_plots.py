from pathlib import Path
import csv
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_plots import make_plots, wilson_interval  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "eval_sample.csv"


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_plots_from_fixture_csv(tmp_path):
    out = make_plots(FIXTURE, tmp_path)
    names = {p.name for p in out}
    assert {"success_vs_payload.png", "success_vs_com_shift.png", "success_vs_wind.png"} <= names
    assert all(p.stat().st_size > 0 for p in out)


def test_optional_plots(tmp_path):
    progress = tmp_path / "progress.csv"
    with open(progress, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["global_step", "stage", "eval/episode_success",
                                          "eval/episode_reward"])
        w.writeheader()
        for i in range(6):
            w.writerow({"global_step": i * 1000, "stage": i // 3,
                        "eval/episode_success": i / 6, "eval/episode_reward": i})
    out = make_plots(FIXTURE, tmp_path, baseline=FIXTURE, progress=progress)
    names = {p.name for p in out}
    assert {"training_curve.png", "baseline_comparison.png"} <= names
