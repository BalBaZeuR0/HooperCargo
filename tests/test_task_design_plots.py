from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_task_design import plot_task_design, thrust_to_weight  # noqa: E402

from hoopercargo.config import HopperParams  # noqa: E402


def test_thrust_to_weight_matches_spec_table():
    p = HopperParams.for_scale(1.0)
    assert abs(thrust_to_weight(p, payload=0.0, fuel=0.6) - 45 / (2.23 * 9.81)) < 0.01   # ~2.1
    assert abs(thrust_to_weight(p, payload=0.8, fuel=0.6) - 45 / (3.03 * 9.81)) < 0.01   # ~1.5


def test_task_design_plots_written(tmp_path):
    out = plot_task_design(tmp_path, samples=256)
    assert {p.name for p in out} == {"task_randomization.png", "thrust_to_weight.png"}
    assert all(p.stat().st_size > 0 for p in out)
