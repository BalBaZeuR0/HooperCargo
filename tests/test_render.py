from pathlib import Path
import sys

import pytest

from hoopercargo.train import TINY_PPO, train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from render_flight import plot_trajectories, render_flight  # noqa: E402


@pytest.fixture(scope="module")
def tiny_params(tmp_path_factory):
    run = tmp_path_factory.mktemp("tiny_render")
    return train(run, steps_per_stage=[1024], num_envs=8, seed=0, scale=1.0, stages=[0],
                 randomize_payload=True, ppo_overrides=TINY_PPO)


def test_render_gif(tmp_path, tiny_params):
    out = render_flight(tiny_params, seed=0, out_gif=tmp_path / "f.gif", stage=0, max_seconds=1.0)
    assert out.exists() and 0 < out.stat().st_size < 10 * 1024 * 1024


def test_trajectory_plot(tmp_path, tiny_params):
    out = plot_trajectories(tiny_params, n=3, seed=0, out_png=tmp_path / "t.png", stage=0,
                            max_seconds=2.0)
    assert out.exists() and out.stat().st_size > 0
