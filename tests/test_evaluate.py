import csv

import pytest

from hoopercargo.evaluate import COLUMNS, evaluate
from hoopercargo.train import TINY_PPO, train


@pytest.fixture(scope="module")
def tiny_params(tmp_path_factory):
    run = tmp_path_factory.mktemp("tiny_run")
    return train(run, steps_per_stage=[1024], num_envs=8, seed=0, scale=1.0, stages=[0],
                 randomize_payload=True, ppo_overrides=TINY_PPO)


def test_eval_writes_csv_with_columns(tmp_path, tiny_params):
    rows = evaluate(tiny_params, episodes=8, seed=1, stage=2, scale=1.0, out_csv=tmp_path / "e.csv")
    disk = list(csv.DictReader(open(tmp_path / "e.csv")))
    assert len(rows) == len(disk) == 8
    assert list(disk[0].keys()) == list(COLUMNS)
    for r in rows:
        assert 0.0 <= r["payload_mass"] <= 0.8
        assert 20.0 <= r["pad_distance"] <= 100.0
        assert r["fail_reason"] in {"none", "tilt", "offpad", "hard", "timeout"}
        assert (r["success"] == 1) == (r["fail_reason"] == "none")


def test_eval_is_reproducible(tmp_path, tiny_params):
    evaluate(tiny_params, 8, 1, 2, 1.0, tmp_path / "a.csv")
    evaluate(tiny_params, 8, 1, 2, 1.0, tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_text() == (tmp_path / "b.csv").read_text()


def test_eval_batches_cover_all_episodes(tmp_path, tiny_params):
    rows = evaluate(tiny_params, 7, 2, 2, 1.0, tmp_path / "c.csv", batch=3)
    assert [r["episode"] for r in rows] == list(range(7))
