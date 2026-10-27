import csv
import json

from hoopercargo.train import TINY_PPO, train


def test_tiny_training_writes_params(tmp_path):
    out = train(tmp_path, steps_per_stage=[2048, 2048], num_envs=8, seed=0, scale=1.0,
                stages=[0, 1], randomize_payload=True, ppo_overrides=TINY_PPO)
    assert out == tmp_path / "params_stage1.pkl"
    assert out.exists() and (tmp_path / "params_stage0.pkl").exists()
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["stages"] == [0, 1] and cfg["scale"] == 1.0 and cfg["randomize_payload"] is True
    assert cfg["ppo"]["policy_hidden_layer_sizes"] == list(TINY_PPO["policy_hidden_layer_sizes"])
    rows = list(csv.DictReader(open(tmp_path / "progress.csv")))
    assert {r["stage"] for r in rows} == {"0", "1"}
    assert "eval/episode_success" in rows[0]
    # global step keeps increasing across stages
    steps = [int(float(r["global_step"])) for r in rows]
    assert steps == sorted(steps)
