import jax
import jax.numpy as jnp
import pytest

from hoopercargo.config import HOP, HOP_STAGE, STAGES, parse_stage, stage_config, stage_name
from hoopercargo.env import HooperCargoEnv


def test_hop_is_not_python_negative_indexing():
    assert stage_config(HOP) is HOP_STAGE
    assert stage_config(HOP) is not STAGES[-1]
    assert stage_config(2) is STAGES[2]
    with pytest.raises(ValueError):
        stage_config(3)


def test_cli_stage_names_round_trip():
    assert [parse_stage(x) for x in "H,0,1,2".split(",")] == [HOP, 0, 1, 2]
    assert parse_stage("h") == HOP
    assert [stage_name(s) for s in (HOP, 0, 2)] == ["H", "0", "2"]


def test_hop_puts_the_pad_a_few_metres_away_in_calm_air():
    env = HooperCargoEnv(stage=HOP)
    states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 256))
    d = jnp.linalg.norm(states.pipeline_state.pad_xy, axis=1)
    assert float(d.min()) >= 2.5 - 1e-5 and float(d.max()) <= 6.0 + 1e-5
    assert float(jnp.abs(states.pipeline_state.wind.mean).max()) == 0.0
