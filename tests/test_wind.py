import jax
import jax.numpy as jnp

from hoopercargo.config import STAGES, HopperParams
from hoopercargo.wind import drag_force, sample_wind, step_gust

P = HopperParams.for_scale(1.0)


def test_no_wind_no_motion_no_force():
    assert jnp.allclose(drag_force(jnp.zeros(3), jnp.zeros(3), P), 0.0)


def test_drag_formula_and_direction():
    f = drag_force(jnp.zeros(3), jnp.array([6.0, 0.0, 0.0]), P)
    assert jnp.allclose(f, jnp.array([0.5 * 1.225 * 1.0 * 0.072 * 36.0, 0.0, 0.0]), rtol=1e-5)
    # moving into still air is pushed back
    f = drag_force(jnp.array([0.0, 3.0, 0.0]), jnp.zeros(3), P)
    assert f[1] < 0.0


def _gust_chain(steps, sigma, dt):
    def body(g, k):
        g = step_gust(k, g, sigma, dt)
        return g, g
    _, gs = jax.lax.scan(body, jnp.zeros(3), jax.random.split(jax.random.PRNGKey(0), steps))
    return gs


def test_gust_stationary_std():
    g = _gust_chain(200_000, 1.5, 0.02)
    assert abs(float(g[1000:, 0].std()) - 1.5) < 0.1
    assert float(jnp.abs(g[:, 2]).max()) == 0.0


def test_sampled_wind_is_horizontal_and_bounded():
    ws = jax.vmap(lambda k: sample_wind(k, STAGES[2]))(jax.random.split(jax.random.PRNGKey(1), 2048))
    speed = jnp.linalg.norm(ws.mean, axis=1)
    assert (speed <= 6.0 + 1e-5).all() and float(speed.max()) > 5.0
    assert (ws.mean[:, 2] == 0).all() and (ws.gust == 0).all()
    calm = jax.vmap(lambda k: sample_wind(k, STAGES[0]))(jax.random.split(jax.random.PRNGKey(1), 16))
    assert (calm.mean == 0).all()
