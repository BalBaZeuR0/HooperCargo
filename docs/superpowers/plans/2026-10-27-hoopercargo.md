# HooperCargo Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Amaç:** Haktan'ın hopper'ını, rastgele yük/yakıt/rüzgâr altında rastgele bir piste güvenli iniş yapan, PPO ile eğitilmiş bir MJX ortamına çevirmek ve 1002 için ölçüm grafiklerini üretmek.

**Mimari:** Saf-JAX bir Brax `base.Env` (`hoopercargo/env.py`). Yük ve yakıt, hopper gövdesine kaynaklı (eklemsiz) iki çocuk gövde olarak modellenir; kütle ve ağırlık merkezleri her adımda `mjx.Model` alanları değiştirilerek güncellenir. İtki, rüzgâr sürüklenmesi ve açısal sönüm `xfrc_applied` ile uygulanır (Haktan'ın yöntemi). Eğitim Brax PPO, kademeli müfredat, rlpanel ile canlı takip.

**Teknoloji:** Python ≥3.11, mujoco==3.8.0, mujoco-mjx==3.8.0, jax==0.7.2, brax==0.14.2, flax, optax, matplotlib, imageio[ffmpeg], pytest, rlpanel v0.1.0.

**Spec:** `docs/superpowers/specs/2026-10-27-hoopercargo-design.md`

## Global Kısıtlar

- Taban model Haktan'ın `assets/hopper_default.xml`'i; README ve dosya başlığında https://github.com/GulogluHaktan/3DRocketRl'e atıf.
- Ölçek kuralları (L = `scale`): uzunluk ×L, kütle ×L³, eylemsizlik ×L⁵, itki ×L³, sürüklenme alanı ×L², hız sınırları/rüzgâr ×√L, süreler ×√L, Isp ×√L, gimbal tork/sertlik/sürtünme ×L⁴, gimbal sönüm ×L^4.5, armature ×L⁵, servo hızı (rad/s) ×L^-0.5, açısal sönüm katsayısı ×L^4.5. Kontrol adımı (dt) sabit.
- `scale = 1` sabitleri: boş kütle 1,6 kg; itki 45 N; TVC ±20°; yakıt 0,6 kg, Isp 150 s, g₀ 9,80665; yük 0–0,8 kg; kargo bölmesi gövde-çerçevesinde z ∈ [0,00, 0,35] m, yanal sapma ≤ 0,03 m; yakıt deposu z = –0,07 m; iniş pisti yarıçapı 1,5 m; pist uzaklığı 20–100 m.
- Güvenli iniş = (a) 4 ayak yerde **ve** pist içinde, (b) dokunuştan 2,0 s sonra eğim < 15°, (c) dokunma anında dikey hız < 2 m/s, yatay hız < 1 m/s. Eğim > 60° anında başarısızlık. Bölüm en fazla 40 s.
- İlk yere değmede (havalandıktan sonra) motor kesilir.
- Rüzgâr: ortalama 0–6 m/s, rastgele yön; Ornstein–Uhlenbeck esinti σ = 1,5 m/s; F = ½ρC_dA|v_rel|v_rel, ρ = 1,225, C_d = 1,0, A = 0,072 m².
- Gözlem: yük **kütlesi** var; yük **konumu** ve **rüzgâr** yok.
- Kontrol: dt = 0,02 s (timestep 0,002 × frame_skip 10), bölüm 2000 adım.
- Kod hem JAX-CPU (Windows) hem JAX-GPU'da çalışır; testler `JAX_PLATFORMS=cpu`, `RLPANEL_DISABLE=1` ile koşar.
- Raporlanan sonuçlar eğitim ödülünden değil, `evaluate.py`'nin deterministik, sabit tohumlu koşusundan gelir.
- README İngilizce ve MTA şablonuna uygun (`mta-github-repo` skill'i).

## Review Focus

1. **Yük + yakıt kütle değişimi gerçekten dinamiğe giriyor mu:** `mjx.Model` alanını `replace` ile değiştirince `mjx.step` yeni kütleyi kullanmalı (önceden hesaplanmış bir alanda takılı kalmamalı). Test Task 4'te: ağır yükte aynı gazla daha yavaş yükselme.
2. **Kalkış anı yanlış "dokunuş" sayılmamalı:** Ayaklar başta yerde. Havalanma bayrağı kalkmadan dokunuş algılanmamalı. Test Task 5'te.
3. **Hiç kalkamayan politika:** Sıfır gazla 40 s bekleyen hopper "güvenli iniş" sayılmamalı, zaman aşımı başarısızlığı olmalı. Test Task 5'te.
4. **Yakıt negatife düşmemeli:** Yakıt sıfıra inince kütle ≥ 1e-4 kalmalı, itki sıfır olmalı, NaN çıkmamalı. Test Task 4'te.
5. **Değerlendirme tekrarlanabilir olmalı:** Aynı tohumla iki `evaluate` koşusu aynı CSV'yi vermeli. Test Task 7'de.

---

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `pyproject.toml`, `.gitignore`, `tests/conftest.py` | paket, bağımlılık, test ortamı |
| `assets/hooper_cargo.xml` | scale=1 MJCF şablonu (Haktan modeli + `payload`, `fuel` çocuk gövdeleri, mocap `landing_pad`, `launch_pad`, geniş zemin, yalnız ayak–zemin teması) |
| `hoopercargo/config.py` | `HopperParams` (scale'e göre türetilmiş tüm fiziksel sabitler), `StageConfig` + `STAGES` |
| `hoopercargo/model.py` | şablonu `scale` ile ölçekleyip `mujoco.MjModel` üretir |
| `hoopercargo/payload.py` | yük örnekleme, yakıt/yük gövdelerinin kütle-eylemsizlik hesabı, yüklü ağırlık merkezi |
| `hoopercargo/wind.py` | OU esinti güncellemesi, sürüklenme kuvveti |
| `hoopercargo/reward.py` | yönlendirme hızı + ödül (saf fonksiyonlar) |
| `hoopercargo/env.py` | `HooperCargoEnv(base.Env)` |
| `hoopercargo/train.py` | Brax PPO + müfredat + rlpanel + parametre kaydı (CLI) |
| `hoopercargo/evaluate.py` | N bölüm deterministik değerlendirme → CSV (CLI) |
| `scripts/make_plots.py`, `scripts/render_flight.py` | grafikler, GIF |

---

### Task 1: Paket iskeleti, model şablonu ve ölçekleme

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `tests/conftest.py`, `assets/hooper_cargo.xml`, `hoopercargo/__init__.py`, `hoopercargo/config.py`, `hoopercargo/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces:
  - `config.HopperParams` (frozen dataclass): `scale, dry_mass, max_thrust, max_tvc_rad, fuel_mass0, isp, payload_mass_max, bay_z_min, bay_z_max, bay_lateral_max, tank_z, pad_radius, leg_span, foot_radius, body_half_height, drag_area, drag_cd, ang_damping, servo_speed, touchdown_vz_max, touchdown_vxy_max, settle_time, tilt_success_rad, tilt_fail_rad, episode_seconds, dt, frame_skip`; `HopperParams.for_scale(scale: float = 1.0) -> HopperParams` Global Kısıtlar'daki ölçek kurallarını uygular.
  - `config.StageConfig(dist_min, dist_max, payload_max, wind_max, gust_sigma)`; `config.STAGES: tuple[StageConfig, ...]` = (5–20 m, 0–0,3 kg, 0 m/s, σ 0) · (20–60, 0–0,8, 0–2, 0,5) · (20–100, 0–0,8, 0–6, 1,5); uzunluklar/kütleler/hızlar `stage.scaled(params)` ile ölçeklenir.
  - `model.build_model(scale: float = 1.0) -> mujoco.MjModel`; gövde adları `hopper`, `payload`, `fuel`, `landing_pad` (mocap), `launch_pad` (mocap); site `thrust_site`; geom `foot1_geom`…`foot4_geom`, `ground`.
  - `model.BodyIds` (NamedTuple: `hopper, payload, fuel, landing_pad_mocap, launch_pad_mocap, foot_geoms, thrust_site, yaw_qpos, pitch_qpos`) ve `model.body_ids(m: mujoco.MjModel) -> BodyIds`.

- [ ] **Step 1: Bağımlılık kurulumu.** `pyproject.toml` (Teknoloji satırındaki sürümler; `jax[cuda13]` yalnız `[project.optional-dependencies] gpu`; rlpanel `rlpanel @ git+https://github.com/BalBaZeuR0/RLpanel@v0.1.0`). `python -m venv .venv` → `.venv\Scripts\python -m pip install -e .[dev]`. Beklenen: `python -c "import mujoco, jax, brax; from mujoco import mjx; print(jax.devices())"` → `[CpuDevice(id=0)]`. Kurulum Windows'ta başarısızsa DUR ve kullanıcıya bildir.
- [ ] **Step 2: `tests/conftest.py`** `os.environ.setdefault("JAX_PLATFORMS","cpu")`, `os.environ.setdefault("RLPANEL_DISABLE","1")`.
- [ ] **Step 3: Failing testler** (`tests/test_model.py`):

```python
def test_model_compiles_and_has_parts():
    m = build_model(1.0)
    for name in ["hopper", "payload", "fuel", "landing_pad", "launch_pad"]:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) >= 0

def test_dry_mass_matches_haktan():
    m = build_model(1.0); p = HopperParams.for_scale(1.0)
    hop = mj_id(m, "hopper")
    # hopper + tvc gimbal gövdeleri = 1.6 + 0.03 (Haktan'daki değerler)
    assert abs(subtree_mass_without(m, hop, ["payload", "fuel"]) - 1.63) < 1e-6

def test_only_feet_collide_with_ground():
    m = build_model(1.0)
    colliding = {mj_geom_name(m, g) for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]}
    assert colliding == {"ground", "foot1_geom", "foot2_geom", "foot3_geom", "foot4_geom"}

def test_scale2_mass_x8_and_same_thrust_to_weight():
    p1, p2 = HopperParams.for_scale(1.0), HopperParams.for_scale(2.0)
    m1, m2 = build_model(1.0), build_model(2.0)
    assert np.isclose(m2.body_mass.sum(), 8 * m1.body_mass.sum())
    assert np.isclose(p2.max_thrust / (p2.dry_mass + p2.fuel_mass0), p1.max_thrust / (p1.dry_mass + p1.fuel_mass0))
    assert np.isclose(p2.leg_span, 2 * p1.leg_span)
```
- [ ] **Step 4:** `pytest tests/test_model.py -v` → FAIL (import hatası).
- [ ] **Step 5: Şablon + `build_model`.** XML: Haktan modeli; `hopper` altına `payload` ve `fuel` jointsiz çocuk gövdeleri (başlangıç kütlesi 1e-4, `fuel` pos z = –0,07); `landing_pad`/`launch_pad` `mocap="true"`, yalnız görsel silindir (contype=conaffinity=0, yarıçap 1,5 / 0,5); zemin `size="150 150 0.1"`; ayaklar `contype=1 conaffinity=0`, zemin `contype=0 conaffinity=1`, diğer tüm geomlar 0/0. `<option>`: timestep 0,002, integrator Euler, solver CG, iterations 8, ls_iterations 4 (Haktan'ın MJX profili). `build_model` şablonu `xml.etree` ile okuyup Global Kısıtlar'daki üslerle `pos/fromto/size` (L), `mass` (L³), `diaginertia` (L⁵), gimbal joint/aktüatör alanlarını ölçekler; `mujoco.MjModel.from_xml_string`.
- [ ] **Step 6:** `pytest tests/test_model.py -v` → PASS.
- [ ] **Step 7: Commit** `git add -A && git commit -m "feat: package skeleton, cargo hopper model and scaling"`

### Task 2: Yük ve yakıt kütle hesabı

**Files:** Create `hoopercargo/payload.py` · Test `tests/test_payload.py`

**Interfaces:**
- Consumes: `HopperParams`, `StageConfig`.
- Produces:
  - `Cargo` (flax struct): `mass: Array[]`, `offset: Array[3]` (gövde-çerçevesi, yük ağırlık merkezi).
  - `sample_cargo(key, params, stage) -> Cargo`: kütle U[0, stage.payload_max]; z U[bay_z_min, bay_z_max]; yanal nokta disk içinde düzgün, yarıçap ≤ bay_lateral_max.
  - `point_inertia(mass, radius) -> Array[3]`: katı küre köşegen eylemsizlik (yük için r = 0,04·L, yakıt için r = 0,05·L).
  - `apply_mass_props(model: mjx.Model, ids: BodyIds, cargo: Cargo, fuel_mass) -> mjx.Model`: `body_mass`, `body_ipos`, `body_inertia` alanlarını `payload` ve `fuel` için yazar; kütle `max(m, 1e-4)`.
  - `loaded_com(params, cargo, fuel_mass) -> Array[3]`: hopper (orijinde) + TVC gimbal (0,03 kg, z = –0,15·L) + yük + yakıt ağırlık merkezi (gövde-çerçevesi). Gimbal kütlesi/konumu `HopperParams`'a `gimbal_mass`, `gimbal_z` olarak eklenir.

- [ ] **Step 1: Failing testler**

```python
def test_sampled_cargo_inside_bay():
    cs = jax.vmap(lambda k: sample_cargo(k, P, STAGES[2]))(jax.random.split(key, 4096))
    assert (cs.offset[:, 2] >= 0.0).all() and (cs.offset[:, 2] <= 0.35).all()
    assert (jnp.linalg.norm(cs.offset[:, :2], axis=1) <= 0.03 + 1e-9).all()
    assert (cs.mass >= 0).all() and (cs.mass <= 0.8).all()

def test_loaded_com_matches_hand_calc():
    c = Cargo(mass=0.8, offset=jnp.array([0.03, 0.0, 0.30]))
    com = loaded_com(P, c, fuel_mass=0.6)
    M = 1.63 + 0.8 + 0.6
    # 0.03 kg TVC gimbal gövdeleri z = -0.15'te; hopper'ın kendi kütlesi gövde orijininde
    assert jnp.allclose(com, jnp.array([0.8*0.03, 0, 0.8*0.30 + 0.6*(-0.07) + 0.03*(-0.15)]) / M, atol=1e-4)

def test_mjx_com_matches_loaded_com():
    m = build_model(1.0); mx = mjx.put_model(m); d = mjx.make_data(mx)
    mx2 = apply_mass_props(mx, body_ids(m), c, 0.6)
    d = mjx.forward(mx2, d)
    assert jnp.allclose(d.subtree_com[ids.hopper] - d.xpos[ids.hopper], loaded_com(P, c, 0.6), atol=2e-3)

def test_zero_fuel_keeps_positive_mass():
    mx2 = apply_mass_props(mx, ids, Cargo(0.0, jnp.zeros(3)), 0.0)
    assert (mx2.body_mass > 0).all()
```
- [ ] **Step 2:** `pytest tests/test_payload.py -v` → FAIL
- [ ] **Step 3:** `payload.py` uygula (`model.replace(body_mass=model.body_mass.at[i].set(...))` biçiminde).
- [ ] **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: payload sampling and mass properties`

### Task 3: Rüzgâr

**Files:** Create `hoopercargo/wind.py` · Test `tests/test_wind.py`

**Interfaces:**
- Produces:
  - `WindState` (flax struct): `mean: Array[3]`, `gust: Array[3]` (z bileşenleri 0).
  - `sample_wind(key, stage_scaled) -> WindState`: hız U[0, wind_max], yön U[0, 2π), gust 0.
  - `step_gust(key, gust, sigma, dt, tau=2.0) -> Array[3]`: OU, kesin ayrıklaştırma `g·e^{-dt/τ} + σ·sqrt(1-e^{-2dt/τ})·N(0,1)` (yalnız x,y).
  - `drag_force(v_body, wind_vel, params) -> Array[3]`: `v_rel = wind_vel - v_body`, `0.5·ρ·Cd·A·|v_rel|·v_rel`.

- [ ] **Step 1: Failing testler**

```python
def test_no_wind_no_motion_no_force():
    assert jnp.allclose(drag_force(jnp.zeros(3), jnp.zeros(3), P), 0)

def test_drag_formula_and_direction():
    f = drag_force(jnp.zeros(3), jnp.array([6.0, 0, 0]), P)
    assert jnp.allclose(f, jnp.array([0.5*1.225*1.0*0.072*36, 0, 0]), rtol=1e-5)

def test_gust_stationary_std():
    g = run_gust_chain(steps=200_000, sigma=1.5, dt=0.02)   # test içi lax.scan yardımcı
    assert abs(float(g[1000:, 0].std()) - 1.5) < 0.1 and float(jnp.abs(g[:, 2]).max()) == 0.0
```
- [ ] **Step 2:** FAIL → **Step 3:** uygula → **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: wind gusts and drag`

### Task 4: Ortam fiziği (reset, step, gözlem)

**Files:** Create `hoopercargo/env.py` · Test `tests/test_env_physics.py`

**Interfaces:**
- Consumes: Task 1–3 arayüzleri.
- Produces:
  - `HooperCargoEnv(stage: int = 2, scale: float = 1.0)`, `observation_size == 31`, `action_size == 3`. Eylem [-1,1]³: `throttle = (a0+1)/2`, `tvc = a[1:3]·max_tvc`.
  - `CargoState` (flax struct, `pipeline_state` olarak taşınır): `data, model_mass (body_mass, body_ipos, body_inertia dizileri), cargo: Cargo, fuel: Array[], wind: WindState, pad_xy: Array[2], tvc_cmd: Array[2], last_action: Array[3], airborne: bool, touched: bool, touch_vz, touch_vxy, settle_timer, step: int, rng`.
  - Gözlem sırası (31): `rel_pad = (pad_xy,0) − com` / (50·L) [3], `com_vel`/(10·√L) [3], döndürme matrisi düzleştirilmiş [9], gövde açısal hızı [3], TVC açıları/max_tvc [2], `fuel/fuel_mass0` [1], `cargo.mass/payload_max` [1], 4 ayak teması (ayak merkezi z ≤ foot_radius + 0,01·L) [4], `airborne` [1], `touched` [1], `last_action` [3].
  - `state.metrics` anahtarları (yalnız son adımda dolu, diğer adımlarda 0 — Brax eval bölüm boyunca toplar): `success, fail_tilt, fail_offpad, fail_hard, fail_timeout, landing_error, fuel_used, flight_time, touch_vz, touch_vxy`.

  Fizik alt-adımı (Haktan'ın `_physics_substep` yapısı): servo hız sınırlı TVC; `model = apply_mass_props(...)` (yakıt her kontrol adımında); itki = `site_xmat[thrust]·ẑ · throttle·max_thrust · (fuel > 0) · (~touched)`; tork = `cross(site_xpos − xipos[hopper], F)`; sürüklenme kuvveti `com_vel`'e göre ve `xipos[hopper]`'a; açısal sönüm `−ang_damping·ω`; yakıt `−= |F|/(isp·g₀)·dt`, `max(·,0)`. Reset: hopper launch_pad (0,0)'da, ayaklar yerde; `pad_xy` = uzaklık U[stage] × rastgele yön; mocap pozisyonları ayarlanır.

- [ ] **Step 1: Failing testler**

```python
def test_reset_shapes_and_jit_vmap():
    env = HooperCargoEnv(stage=2)
    s = jax.jit(jax.vmap(env.reset))(jax.random.split(key, 8))
    assert s.obs.shape == (8, 31)
    s2 = jax.jit(jax.vmap(env.step))(s, jnp.zeros((8, 3)))
    assert jnp.isfinite(s2.obs).all()

def test_zero_throttle_stays_on_ground():      # a0 = -1 → gaz 0
    s = rollout(env, action=[-1,0,0], seconds=1.0)
    assert com_z(s) < start_com_z + 0.01

def test_full_throttle_climbs_and_heavier_climbs_slower():
    light = rollout(env_with_cargo(0.0), [1,0,0], 1.0); heavy = rollout(env_with_cargo(0.8), [1,0,0], 1.0)
    assert com_z(light) > com_z(heavy) > start_com_z + 0.5

def test_fuel_burns_and_thrust_stops_when_empty():
    s = rollout(env, [1,0,0], seconds=60.0, stop_at_done=False)
    assert float(s.pipeline_state.fuel) == 0.0 and jnp.isfinite(s.obs).all()

def test_pad_distance_in_stage_range():
    s = jax.vmap(env.reset)(jax.random.split(key, 1024))
    d = jnp.linalg.norm(s.pipeline_state.pad_xy, axis=1)
    assert (d >= 20).all() and (d <= 100).all()
```
(`rollout`, `env_with_cargo` test-içi yardımcılar; `env_with_cargo` reset sonrası `cargo`'yu `replace` eder.)
- [ ] **Step 2:** FAIL → **Step 3:** uygula → **Step 4:** PASS (`pytest tests/test_env_physics.py -v`; CPU derleme ~1 dk sürebilir, fixture `scope="module"`).
- [ ] **Step 5: Commit** `feat: cargo hopper MJX env physics`

### Task 5: İniş mantığı ve ödül

**Files:** Create `hoopercargo/reward.py`; Modify `hoopercargo/env.py` · Test `tests/test_env_landing.py`, `tests/test_reward.py`

**Interfaces:**
- Produces:
  - `guidance_velocity(rel_xy: Array[2], height: Array[], params) -> Array[3]`: `d=|rel_xy|`; yatay `rel_xy/d · min(8·√L, 0.5·d)`; hedef irtifa `h_des = clip(0.3·d, 0, 15·L)` (d > pad_radius iken), değilse 0; dikey `clip(0.8·(h_des − h), −2·√L, 3·√L)`, pist üstünde (d ≤ pad_radius) `−clip(0.5·h + 0.3·√L, 0.3·√L, 2·√L)`.
  - `step_reward(obs_terms, v, v_des, tilt, throttle, d_action, event) -> Array[]` başlangıç değerleri: `1.0·exp(−|v−v_des|²/4) + 0.5·exp(−tilt²/0.1) − 0.02·throttle − 0.1·|Δa|²`; olay ödülleri: başarı +200, devrilme/sert iniş/pist dışı −100, zaman aşımı −100. Bu katsayılar Task 8'de ayarlanabilir; sabitler `reward.py` başında `REWARD = dict(...)`.
  - Env mantığı: `airborne` = en alçak ayak z > 0,3·L olduğu ilk an; `touched` = airborne iken herhangi bir ayak temas ettiği ilk an → o anki `|v_z|`, `|v_xy|` kaydedilir, motor kesilir, `settle_timer` başlar. `settle_timer ≥ settle_time` → değerlendir: (4 ayak temasta) ∧ (4 ayak pist içinde) ∧ (eğim < 15°) ∧ (touch_vz < 2) ∧ (touch_vxy < 1) → `success`; aksi hâlde ilk sağlanmayan şarta göre `fail_offpad / fail_tilt / fail_hard`. Herhangi bir anda eğim > 60° → `fail_tilt`, bitti. Adım = 2000 → `fail_timeout`. `landing_error` = ayak merkezleri ortalamasının pist merkezine yatay uzaklığı.

- [ ] **Step 1: Failing testler**

```python
# test_env_landing.py — hopper'ı pist üstüne yerleştirip (airborne=True) bırakan yardımcı: drop(env, height, tilt_deg, vxy, offset_xy)
def test_upright_slow_drop_on_pad_is_success():     s = drop(env, 0.3, 0, 0, (0,0));      assert m(s,"success")==1
def test_tilted_drop_tips_over_is_fail():           s = drop(env, 0.3, 40, 0, (0,0));     assert m(s,"fail_tilt")==1
def test_off_pad_landing_is_fail():                 s = drop(env, 0.3, 0, 0, (3.0,0));    assert m(s,"fail_offpad")==1
def test_hard_touchdown_is_fail():                  s = drop(env, 0.0, 0, 0, (0,0), vz=-3.0); assert m(s,"fail_hard")==1
def test_launch_contact_is_not_touchdown():
    s = rollout(env, [-1,0,0], seconds=1.0); assert not s.pipeline_state.touched
def test_never_taking_off_times_out():
    s = rollout(env, [-1,0,0], seconds=41.0); assert m(s,"fail_timeout")==1 and m(s,"success")==0
def test_engine_cut_after_touchdown():
    s = drop(env, 0.3, 0, 0, (0,0), action=[1,0,0])   # tam gaz istense bile
    assert m(s,"success")==1                              # yeniden kalkmadı

# test_reward.py
def test_guidance_points_to_pad_and_descends_on_pad():
    v = guidance_velocity(jnp.array([50.,0]), 10., P); assert v[0] > 0 and abs(v[1]) < 1e-6
    v = guidance_velocity(jnp.array([0.2,0]), 5., P);  assert v[2] < 0
def test_tracking_reward_higher_when_matching():
    assert step_reward(..., v=vd, v_des=vd, ...) > step_reward(..., v=vd+2, v_des=vd, ...)
```
- [ ] **Step 2:** FAIL → **Step 3:** uygula → **Step 4:** `pytest tests -v` → hepsi PASS
- [ ] **Step 5: Commit** `feat: landing criteria and reward`

### Task 6: PPO eğitimi + müfredat + rlpanel

**Files:** Create `hoopercargo/train.py` · Test `tests/test_train_smoke.py`

**Interfaces:**
- Produces:
  - CLI `python -m hoopercargo.train --run-dir runs/<ad> --steps-per-stage 50e6,100e6,150e6 --num-envs 1024 --seed 0 [--scale 1.0] [--stages 0,1,2] [--no-payload-randomization]`.
  - `train(run_dir: Path, steps_per_stage: list[int], num_envs: int, seed: int, scale: float, stages: list[int], randomize_payload: bool, ppo_overrides: dict | None = None) -> Path` → son parametre dosyası `run_dir/params_stage{k}.pkl` (`brax.io.model.save_params`), `run_dir/config.json`, `run_dir/progress.csv`.
  - `--no-payload-randomization`: yük kütlesi sabit 0,4 kg, konum bölme ortası (spec §6 karşılaştırması için).
  - Varsayılan PPO: `episode_length=2000, unroll_length=20, batch_size=512, num_minibatches=32, num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-3, discounting=0.99, reward_scaling=0.1, normalize_observations=True, network: policy/value (256,256,256)`. Kademe k+1, k'nın parametreleriyle başlar (`ppo.train(..., restore_params=...)`; Brax 0.14.2'de bu argümanın adı farklıysa eşdeğerini kullan).
  - rlpanel: `Panel(project="HooperCargo", run=f"{run_dir.name}/seed_{seed}", seed=seed, config=cfg, total_steps=sum(steps))`; `progress_fn(step, metrics)` → `panel.log(metrics, step=offset+step)`, `panel.progress(offset+step, total)`; sonda `panel.result({"final_success": ...})`, `panel.finish()`.

- [ ] **Step 1: Failing test**

```python
def test_tiny_training_writes_params(tmp_path):
    out = train(tmp_path, steps_per_stage=[4096], num_envs=8, seed=0, scale=1.0, stages=[0],
                randomize_payload=True, ppo_overrides=dict(episode_length=50, batch_size=8, num_minibatches=2,
                unroll_length=8, num_evals=2, policy_hidden_layer_sizes=(16,), value_hidden_layer_sizes=(16,)))
    assert out.exists() and (tmp_path / "config.json").exists() and (tmp_path / "progress.csv").exists()
```
- [ ] **Step 2:** FAIL → **Step 3:** uygula → **Step 4:** PASS (CPU, birkaç dakika)
- [ ] **Step 5: Commit** `feat: PPO training with curriculum and rlpanel`

### Task 7: Değerlendirme

**Files:** Create `hoopercargo/evaluate.py` · Test `tests/test_evaluate.py`

**Interfaces:**
- Produces: CLI `python -m hoopercargo.evaluate --params runs/<ad>/params_stage2.pkl --episodes 1000 --seed 12345 --out results/eval.csv [--scale 1.0] [--stage 2]`; `evaluate(params_path, episodes, seed, stage, scale, out_csv) -> list[dict]`; deterministik politika; bölümler `jax.vmap` ile 250'lik gruplarda. CSV kolonları: `episode, payload_mass, payload_z, payload_lateral, com_shift, wind_mean, pad_distance, success, fail_reason, landing_error, fuel_used, flight_time, touch_vz, touch_vxy`. `com_shift` = `|loaded_com(cargo, fuel0) − loaded_com(Cargo(0), fuel0)|`. Konsola özet: başarı %, ortalama iniş hatası, ortalama yakıt %.

- [ ] **Step 1: Failing testler**

```python
def test_eval_writes_csv_with_columns(tmp_path, tiny_params):   # tiny_params: Task 6 test ayarlarıyla üretilen fixture
    evaluate(tiny_params, episodes=8, seed=1, stage=2, scale=1.0, out_csv=tmp_path/"e.csv")
    rows = list(csv.DictReader(open(tmp_path/"e.csv"))); assert len(rows) == 8 and set(COLUMNS) <= set(rows[0])

def test_eval_is_reproducible(tmp_path, tiny_params):
    a = evaluate(tiny_params, 8, 1, 2, 1.0, tmp_path/"a.csv"); b = evaluate(tiny_params, 8, 1, 2, 1.0, tmp_path/"b.csv")
    assert open(tmp_path/"a.csv").read() == open(tmp_path/"b.csv").read()
```
- [ ] **Step 2:** FAIL → **Step 3:** uygula → **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: deterministic evaluation to CSV`

### Task 8: Grafikler ve uçuş videosu

**Files:** Create `scripts/make_plots.py`, `scripts/render_flight.py`, `tests/fixtures/eval_sample.csv` (20 satır elle) · Test `tests/test_plots.py`

**Interfaces:**
- Produces: `make_plots.py --eval results/eval.csv [--baseline results/eval_fixed.csv] [--progress runs/<ad>/progress.csv] --out plots/` → `success_vs_payload.png`, `success_vs_com_shift.png`, `success_vs_wind.png`, `training_curve.png` (progress verilirse), `baseline_comparison.png` (baseline verilirse). Başarı oranları 5 eşit genişlikli kutuda, her kutuda Wilson %95 güven aralığı (hata çubuğu), eksen etiketleri İngilizce. `render_flight.py --params ... --seed N --out assets/demo.gif [--topdown plots/trajectories.png --n 20]`: tek bölümü JAX'te koşturur, qpos+mocap'i host `mujoco.Renderer` ile 640×480, 25 fps GIF'e çevirir (< 10 MB); `--topdown` ile 20 bölümün yörüngelerini üstten/yandan çizer.
- [ ] **Step 1: Failing test** `test_plots_from_fixture_csv(tmp_path)`: `make_plots(eval=fixture, out=tmp_path)` sonrası 3 PNG var ve boyutu > 0.
- [ ] **Step 2:** FAIL → **Step 3:** uygula (`dataviz` skill'ine bak) → **Step 4:** PASS
- [ ] **Step 5: Commit** `feat: result plots and flight rendering`

### Task 9: Eğitim koşusu ve ayar (kullanıcıyla)

Kod testi yok; çıktı sonuçlar.

- [ ] **Step 1: Çalışma ortamı kararı — DUR, kullanıcıya sor.** Seçenekler: (a) WSL2 Ubuntu kurup RTX 4050'de `pip install -e .[gpu]`; (b) Windows'ta JAX-CPU. Önce her ikisinde `benchmark` ölçümü: 1024 env × 100 adım süresi (`python -m hoopercargo.train --benchmark`; Task 6'ya küçük bir bayrak).
- [ ] **Step 2: Kademe 0 kısa koşu** (~20M adım). Kabul: eval/episode_success > 0,5. Değilse `REWARD` katsayılarını ayarla, her değişikliği `runs/<ad>/NOTES.md`'ye yaz.
- [ ] **Step 3: Tam müfredat** (kademe 0→1→2). Kabul ölçütü yok (sonuç ne çıkarsa raporlanır), ama hedef kademe 2'de başarı ≥ %80.
- [ ] **Step 4: Değerlendirme + grafikler:** `evaluate --episodes 1000 --seed 12345` → `make_plots` → `render_flight`.
- [ ] **Step 5: Yakıt kontrolü:** 80–100 m bölümlerinde medyan `fuel_used` %55–85 dışındaysa Isp'yi ayarla, kullanıcıya bildir (yeniden eğitim gerekir).
- [ ] **Step 6 (zaman kalırsa):** `--no-payload-randomization` ile eğit, aynı tohumla değerlendir (`results/eval_fixed.csv`), `baseline_comparison.png`.
- [ ] **Step 7 (zaman kalırsa):** `--scale 2.0` eğit + değerlendir.
- [ ] **Step 8: Commit** grafikler + `results/*.csv` + GIF (`runs/` git-ignored).

### Task 10: README ve yayın

- [ ] **Step 1:** `mta-github-repo` skill'ini çağır; İngilizce README (özet tablo, mermaid akış, sonuç grafikleri, kurulum, kullanım, sonuçları yeniden üretme tablosu, Haktan atfı, Terzi & Nicoli 2026 atfı). `scripts/check_readme.py` → geçmeli.
- [ ] **Step 2:** `pytest -q` → hepsi PASS.
- [ ] **Step 3: Commit**, sonra **push için kullanıcıya sor.**
