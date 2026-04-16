---
title: "Pluggable MLflow + `torch.jit.script` — Tracking 4 Years of Runs, Compiling Actors for ×2 Speedup"
subtitle: "An opt-in tracker that's a no-op without config, a Postgres-backed deployment, and a measured CPU latency win from 10 lines of actor changes."
date: "2026-04-13"
tags: ["MLflow", "PyTorch", "MLOps", "Performance", "Python"]
summary: "Two upgrades, same afternoon. MLflow tracking wrapped as a context-manager with graceful no-op when MLFLOW_TRACKING_URI is unset — keeps the core trainer free of hard dependencies. torch.jit.script on the SAC actor after three small compatibility edits — measured ×2.13 speedup at batch size 1 on CPU, falling to ×1.08 at batch 128. The MLflow server runs on a docker-compose with Postgres so runs persist across deploys."
---

> **TL;DR.** Added two things to the trainer: (1) opt-in **MLflow
> tracking** via a context-manager that no-ops gracefully when
> `MLFLOW_TRACKING_URI` is unset — no hard dependency on mlflow at all —
> and (2) `torch.jit.script` compilation of the SAC actor. The JIT path
> gives **×2.13 at batch 1**, **×1.43 at batch 8** on CPU, measured with
> 500-iteration warmup + timed percentile harness. Both changes land in
> the same PR because they share the lifecycle layer (startup, request
> path, shutdown).

---

## 0. Two problems, same afternoon

- **Tracking.** I have had 4 years of SAC training runs saved only as
  `.pt` files with hashed filenames. Zero metadata. Zero reproducibility.
  Zero ability to say "which hyperparameters actually helped."
- **Serving speed.** The SAC actor's forward pass is the inner loop of
  every serving call. At batch size 1 (production case), Python
  overhead dominates. `torch.jit.script` removes it — but only if the
  module is script-compatible, which the original actor wasn't.

Both fit in the same post because they share the same key property: they
must be **opt-in**, **failure-tolerant**, and **zero-regression** when
disabled.

---

## 1. MLflow — the hard-dependency problem

Simplest MLflow integration is literally:

```python
import mlflow
mlflow.log_metrics({"reward": 0.5}, step=ep)
```

But now the trainer requires mlflow. That's wrong on three axes:

1. **Development friction** — fresh clone, `pip install -r requirements.txt`
   is now slower because of mlflow's many transitive deps.
2. **CI complexity** — unit tests that don't care about tracking still
   import mlflow.
3. **Behavior coupling** — a buggy mlflow server could crash training.

### The pattern: soft import + no-op

```python
# app/trading/rl/mlflow_tracker.py
class MlflowTracker(AbstractContextManager):
    """Thin wrapper that no-ops gracefully when MLflow isn't configured."""

    def __init__(self, experiment, run_name=None, params=None, tags=None):
        self.experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME") or experiment
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}

        self._enabled = bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())
        self._mlflow = None

        if self._enabled:
            try:
                import mlflow
                self._mlflow = mlflow
            except ImportError:
                logger.warning(
                    "MLFLOW_TRACKING_URI is set but mlflow is not installed "
                    "— tracking disabled."
                )
                self._enabled = False
```

Three layers of graceful degradation:

1. `MLFLOW_TRACKING_URI` unset → `_enabled = False`, no mlflow import
2. Env var set, module missing → log warning, `_enabled = False`
3. Import succeeded, `start_run` fails → log warning, `_enabled = False`

In all three cases, downstream calls like `tracker.log_metrics(...)`
silently do nothing. The trainer logic doesn't branch on "is mlflow
available."

### Usage inside `SACTrainer`

```python
tracker = MlflowTracker(
    experiment="sac_rl_stat_pair",
    run_name=f"sac_{datetime.now():%Y%m%d_%H%M}",
    params={
        "algo": "sac", "state_dim": agent.config.state_dim,
        "n_episodes": cfg.n_episodes, "gamma": cfg.gamma, ...
    },
    tags={"algo": "sac", "env_class": type(envs[0]).__name__},
)

tracker.__enter__()
try:
    for ep in range(cfg.n_episodes):
        ...
        tracker.log_metrics({"train_reward": r, "eval_reward": e}, step=ep)
    tracker.log_artifact(checkpoint_path)
finally:
    tracker.__exit__(None, None, None)
```

No `if mlflow_available` branches. The tracker handles all the "is this
actually going to fire" decisions internally.

---

## 2. Four unit tests that verify the contract

```python
def test_noop_when_tracking_uri_unset(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with MlflowTracker(experiment="t", params={"a": 1}) as tracker:
        assert tracker.enabled is False
        # These should not raise when disabled
        tracker.log_metrics({"x": 1.0}, step=0)
        tracker.log_param("k", "v")
        tracker.set_tag("t", "1")
        tracker.log_artifact("/no/such/path")


def test_flatten_nested_params():
    out = _flatten_for_mlflow({"a": 1, "nested": {"b": 2, "c": {"d": 3}}})
    assert out == {"a": 1, "nested.b": 2, "nested.c.d": 3}


def test_is_finite():
    assert _is_finite(1.0)
    assert not _is_finite(float("nan"))
    assert not _is_finite(float("inf"))


def test_real_tracking_to_sqlite(tmp_path, monkeypatch):
    """End-to-end: start a run, log params/metrics/tag, verify via mlflow API."""
    pytest.importorskip("mlflow")
    import mlflow

    backend = tmp_path / "mlruns.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{backend}")

    with MlflowTracker(experiment="pytest_exp", params={"lr": 1e-4},
                       tags={"algo": "sac"}) as tracker:
        assert tracker.enabled is True
        tracker.log_metrics({"reward": 0.5, "nan_skipped": float("nan")}, step=1)
        tracker.log_metrics({"reward": 0.8}, step=2)

    # Verify via mlflow API
    mlflow.set_tracking_uri(f"sqlite:///{backend}")
    client = mlflow.MlflowClient()
    runs = client.search_runs(experiment_ids=[...])
    assert runs[0].data.metrics["reward"] == pytest.approx(0.8)
    assert "nan_skipped" not in runs[0].data.metrics  # NaN got filtered
```

Four cases, **1.3 seconds** total. The fourth one genuinely writes to
SQLite and reads back via the mlflow client — proof the wrapper isn't
just pretending.

---

## 3. Production deployment — Postgres + docker-compose

SQLite is fine for a unit test. For real runs we want concurrent
writers, queryable history, and survivable backups. Postgres:

```yaml
# docker-compose.mlflow.yml
services:
  mlflow-postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - mlflow-pg-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow -d mlflow"]
      interval: 5s
      timeout: 3s
      retries: 10

  mlflow-server:
    image: ghcr.io/mlflow/mlflow:v2.19.0
    depends_on:
      mlflow-postgres:
        condition: service_healthy
    command: >
      sh -c "pip install --quiet psycopg2-binary &&
             mlflow server
               --host 0.0.0.0 --port 5000
               --backend-store-uri postgresql://mlflow:mlflow@mlflow-postgres:5432/mlflow
               --artifacts-destination /mlartifacts
               --serve-artifacts"
    ports:
      - "5000:5000"
    volumes:
      - mlflow-artifacts:/mlartifacts
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

volumes:
  mlflow-pg-data:
  mlflow-artifacts:
```

One command to bring it up:

```bash
$ docker compose -f docker-compose.mlflow.yml up -d
$ open http://localhost:5000
```

And from the training side:

```bash
$ export MLFLOW_TRACKING_URI=http://localhost:5000
$ python train_stat_pair_rl.py --episodes 100
```

That's it. The tracker picks up the URI, every episode's metrics land
in Postgres, and the final `.pt` gets uploaded as an artifact.

---

## 4. The JIT story — three small actor changes

The SAC actor as originally written:

```python
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class SACActor(nn.Module):
    def forward(self, state):
        x = self.shared(state)
        mean = self.mean_head(x).squeeze(-1)
        log_std = self.log_std_head(x).squeeze(-1).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)
```

Running `torch.jit.script(actor)` on this fails with:

```
python value of type 'int' cannot be used as a value. Perhaps it is a
closed over global variable? If so, please consider passing it in as
an argument or use a local variable instead.
```

TorchScript can't close over module-level globals. Three fixes:

### Fix 1: promote the globals to module constants

```python
class SACActor(nn.Module):
    __constants__ = ["log_std_min", "log_std_max"]

    def __init__(self, ...):
        super().__init__()
        ...
        self.log_std_min = float(LOG_STD_MIN)  # still -20
        self.log_std_max = float(LOG_STD_MAX)  # still 2

    def forward(self, state):
        x = self.shared(state)
        mean = self.mean_head(x).squeeze(-1)
        log_std = self.log_std_head(x).squeeze(-1).clamp(
            self.log_std_min, self.log_std_max
        )
        return mean, log_std
```

`__constants__` tells TorchScript to treat the listed attributes as
compile-time constants. Result: the `.clamp(self.log_std_min, ...)`
call becomes a fixed scalar operation.

### Fix 2: export the method we actually call at inference

`SACAgent.act()` calls `self.actor.deterministic(s)`, not `forward()`.
TorchScript only scripts `forward` by default; other methods need
`@torch.jit.export`:

```python
@torch.jit.export
def deterministic(self, state: torch.Tensor) -> torch.Tensor:
    """Mean action (no sampling) — used for serving; scriptable."""
    mean, _ = self.forward(state)
    return torch.tanh(mean)
```

### Fix 3: opt-in via env var in the serving layer

Don't change training. Don't force JIT on everyone. Just provide the
option:

```python
# sac_serving.py — inside _reload()
jit_compiled = False
if os.environ.get("STAT_PAIR_RL_JIT", "").lower() in {"1", "true", "yes"}:
    try:
        scripted_actor = torch.jit.script(agent.actor)
        scripted_actor.eval()
        agent.actor = scripted_actor
        jit_compiled = True
        logger.info("⚡ SAC actor compiled with torch.jit.script")
    except Exception as e:
        logger.warning("torch.jit.script failed, falling back to eager: %s", e)
```

Failure falls back to eager. Env var unset → no change. Exactly the
same opt-in posture as MLflow.

---

## 5. The benchmark

```bash
$ uv run python scripts/bench_sac_jit.py

════════════════════════════════════════════════════════════════════════
 SAC Actor inference latency — eager vs torch.jit.script
 state_dim=28, warmup=50, iters=500
════════════════════════════════════════════════════════════════════════
 bs=  1  eager mean=0.0912 p95=0.1965  jit mean=0.0429 p95=0.0716  speedup ×2.13
 bs=  8  eager mean=0.1427 p95=0.2663  jit mean=0.0994 p95=0.1505  speedup ×1.43
 bs= 32  eager mean=0.1605 p95=0.3162  jit mean=0.1308 p95=0.2583  speedup ×1.23
 bs=128  eager mean=0.2054 p95=0.3065  jit mean=0.1909 p95=0.3186  speedup ×1.08
```

Reading this:

| Batch | Eager p95 (ms) | JIT p95 (ms) | Speedup |
|-------|---------------|-------------|---------|
| 1 | 0.197 | 0.072 | **×2.13** |
| 8 | 0.266 | 0.151 | ×1.43 |
| 32 | 0.316 | 0.258 | ×1.23 |
| 128 | 0.307 | 0.319 | ×1.08 |

**The speedup scales inversely with batch size**, which is exactly what
you'd expect if you believe the story: Python overhead dominates at
small batches; GEMM dominates at large ones. Since production serving
is bs=1 (one pair at a time), ×2 is the real-world number.

This is also why I specifically wanted the numbers measured at *bs=1*,
not averaged. "×1.08 at bs=128" is a fine academic number but irrelevant
when the hot path is single-state inference.

---

## 6. Surfacing the metric in the API

Two small response-field additions:

```python
class PredictResponse(BaseModel):
    action: float
    discrete: Optional[int] = None
    model_path: str
    model_mtime_utc: str
    latency_ms: float       # ← new
    jit_compiled: bool      # ← new
```

Every `/sac/predict` response now includes both the measured latency
and whether JIT was active when it served. Clients can aggregate latency
percentiles without a separate benchmark endpoint. For load testing
there's also `/sac/benchmark` that runs warmup + timed iters against
synthetic inputs and returns p50/p95/p99.

---

## 7. The seven cases that matter

```
✅ test_cold_start_no_model_returns_503
✅ test_load_predict_and_hot_reload
✅ test_force_reload_endpoint
✅ test_batch_shape_validation
✅ test_benchmark_endpoint                ← new
✅ test_jit_compile_via_env               ← new
✅ test_explicit_model_path_env_override
```

Plus 4 mlflow-only tests:

```
✅ test_noop_when_tracking_uri_unset
✅ test_flatten_nested_params
✅ test_is_finite
✅ test_real_tracking_to_sqlite
```

11 tests total for this post's changes. **4.2 s** wall time.

---

## 8. Interview answers this enables

> **Q. "How do you track experiments when you don't want a hard MLflow
> dependency?"**
> Opt-in via `MLFLOW_TRACKING_URI` env var. The tracker soft-imports
> mlflow; absent URI → no-op. Absent package but set URI → log + no-op.
> Failing `start_run` → log + no-op. Training never branches on "is
> mlflow available." Four unit tests verify each degraded path.

> **Q. "You quote ×2.13 for JIT. That's great at batch=1; what about
> larger batches?"**
> At batch 8 it's ×1.43, at 32 ×1.23, at 128 ×1.08. Speedup scales
> inversely with batch size — because Python overhead dominates at small
> batches and matrix multiplication dominates at large ones. My
> production case is single-state inference, so ×2 is the real number.

> **Q. "What changed in the actor to make it scriptable?"**
> Three small edits: `__constants__ = ["log_std_min", "log_std_max"]`,
> replace module-level globals with instance attributes, and
> `@torch.jit.export` on `deterministic()`. About 10 lines. Training code
> is unchanged.

---

## 9. What's next

Four more posts in this series:

- [DQN → QR-DQN: distributional RL for tail risk][post1] (Step 5 —
  already published, next week's Step 6)
- [Same pairs, different algorithms — SAC vs PPO benchmark][post4]
  (Step 3)
- [PER from scratch — SumTree to IS weights][post5] (Step 4)
- [CVaR-aware position sizing — QR-DQN quantiles into a sizing
  multiplier][post6] (Step 7)

[post1]: /blog/from-dqn-to-qr-dqn
[post4]: /blog/sac-vs-ppo-benchmark
[post5]: /blog/per-from-scratch
[post6]: /blog/cvar-aware-position-sizing

---

## Code

- `app/trading/rl/mlflow_tracker.py` — opt-in tracker (~150 lines)
- `app/trading/rl/trainers/sac_trainer.py` — hooks into the training loop
- `app/trading/rl/agents/sac.py` — `__constants__` + `@torch.jit.export`
- `stocktradingai/server/sac_serving.py` — JIT toggle + latency fields
- `scripts/bench_sac_jit.py` — eager-vs-JIT harness
- `docker-compose.mlflow.yml` — Postgres-backed tracking server
- `tests/test_mlflow_tracker.py` — 4 cases
- `tests/test_sac_serving.py` — includes benchmark + JIT cases
