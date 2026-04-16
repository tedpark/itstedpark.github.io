---
title: "Hot-Reloading `.pt` Checkpoints with FastAPI — File `mtime` Is All You Need"
subtitle: "A tiny serving layer for RL agents that picks up new training runs without a restart. No watchers, no pub/sub, no Kubernetes."
date: "2026-04-12"
tags: ["FastAPI", "PyTorch", "MLOps", "Reinforcement Learning", "Python"]
summary: "The training loop drops a new `.pt` into cache/models/rl/stat_pair/ every few hours. The serving process should pick it up — without a restart, without a file watcher, without race conditions. I built a 270-line FastAPI router that does this with one `os.stat()` call per request. Seven end-to-end tests cover cold start, auto-swap, forced reload, and shape validation."
---

> **TL;DR.** A SAC training job writes
> `cache/models/rl/stat_pair/model_YYYY-MM-DD.pt` when it finishes. A
> FastAPI serving process should pick up the newer file automatically on
> the next request. The implementation is one class, one `os.stat()` per
> request, and a `threading.Lock` around the reload path. Seven e2e
> tests cover the full lifecycle.

---

## 0. Why this problem even exists

Serving a learned model sounds simple until the training loop produces a
new one. The common options:

| Approach | Why I rejected it |
|----------|-------------------|
| Rolling restart | Kills in-flight requests, wastes warm-up latency |
| Kubernetes rolling deploy | Overkill for a single-process serving box |
| File watcher (inotify / fsevents) | Extra dependency, platform-specific, subtle race conditions |
| Message queue pub/sub | Requires a second service (Redis/NATS), synchronization bugs |
| Poll on a timer | Wastes CPU when idle, and introduces reload latency jitter |

What I actually want is simpler: **detect the swap on the next request**,
without scheduling anything. A request hits the endpoint → stat the
checkpoint → if newer than what's loaded, reload → continue. `os.stat`
on a local file is sub-microsecond; the overhead is unmeasurable.

---

## 1. The API surface

Six endpoints, all under `/sac/*`:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/sac/health` | Quick "is a model loaded" check (public — no auth) |
| GET | `/sac/model/info` | Path, mtime, `state_dim`, `jit_compiled` flag |
| POST | `/sac/model/reload` | Ops-triggered hard reload (ignore mtime) |
| POST | `/sac/predict` | Single state → action |
| POST | `/sac/predict/batch` | Batch states → actions |
| POST | `/sac/benchmark` | Latency percentiles (warmup + timed loop) |

The `benchmark` one shows up again in the [MLflow + jit.script
post][post3]: it's how I proved the `torch.jit.script` compilation gave a
×2.13 speedup at batch size 1.

[post3]: /blog/mlflow-and-jit-script

---

## 2. Checkpoint resolution — three-tier priority

```python
def _resolve_checkpoint_path() -> Optional[Path]:
    explicit = os.environ.get("STAT_PAIR_RL_MODEL", "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None

    dir_env = os.environ.get("STAT_PAIR_RL_DIR", "").strip()
    model_dir = (
        Path(dir_env).expanduser().resolve()
        if dir_env
        else _DEFAULT_MODEL_DIR.resolve()
    )
    if not model_dir.is_dir():
        return None

    candidates = sorted(
        model_dir.glob(_MODEL_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
```

Priority:

1. `STAT_PAIR_RL_MODEL` env var — explicit absolute path override
2. `STAT_PAIR_RL_DIR` env var — scan that directory for `model_*.pt`
3. Default `cache/models/rl/stat_pair/`

If all three are missing, the endpoint returns `503` — the **server
still runs**, only `/sac/*` is disabled until a model appears. That's
non-negotiable: the rest of the API (positions, orders, health) must
not depend on SAC being available.

---

## 3. The singleton holder

One instance per process. Lazy-loaded. Thread-safe reloads.

```python
@dataclass
class _LoadedModel:
    agent: object                    # SACAgent (lazy-imported)
    path: Path
    mtime: float
    state_dim: int
    loaded_at: datetime
    jit_compiled: bool = False


class _ModelHolder:
    def __init__(self) -> None:
        self._model: Optional[_LoadedModel] = None
        self._lock = threading.Lock()

    def ensure_fresh(self) -> Optional[_LoadedModel]:
        path = _resolve_checkpoint_path()
        if path is None:
            return self._model

        try:
            mtime_on_disk = path.stat().st_mtime
        except OSError:
            return self._model

        current = self._model
        if (current is None
                or path != current.path
                or mtime_on_disk > current.mtime):
            self._reload(path, mtime_on_disk)
        return self._model
```

The flow on every request:

1. `ensure_fresh()` stat's the current canonical path
2. Compares the disk mtime against the in-memory one
3. If newer → call `_reload(path, mtime)`
4. Return the (possibly new) model handle

### Double-check inside the lock

This is the part that's easy to get wrong:

```python
def _reload(self, path: Path, mtime: float) -> None:
    with self._lock:
        # Re-check inside the lock — another thread may have beaten us
        if (self._model is not None
                and path == self._model.path
                and mtime <= self._model.mtime):
            return

        try:
            from app.trading.rl.agents.sac import SACAgent
        except Exception as e:
            logger.exception("SAC import failed; serving disabled")
            raise HTTPException(status_code=503, detail=f"SAC import failed: {e}")

        try:
            agent = SACAgent.load(str(path))
            agent.eval_mode()
        except Exception as e:
            logger.exception("SACAgent.load failed for %s", path)
            raise HTTPException(status_code=500, detail=f"Checkpoint load failed: {e}")

        state_dim = int(getattr(agent.config, "state_dim", _EXPECTED_STATE_DIM))
        self._model = _LoadedModel(
            agent=agent, path=path, mtime=mtime,
            state_dim=state_dim,
            loaded_at=datetime.now(timezone.utc),
        )
        logger.info("🔁 SAC hot-reloaded: %s", path.name)
```

The re-check is the classic double-checked-locking pattern: under
concurrency, two threads might both see a newer mtime outside the lock
and queue up to reload. The first one wins; the second (now inside the
lock) re-reads `self._model` and sees the swap is already done.

---

## 4. The endpoints — straightforward once the holder is right

```python
router = APIRouter(prefix="/sac", tags=["sac-serving"])

@router.post("/predict", response_model=PredictResponse)
async def sac_predict(req: PredictRequest) -> PredictResponse:
    cur = _holder.ensure_fresh()
    if cur is None:
        raise HTTPException(status_code=503, detail="No SAC model loaded")

    if len(req.state) != cur.state_dim:
        raise HTTPException(
            status_code=400,
            detail=f"state_dim mismatch: got {len(req.state)}, "
                   f"expected {cur.state_dim}",
        )

    state = np.asarray(req.state, dtype=np.float32)
    t0 = time.perf_counter()
    action = float(cur.agent.act(state, deterministic=req.deterministic))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PredictResponse(
        action=action,
        model_path=str(cur.path),
        model_mtime_utc=datetime.fromtimestamp(
            cur.mtime, tz=timezone.utc
        ).isoformat(),
        latency_ms=round(latency_ms, 4),
        jit_compiled=cur.jit_compiled,
    )
```

Two things worth noticing:

- Each response echoes `model_path` and `model_mtime_utc`. Clients
  debugging flaky results can immediately tell which checkpoint
  produced a given prediction.
- Latency is measured *inside* the handler so the client sees the real
  inference cost without network noise.

---

## 5. Lifecycle glue — preload at startup, don't fail

```python
# main.py lifespan hook
try:
    from sac_serving import preload_if_available
    loaded_path = preload_if_available()
    if loaded_path:
        logger.info("SAC serving: preloaded %s", loaded_path)
    else:
        logger.info("SAC serving: no checkpoint found — "
                    "/sac/* stays 503 until a model appears")
except Exception as e:
    logger.warning("SAC serving preload skipped: %s", e)
```

Three failure modes and how each is handled:

- **No checkpoint exists** → warn, keep running. Server is healthy.
- **Checkpoint import fails** (torch missing, bad file) → warn, keep
  running. `/sac/*` returns 503 until fixed.
- **Some other exception** → log + warn, keep running.

The pattern is "serving is always opt-in, never mandatory." Adding a
new capability must never take down the rest of the API.

---

## 6. The seven tests

The file is `tests/test_sac_serving.py`. Every test uses a fresh
temporary directory and creates a dummy SAC checkpoint with random
weights.

```python
def test_load_predict_and_hot_reload(serving_app):
    app, sac_serving, tmp_path = serving_app

    # Create first checkpoint
    ckpt1 = _make_dummy_checkpoint(tmp_path, "2026-04-01")
    mtime1 = ckpt1.stat().st_mtime

    # Preload (lifespan-equivalent)
    loaded_path = sac_serving.preload_if_available()
    assert loaded_path == ckpt1

    with TestClient(app) as client:
        # First request works with ckpt1
        r = client.post("/sac/predict", json={"state": [0.0] * 28})
        assert r.status_code == 200
        assert r.json()["model_path"] == str(ckpt1)

        # Hot reload: write a newer checkpoint
        time.sleep(0.05)  # make sure mtime differs
        ckpt2 = _make_dummy_checkpoint(tmp_path, "2026-04-02")
        assert ckpt2.stat().st_mtime > mtime1

        # Next /model/info picks up ckpt2 — no restart, no explicit reload
        r = client.get("/sac/model/info")
        assert r.json()["path"] == str(ckpt2)
```

That one test is the whole product: write a newer file, the next
request sees it. No waiting, no polling interval, no restart.

The other six:

```
✅ test_cold_start_no_model_returns_503
✅ test_force_reload_endpoint
✅ test_batch_shape_validation
✅ test_benchmark_endpoint
✅ test_jit_compile_via_env
✅ test_explicit_model_path_env_override
```

Full run: **14 seconds**, including the cost of `torch.save` +
`torch.load` for each dummy model.

---

## 7. Common questions I expected

> **"Why not use `fsevents` / `inotify`?"**
> Platform-specific, extra dependency, and introduces its own race
> conditions (events can arrive before the file finishes writing). `stat`
> on a local file is faster than people think, and "check on request" is
> exactly the right trigger — reload happens only when someone actually
> wants to serve.

> **"What about atomic renames?"**
> `SACAgent.save()` writes in one `torch.save` call, which does a single
> syscall write under the hood. Training is sequential, so there's no
> "half-written" window to catch mid-swap. If I ever moved to a
> write-to-tmp-then-rename pattern, the mtime check would still work —
> `os.rename` updates mtime atomically.

> **"Does the stat call slow down requests?"**
> One `os.stat` is ~1 microsecond on a warm kernel cache. Compared to the
> SAC actor forward pass (~100 microseconds for batch=1), it's noise.

> **"Why `threading.Lock` and not `asyncio.Lock`?"**
> Because `torch.save` / `torch.load` are CPU-bound blocking calls, not
> awaitable. The handler is `async` only because FastAPI wants it that
> way; the reload path is effectively synchronous. `threading.Lock` gives
> correct semantics in both sync and async workers.

---

## 8. What this was missing that Step 2 fixes

The careful reader will notice: the `benchmark` endpoint exists but
`jit_compiled` is always `False` unless you opt in. That's because
making the SAC actor `torch.jit.script`-compatible needed two small
changes (`__constants__` + `@torch.jit.export`) and gave a measured
×2.13 speedup at batch size 1 — the subject of the [next post][post3].

---

## 9. What's next

- [MLflow + `torch.jit.script` — experiment tracking + inference
  compile][post3] (Step 2)
- [Same pairs, different algorithms — SAC vs PPO benchmark][post4] (Step 3)
- [PER from scratch — SumTree to IS weights][post5] (Step 4)
- [CVaR-aware position sizing][post6] (Step 7) — the follow-up to the
  [QR-DQN post][post1]

[post1]: /blog/from-dqn-to-qr-dqn
[post4]: /blog/sac-vs-ppo-benchmark
[post5]: /blog/per-from-scratch
[post6]: /blog/cvar-aware-position-sizing

---

## Code

- `stocktradingai/server/sac_serving.py` — router + holder (~280 lines)
- `stocktradingai/server/main.py` — lifespan preload hook (~20 lines)
- `tests/test_sac_serving.py` — 7 e2e cases (~200 lines)
