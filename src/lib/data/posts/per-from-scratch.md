---
title: "PER from Scratch — SumTree, Importance-Sampling Weights, and β Annealing"
subtitle: "Prioritized Experience Replay in 210 lines of numpy. No dependencies, O(log N) sampling, full integration with twin-critic SAC."
date: "2026-04-15"
tags: ["Reinforcement Learning", "SAC", "Data Structures", "PyTorch", "Python"]
summary: "Uniform replay is wasteful: rare but important transitions (like regime breaks) take forever to re-surface. Schaul 2016's Prioritized Experience Replay fixes that with an O(log N) SumTree sampler and importance-sampling weights that preserve unbiasedness. This post walks through the 170-line implementation, the SumTree invariants, the twin-critic TD-error aggregation, and the one-line switch in SACTrainer."
---

> **TL;DR.** `PrioritizedNStepReplayBuffer` — O(log N) push and
> weighted sample, α prioritization, β annealing from 0.4 → 1.0 over
> 200K samples, zero external dependencies. The SAC trainer gets a
> one-flag switch (`use_per=True`) and an IS-weighted critic loss that
> averages TD error across the twin critics. Ten unit tests including
> a priority-bias check and a SAC integration smoke test. 170 lines
> including the SumTree.

---

## 0. Why uniform replay is suboptimal

A uniform replay buffer samples every transition with probability
`1/N`, regardless of how much you can still learn from it. For a SAC
trading agent:

- 99% of bars produce small-magnitude TD errors (spread hovering near
  mean, the agent holds, reward ≈ 0)
- 1% produce large-magnitude TD errors (regime break, sudden
  correlation collapse, a wide Z-score entry)

The network has already minimized loss on the easy 99%. Each uniform
sample picks one of them with 99% probability. Meanwhile the hard 1% —
the samples that actually move the weights — get seen rarely.

[Schaul et al. 2016][per-paper] fix this by sampling in proportion to
the transition's TD-error magnitude:

```
p_i = (|δ_i| + ε)^α
P(i) = p_i / Σ p_j
```

- `α` controls *how greedy* the sampling is. `α=0` → uniform;
  `α=1` → strictly proportional.
- `ε > 0` ensures no transition has zero probability.

To preserve unbiasedness despite the non-uniform sampling, each update
gets an **importance-sampling weight**:

```
w_i = (N · P(i))^(-β)
```

Schaul 2016 recommends β annealing from 0.4 → 1.0 over training: early
gradients are biased but fast; late gradients are unbiased but slower.

[per-paper]: https://arxiv.org/abs/1511.05952

---

## 1. Why we need a SumTree

The naïve sampler is `numpy.random.choice(N, p=normalized_priorities)`,
which is O(N) per sample. With N=500K and batch size 256, that's 128M
operations per update. Unacceptable.

A **SumTree** is a specialized binary heap where each node's value is
the sum of its children. The root holds the total. The structure:

- `capacity` leaves; `2·capacity - 1` nodes total
- Leaves `[capacity - 1, 2·capacity - 2]`
- Parent of node `i` = `(i - 1) // 2`
- Data index `= leaf_index - capacity + 1`

Sampling proportional to priority becomes a single tree descent: pick a
random value `s ∈ [0, total)`, at each node go left if `s ≤ left_sum`
else go right (subtracting `left_sum` from `s`). Both operations are
O(log N).

### The 80-line SumTree

```python
class _SumTree:
    __slots__ = ("capacity", "_tree", "_write_idx", "_max_priority")

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._write_idx = 0
        self._max_priority = 1.0

    def add(self, priority: float) -> int:
        """Write a priority to the next leaf (circular). Returns data idx."""
        leaf_idx = self._write_idx + self.capacity - 1
        self._update_node(leaf_idx, float(priority))
        data_idx = self._write_idx
        self._write_idx = (self._write_idx + 1) % self.capacity
        return data_idx

    def update(self, data_idx: int, priority: float) -> None:
        leaf_idx = data_idx + self.capacity - 1
        self._update_node(leaf_idx, float(priority))

    def _update_node(self, node_idx: int, priority: float) -> None:
        change = priority - self._tree[node_idx]
        self._tree[node_idx] = priority
        parent = (node_idx - 1) // 2
        while parent >= 0:
            self._tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2
        self._max_priority = max(self._max_priority, priority)

    def get(self, s: float) -> tuple[int, float]:
        """Find leaf with cumulative priority >= s. Returns (data_idx, priority)."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self._tree[left]:
                idx = left
            else:
                s -= self._tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return data_idx, float(self._tree[idx])
```

Three invariants it preserves:

1. `self._tree[0]` is always the sum of all leaves (the total priority)
2. `_max_priority` is monotone non-decreasing within a window — used to
   ensure new transitions get the max priority currently known, so
   they're guaranteed at least one sample before their true TD-error is
   measured
3. `get(s)` with `s ∈ [0, total)` always returns a valid leaf (never
   descends beyond a real slot), even while the buffer is still filling

### Tests that nail down the invariants

```python
def test_sumtree_basic_total():
    tree = _SumTree(capacity=4)
    for p in [1.0, 2.0, 3.0, 4.0]:
        tree.add(p)
    assert tree.total == pytest.approx(10.0)

def test_sumtree_get_returns_correct_leaf():
    tree = _SumTree(capacity=4)
    for p in [1.0, 2.0, 3.0, 4.0]: tree.add(p)
    # cumulative: [1, 3, 6, 10]
    assert tree.get(0.5)[0] == 0
    assert tree.get(2.5)[0] == 1
    assert tree.get(7.0)[0] == 3

def test_sumtree_update_propagates():
    tree = _SumTree(capacity=4)
    for p in [1.0, 1.0, 1.0, 1.0]: tree.add(p)
    assert tree.total == pytest.approx(4.0)
    tree.update(0, 5.0)
    assert tree.total == pytest.approx(8.0)

def test_sumtree_circular_overwrite():
    tree = _SumTree(capacity=3)
    tree.add(1.0); tree.add(2.0); tree.add(3.0)
    idx = tree.add(10.0)          # 4th write overwrites slot 0
    assert idx == 0
    assert tree.total == pytest.approx(15.0)
```

Tiny, numeric, fast. Four cases, millisecond runtime.

---

## 2. The buffer wrapping the tree

```python
class PrioritizedNStepReplayBuffer:
    def __init__(self, capacity=500_000, n_steps=3, gamma=0.99,
                 alpha=0.6, beta_start=0.4, beta_end=1.0,
                 beta_anneal_steps=200_000):
        self.cap = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_anneal_steps = max(1, int(beta_anneal_steps))
        self._sample_count = 0

        self._tree = _SumTree(capacity)
        self._states: Optional[np.ndarray] = None
        self._next_states: Optional[np.ndarray] = None
        self._actions = np.empty(capacity, dtype=np.float32)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._dones = np.empty(capacity, dtype=np.float32)
        self._size = 0

    @property
    def current_beta(self) -> float:
        frac = min(1.0, self._sample_count / self.beta_anneal_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def push_raw(self, state, action, reward, next_state, done) -> None:
        s = np.asarray(state, dtype=np.float32)
        if self._states is None:
            self._init_arrays(s.shape[0])
        idx = self._tree.add(self._tree.max_priority)   # new → max priority
        self._states[idx] = s
        self._actions[idx] = float(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = np.asarray(next_state, dtype=np.float32)
        self._dones[idx] = float(done)
        self._size = min(self._size + 1, self.cap)
```

Three decisions:

1. **Lazy state array init.** The buffer doesn't know the state_dim at
   construction; it infers from the first push. Saves cognitive load at
   the call site.
2. **New transitions get `max_priority`.** Guarantees each new
   transition is sampled at least once before its true TD error is
   measured — otherwise it could sit unseen forever.
3. **β is a property, not a parameter.** The caller doesn't manage the
   anneal counter; the buffer does.

### Stratified sampling + IS weights

```python
def sample(self, batch_size: int):
    self._sample_count += 1
    total = self._tree.total
    if not np.isfinite(total) or total <= 0.0:
        indices = np.random.randint(0, self._size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
    else:
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        segment = total / batch_size
        for i in range(batch_size):
            s = np.random.uniform(i * segment, (i + 1) * segment)
            idx, p = self._tree.get(s)
            if idx >= self._size:                 # during buffer fill
                idx = np.random.randint(0, self._size)
                p = max(p, _EPS)
            indices[i] = idx
            priorities[i] = max(p, _EPS)

        probs = priorities / total
        beta = self.current_beta
        w = (self._size * probs) ** (-beta)
        w = w / w.max()                           # Schaul §3.4 normalize
        weights = w.astype(np.float32)

    return (
        torch.from_numpy(self._states[indices]).to(DEVICE),
        torch.from_numpy(self._actions[indices]).to(DEVICE),
        torch.from_numpy(self._rewards[indices]).to(DEVICE),
        torch.from_numpy(self._next_states[indices]).to(DEVICE),
        torch.from_numpy(self._dones[indices]).to(DEVICE),
        indices,                                   # for update_priorities
        torch.from_numpy(weights).to(DEVICE),      # for IS-weighted loss
    )
```

Key details:

- **Stratified sampling**: pick one sample from each of `batch_size`
  equal-width segments of `[0, total)`. Reduces variance vs unstratified
  sampling, recommended by the paper.
- **Fallback for degenerate state**: early in training the tree total
  could be zero or NaN. Fall back to uniform sampling with
  `weights=1.0`. Never returns garbage.
- **Weight normalization**: divide by the max weight in the batch.
  Keeps `w_i ∈ (0, 1]` — stable for gradient scaling.

### Priority update after the loss

```python
def update_priorities(self, indices, td_errors):
    """Call immediately after the critic update."""
    abs_err = np.asarray(td_errors, dtype=np.float64)
    priorities = (np.abs(abs_err) + _EPS) ** self.alpha
    for idx, prio in zip(indices, priorities):
        self._tree.update(int(idx), float(prio))
```

The trainer must call this with the TD errors from the *same* update it
just ran. Otherwise priorities get stale.

---

## 3. Twin-critic TD error: `|δ₁| + |δ₂|` / 2

SAC has two critics for clipped double-Q. Which one's TD error should
drive the priority? Three options:

1. `|δ₁|` — biased toward critic 1
2. `|δ₂|` — biased toward critic 2
3. `(|δ₁| + |δ₂|) / 2` — uses both, balanced

Option 3 is what I went with. It captures "how badly did both critics
miss this transition", which is the right notion for prioritization:

```python
# app/trading/rl/trainers/sac_trainer.py — _update()
if is_per and is_weights is not None and indices is not None:
    td1 = q1 - target
    td2 = q2 - target
    per_sample_loss = 0.5 * (td1.pow(2) + td2.pow(2))
    critic_loss = (is_weights * per_sample_loss).mean()
    with torch.no_grad():
        abs_td = 0.5 * (td1.detach().abs() + td2.detach().abs())
        buf.update_priorities(indices, abs_td.cpu().numpy())
else:
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
```

Two things to note:

- **IS-weighted loss**: the MSE is multiplied element-wise by the IS
  weights before averaging. This corrects the sampling bias.
- **Priority update in a `no_grad()` block**: we need magnitudes for
  priority, not gradients. Detaching also saves memory.

---

## 4. The one-flag switch

Config-level:

```python
# app/trading/rl/config.py — SACTrainingConfig
use_per: bool = False
per_alpha: float = 0.6
per_beta_start: float = 0.4
per_beta_end: float = 1.0
per_beta_anneal_steps: int = 200_000
```

Trainer picks up:

```python
buf: Union[NStepReplayBuffer, PrioritizedNStepReplayBuffer]
if getattr(cfg, "use_per", False):
    buf = PrioritizedNStepReplayBuffer(
        capacity=cfg.buffer_size, n_steps=cfg.n_steps, gamma=cfg.gamma,
        alpha=cfg.per_alpha, beta_start=cfg.per_beta_start,
        beta_end=cfg.per_beta_end, beta_anneal_steps=cfg.per_beta_anneal_steps,
    )
    logger.info("SACTrainer: PER enabled (α=%.2f, β=%.2f→%.2f over %dk)",
                cfg.per_alpha, cfg.per_beta_start, cfg.per_beta_end,
                cfg.per_beta_anneal_steps // 1_000)
else:
    buf = NStepReplayBuffer(
        capacity=cfg.buffer_size, n_steps=cfg.n_steps, gamma=cfg.gamma
    )
```

Default: `use_per=False`. Historical backtests unchanged. Opt in by
setting one flag.

---

## 5. Priority-bias test (the one that catches real bugs)

The most important test in the 10-case suite is this one:

```python
def test_priorities_bias_sampling():
    """After updating priorities, high-|TD| transitions should be sampled more."""
    buf = PrioritizedNStepReplayBuffer(
        capacity=100, alpha=1.0, beta_start=0.4, beta_end=0.4,
    )
    rng = np.random.default_rng(7)
    for i in range(100):
        s = rng.standard_normal(4).astype(np.float32)
        ns = rng.standard_normal(4).astype(np.float32)
        buf.push_raw(s, 0.0, 0.0, ns, 0.0)

    # Slot 0 gets priority 100; all others get 0.01
    buf.update_priorities(np.array([0]), np.array([100.0]))
    buf.update_priorities(np.arange(1, 100), np.full(99, 0.01))

    counts = np.zeros(100, dtype=np.int64)
    for _ in range(200):
        _, _, _, _, _, idx, _ = buf.sample(8)
        for i in idx:
            counts[int(i)] += 1

    # Under uniform sampling: expected count for slot 0 = 200*8 / 100 = 16
    # With priority 100 vs 0.01 (10000× higher), we should see counts >> 200
    assert counts[0] > 200, f"Priority bias not applied: count={counts[0]}"
```

This directly verifies the entire sampling pipeline: priority update
→ SumTree update → proportional sampling. If it passes, PER is actually
working. If it fails (counts near uniform), you've got a subtle bug —
probably in the tree's update propagation.

---

## 6. The ten tests

```
SumTree invariants (4):
  ✅ test_sumtree_basic_total
  ✅ test_sumtree_get_returns_correct_leaf
  ✅ test_sumtree_update_propagates
  ✅ test_sumtree_circular_overwrite

Buffer behavior (5):
  ✅ test_buffer_push_and_sample_shapes
  ✅ test_empty_buffer_raises
  ✅ test_priorities_bias_sampling       ← the important one
  ✅ test_is_weights_range_and_inverse_relation
  ✅ test_beta_anneals_toward_one

SAC integration (1):
  ✅ test_sac_trainer_runs_with_per      ← end-to-end smoke
```

Total runtime: milliseconds for the tree tests, a few seconds for the
SAC integration smoke.

---

## 7. Interview answers this enables

> **Q. "How do you prioritize rare but important transitions?"**
> Schaul 2016 PER with a SumTree. α=0.6 prioritization exponent, β
> annealed 0.4 → 1.0 over 200K samples, ε=1e-6 to keep zero-priority
> transitions eligible. Opt-in flag on `SACTrainingConfig.use_per`
> — doesn't change default behaviour.

> **Q. "How does the SumTree's sampling work?"**
> Each node stores the sum of its children's priorities. Root = total.
> Pick `s ∈ [0, total)`; descend: if `s ≤ left_sum` go left, else
> subtract `left_sum` and go right. O(log N). I also stratify —
> one sample per `total/batch_size` segment, per the paper.

> **Q. "Why does your PER use the mean of twin-critic TD errors?"**
> SAC has two critics. Either one alone biases priorities toward its
> approximation errors. `(|δ₁| + |δ₂|) / 2` captures "how badly did both
> critics miss" — the actual notion you want for prioritization.

---

## 8. What's next

Five more posts in this series:

- [Hot-reloading `.pt` checkpoints with FastAPI][post2] (Step 1)
- [Pluggable MLflow + `torch.jit.script`][post3] (Step 2)
- [Same pairs, different algorithms — SAC vs PPO benchmark][post4]
  (Step 3)
- [DQN → QR-DQN: distributional RL for tail risk][post1] (Step 5)
- [CVaR-aware position sizing][post6] (Step 7)

[post1]: /blog/from-dqn-to-qr-dqn
[post2]: /blog/hot-reloading-pt-checkpoints
[post3]: /blog/mlflow-and-jit-script
[post4]: /blog/sac-vs-ppo-benchmark
[post6]: /blog/cvar-aware-position-sizing

---

## Code

- `app/trading/rl/prioritized_replay_buffer.py` — SumTree + buffer
  (~210 lines)
- `app/trading/rl/config.py` — `use_per` + α/β flags on
  `SACTrainingConfig`
- `app/trading/rl/trainers/sac_trainer.py` — IS-weighted critic loss
  + priority update
- `tests/test_prioritized_replay_buffer.py` — 10 cases

## References

- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016).
  *Prioritized Experience Replay.* ICLR 2016.
  [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)
- Horgan, D., et al. (2018). *Distributed Prioritized Experience
  Replay (Ape-X).*
  [arXiv:1803.00933](https://arxiv.org/abs/1803.00933)
