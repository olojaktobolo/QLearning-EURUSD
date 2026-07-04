# Project Guide — RL Forex Trading (EUR/USD)

*Full evaluation as of 2026-07-05. Companion to [README.md](README.md). This document maps the project, records every issue and mistake found, and sets up the feasibility discussion.*

---

## 1. What this project is

An attempt to train reinforcement-learning agents (DQN, then PPO) to trade EUR/USD using hourly OHLC data and technical indicators, built on **stable-baselines3** + **gymnasium**, with a custom `TradingEnv`. Despite the repo name, there is no tabular Q-learning — `Qlearning.ipynb` is an early Colab scratchpad; the real work is DQN and PPO.

**Chronology (from git history + file contents):**

1. `Qlearning.ipynb` — Colab-era scratchpad. Mixed `gym`/`gymnasium` imports, `!pip install` cells, several generations of `TradingEnv`, appendix with older code. Historical value only.
2. `DQN.ipynb` — tidied DQN attempt (discrete actions: sell/hold/buy, all-in position sizing), plus a "PPO MODEL ATTEMPT" section bolted on at the end.
3. `PPO.ipynb` / `PPO.py` — current line of work. Continuous action space (−1…1 → fraction of balance to trade, capped at 10 %), 20-bar observation window, 8 parallel envs, 3 M timesteps. `PPO.py` is the script version of `PPO.ipynb` cell 0; the notebook also contains a discrete-action variant and test/plotting cells.

**Data:** `EURUSD-Hour.csv` — 10,439 hourly bars, 2022-11-25 → 2024-07-30 (~20 months). Columns: Date, Time, OHLC, TickVolume, Bar Volume.

**Artifacts:** `models/` (~572 MB of checkpoints: `periodic/` from DQN runs, `periodic2/` from PPO runs, plus `best_model.zip`, `interrupted_model.zip`), `logs/evaluations.npz`, `tensorboard_logs/`, `trading_results.csv` (a test rollout — note the "prices" column contains values like 0.39 and 0.99, which is Bug #1 in action).

---

## 2. File map

| File | Status | Notes |
|---|---|---|
| `PPO.py` | **Current** | Canonical training script (PPO, continuous actions) |
| `PPO.ipynb` | Current | Same code as notebook + discrete variant + evaluation/plotting |
| `DQN.ipynb` | Superseded | Previous DQN approach; keep for reference |
| `Qlearning.ipynb` | Legacy | Colab scratchpad; safe to delete once mined for ideas |
| `EURUSD-Hour.csv` | Data | Only dataset in the repo |
| `models/`, `logs/`, `tensorboard_logs/`, `trading_results.csv` | Artifacts | Now git-ignored |
| `qlearning/splits.py` | **Current** | Purged walk-forward / purged K-fold splitters (leak-free CV, universe-agnostic) |
| `tests/test_splits.py` | Current | Invariant checks for the splitters — `python -m tests.test_splits` |
| `pyproject.toml`, `poetry.lock` | Env | Poetry, Python ^3.12 |

Cleanup already applied: deleted `get-pip.py` (2.2 MB stray installer) and `DQN copy.ipynb` (byte-identical duplicate of `DQN.ipynb`); added `.gitignore`; removed `models/*.zip`, `logs/evaluations.npz`, `get-pip.py` from git tracking (files kept on disk).

---

## 3. Issues and mistakes

### 🔴 Critical — invalidate the results

**C1. The environment trades at *normalized* prices, not real prices.**
In `PPO.py` the whole DataFrame — including `Close` — is rolling z-score normalized (`PPO.py:99`), and the env then executes trades at that normalized "price" (`PPO.py:162`). Z-scores hover around 0 and go negative, so `units_to_buy = trade_amount / current_price` produces absurd or negative position sizes, and net worth is economically meaningless. `trading_results.csv` confirms it: recorded "prices" are 0.39–0.99 instead of ~1.05–1.10. The DQN env has the same flaw (`self.df = (df - df.mean()) / df.std()` then trades on `self.df['Close']`).
*Fix: keep the raw `Close` for accounting; normalize only the observation features.*

**C2. No train/validation/test split — anywhere.**
Every notebook and script trains on the full 20 months and evaluates on the same data. Any apparent profit is memorization, not edge. With ~10k bars per episode, 3 M timesteps ≈ ~290 passes (PPO) and 20 M ≈ ~2,000 passes (DQN) over the identical, deterministic episode (reset always starts at the same bar). The agent can simply memorize the price path.

**C3. `EvalCallback` evaluates on the training env itself.**
Both `PPO.py:262` and the DQN notebook pass the same (vectorized training) env to `EvalCallback`, so `best_model.zip` is "best on training data" — and evaluation interleaves with training on shared env state. A separate held-out eval env is required.

**C4. Look-ahead bias in DQN normalization.**
`(df - df.mean()) / df.std()` uses statistics of the *entire* dataset, leaking future information into every observation. (The PPO rolling normalization is backward-looking and fine in principle — the mistake there is C1, applying it to the traded price.)

**C5. DQN agent cannot see its own position.**
The DQN observation is just the current row of features — no balance, no position. But reward depends on position (buy is a no-op when long; sell is a no-op when flat). The task is thus partially observed in a way that makes the optimal policy unlearnable. (The PPO env does include balance and position — correct.)

### 🟠 Bugs — wrong behavior, fixable

**B1. Dead checkpoint-resume logic (DQN notebook, training cell).** The carefully written "resume from checkpoint + load replay buffer" branch is immediately overwritten by an unconditional `model = DQN(...)` on the next line — resume never happens.

**B2. Module-level random rollout in `PPO.py` (`PPO.py:217-223`).** A full ~10k-step random episode with per-step `print` runs at import time, outside the `if __name__ == '__main__'` guard. Under `spawn` multiprocessing every one of the 8 workers would re-run it; even under Linux `fork` it wastes a full episode and floods stdout on every launch.

**B3. Mislabeled indicator.** `supertrend()` in `PPO.py:77-79` actually computes the **Schaff Trend Cycle** (`STCIndicator`), not SuperTrend, and its `multiplier` argument is unused. Three "SuperTrend" features are therefore three STC variants.

**B4. Long-only env with a symmetric action space.** Actions span −1…1 (suggesting shorts), but positions are clipped at ≥ 0 (`PPO.py:182`). A sell with no position is a silent no-op — the agent burns half its action space learning nothing, and for a forex task the inability to short is itself a design gap.

**B5. Silent balance clamping.** `self.current_balance = max(self.current_balance, 0)` (`PPO.py:181`) can create money out of nothing when combined with C1's broken prices; clamps like this hide accounting bugs rather than prevent them.

**B6. Double normalization of boolean/derived columns.** `EMA_Crossover` (a boolean) and already-oscillating indicators (RSI, STC) get rolling-z-scored along with everything else (`PPO.py:99`), producing noisy, occasionally divide-by-near-zero features (guarded only by `1e-8`).

### 🟡 Design weaknesses — worth rethinking

- **W1. Feature bloat / collinearity:** ~26 columns, many near-duplicates (7 EMAs/SMAs of one series, 3 STCs, MACD *and* its signal). × 20-bar window ⇒ ~522-dim observation for an MlpPolicy. More noise than signal.
- **W2. Deterministic episode start:** every episode starts at bar 20. Random start offsets (or episode slicing) would give the 8 parallel envs decorrelated experience; currently all 8 envs replay the same path.
- **W3. Reward = one-step net-worth delta** (raw $ for DQN, fractional for PPO). No cost for churning, no risk adjustment (drawdown/Sharpe), and with hourly bars the per-step signal-to-noise is tiny.
- **W4. Transaction cost model is thin:** flat 0.0001 proportional fee; no spread, no slippage, no overnight swap — for EUR/USD the spread is the dominant real cost and it's absent.
- **W5. Hyperparameter grab-bag:** e.g. an `lr_schedule` is defined but never passed to PPO (`PPO.py:230`); DQN `buffer_size=2_000_000` vs 10k data rows; `total_timesteps=20_000_000` on 10k bars.
- **W6. `Bar Volume` column looks cumulative** (monotonically growing within the file) and is fed in as a raw feature; verify its meaning before using it.

### ⚪ Hygiene (mostly addressed in this cleanup)

- Empty `README.md` → now written. No `.gitignore` → added; 572 MB of checkpoints untracked from git.
- `get-pip.py` and byte-identical `DQN copy.ipynb` → deleted.
- Empty `qlearning/` package and `tests/` — either move code in or remove.
- `pyproject.toml` declares **both** `gym` and `gymnasium` (current code only needs `gymnasium`), plus unused `torchvision`/`torchaudio`.
- Branch is named `new-branch-name`; all recent work is uncommitted.
- Repo/package named "qlearning" though the project is DQN/PPO.

---

## 4. Suggested order of work (pre-feasibility)

1. **Fix C1** (trade on raw prices) — until then, no result means anything.
2. **Split the data** — e.g. train ≤ 2023-12, validate 2024-01→04, test 2024-05→07 — and give `EvalCallback` a real held-out env (fixes C2/C3).
3. **Guard `PPO.py`** (`if __name__ == '__main__':` around the test rollout) and delete dead code blocks.
4. **Random episode starts + shorter episodes** so parallel envs decorrelate.
5. Prune features (start with Close returns + 3–5 indicators), then re-run a short training as a smoke test.
6. Only after a clean pipeline: baselines (buy-and-hold, SMA crossover) to compare against — this is the feasibility gate.

## 5. Feasibility — the honest framing (preview for next step)

Two separate questions must not be conflated:

1. **Can the pipeline be made technically sound?** Yes — the fixes above are days, not months, of work.
2. **Can an RL agent profitably trade hourly EUR/USD from 20 months of OHLC + indicators?** This is the hard question. 10k bars is a very small dataset by RL standards; hourly FX is close to efficient after spread; every published positive result in this space struggles to survive out-of-sample testing. The realistic goal for this project is a *correct experimental rig* that can demonstrate whether any edge exists — with buy-and-hold and simple rule baselines as the bar to beat. Expect the answer on live profitability to lean negative; the project is still valuable as an RL engineering exercise.

To be discussed next.
