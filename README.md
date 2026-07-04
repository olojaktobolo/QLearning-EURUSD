# RL Forex Trading — EUR/USD (DQN & PPO)

Experiments in training reinforcement-learning agents to trade EUR/USD on hourly bars, using [stable-baselines3](https://stable-baselines3.readthedocs.io/) (DQN and PPO) with a custom Gymnasium trading environment and technical-indicator features from the [`ta`](https://technical-analysis-library-in-python.readthedocs.io/) library.

> ⚠️ **Status: experimental / not yet valid.** The current pipeline has known critical bugs (it trades at normalized prices and has no train/test split), so existing results and saved models are not meaningful. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the full evaluation, issue list, and roadmap.

## Project layout

| Path | Purpose |
|---|---|
| `PPO.py` | **Current main script** — PPO agent, continuous action space (fraction of balance to buy/sell), 20-bar observation window, 8 parallel envs |
| `PPO.ipynb` | Notebook version of the above, plus a discrete-action variant and evaluation/plotting cells |
| `DQN.ipynb` | Earlier DQN approach (discrete sell/hold/buy, all-in sizing) — superseded, kept for reference |
| `Qlearning.ipynb` | Original Colab scratchpad — legacy |
| `EURUSD-Hour.csv` | Dataset: 10,439 hourly EUR/USD bars, 2022-11-25 → 2024-07-30 |
| `models/`, `logs/`, `tensorboard_logs/` | Training artifacts (git-ignored) |
| `PROJECT_GUIDE.md` | Full project evaluation: architecture, issues, roadmap, feasibility notes |

## How it works

1. **Data prep** — load hourly OHLC + volume, add ~18 indicator columns (EMAs, SMAs, RSI, MACD, Parabolic SAR, ADX, STC, Bollinger Bands), drop NaNs, rolling z-score normalize.
2. **Environment** — custom `TradingEnv(gymnasium.Env)`: observation is the last 20 bars of all features plus normalized balance and position; action is a scalar in [−1, 1] (buy/sell up to 10 % of balance, long-only); reward is the fractional change in net worth per step; 0.01 % proportional transaction cost.
3. **Training** — PPO (`MlpPolicy`) over 8 `SubprocVecEnv` workers, 3 M timesteps, TensorBoard logging, `EvalCallback` checkpointing.
4. **Evaluation** — roll the trained model through the environment, record net worth / actions / positions to `trading_results.csv`, plot equity curve.

## Setup

Requires Python ≥ 3.12 and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

## Usage

Train the PPO agent (writes checkpoints to `models/periodic2/`, logs to `tensorboard_logs/`):

```bash
poetry run python PPO.py
```

Monitor training:

```bash
poetry run tensorboard --logdir tensorboard_logs
```

Notebooks:

```bash
poetry run jupyter notebook
```

## Known limitations (read before trusting any output)

- Trades execute at **z-score-normalized prices**, so P&L is currently not economically meaningful.
- Trained and evaluated on the **same data** — no out-of-sample test exists yet.
- Long-only; no spread/slippage modeling; deterministic episode start.

Full list with fixes in [PROJECT_GUIDE.md](PROJECT_GUIDE.md#3-issues-and-mistakes).

## Disclaimer

Research/educational code. Nothing here is financial advice, and no model in this repo should be connected to real money.
