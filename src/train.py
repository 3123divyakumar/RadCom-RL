"""Train a DQN on each scenario and record the learning curve.

The agent comes from Stable-Baselines3 rather than being hand-written: the paper
uses standard Mnih-style DQN with nothing custom, so the interesting work is the
environment, not a re-implementation of the algorithm. SB3's documented defaults
also give a defensible answer to the hyperparameters the paper never states.

    .venv\\Scripts\\python.exe -m src.train
    .venv\\Scripts\\python.exe -m src.train --steps 50000 --scenario highway

src/search_reward.py and src/sweep.py call train() directly.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from . import config
from .env import DFRCEnv

RUNS = Path(__file__).resolve().parent.parent / "runs"

# --- Why the rewards are divided before the agent sees them ------------------
#
# SB3's DQN trains with Huber loss, which is quadratic only while the error is
# below 1 and linear beyond it. Our rewards are +7 and -50, so every error lands
# far outside that window and the loss behaves like L1 - and L1 regression
# converges to the MEDIAN, not the mean. The median of "+7 on 89% of steps,
# -50 on 11%" is +7, so the agent learned that communicating is free and the
# miss penalty does not exist. Measured directly: it put Q(comm) at 6.71 in a
# state whose true value is -0.66, and settled on always-comm, which earns 60
# where 542 was available.
#
# Dividing every reward by the largest one it can be puts the errors back
# inside the quadratic region. With the paper's values that is the miss
# penalty, 50. Scaling by a positive constant cannot change which action is
# best, so the optimal policy is untouched; only the numbers the network fits
# change. The divisor follows the reward values so the reward search
# (src/search_reward.py) never leaves the quadratic region.
# Evaluation and every reported figure use the UNSCALED environment.


def reward_scale() -> float:
    """The largest reward, in either direction, the current values allow."""
    top_w = max(config.PRIORITY_WEIGHT.values())
    return max(config.PENALTY_MISS, config.PENALTY_URGENT_WAIT,
               config.PENALTY_INFO_WAIT,
               config.REWARD_DETECT * (config.URGENT_MIN_BAD + 1),
               config.REWARD_COMM_GOOD * top_w)


class ScaleReward(gym.RewardWrapper):
    """Shrink rewards into the range Huber loss handles as squared error."""

    def __init__(self, env):
        super().__init__(env)
        self.scale = reward_scale()

    def reward(self, r: float) -> float:
        return r / self.scale


LR_START = 1e-3
LR_END = 1e-4


def learning_rate(progress_remaining: float) -> float:
    """LR_START at the start, falling in a straight line to LR_END at the end.

    Why it falls (2026-09-29): a hazard is a coin flip, so every sample the
    network learns from is noisy - a comm step returns +6 or -50. At a fixed
    step size the Q-values keep jiggling by that noise for ever, and wherever
    two actions are close the jiggle flips the choice, which is the wobble in
    the old curve. Shrinking the step size late in training averages the noise
    out, so the policy settles and stays settled. Standard practice (Robbins-
    Monro: step sizes must shrink for a noisy estimate to converge).
    """
    return LR_END + (LR_START - LR_END) * progress_remaining


# Where the paper is silent, these are SB3's defaults unless noted.
HYPERPARAMS = dict(
    policy="MlpPolicy",
    learning_rate=learning_rate, # 1e-3 falling to 1e-4, see above
    buffer_size=100_000,         # SB3 default 1e6, unnecessary here
    learning_starts=2_000,
    batch_size=64,
    gamma=0.90,                  # the decision is near-myopic; only the queue
                                 # couples steps, and 0.99 inflates the values
                                 # until the gap between actions is unresolvable
    train_freq=4,
    target_update_interval=500,  # SB3 default 10_000, too slow for this length
    exploration_fraction=0.3,
    exploration_final_eps=0.02,
    policy_kwargs=dict(net_arch=[64, 64]),  # the paper's "two hidden layers"
    verbose=0,
)


def run_greedy(model, scenario: config.Scenario, episodes: int, seed: int) -> dict:
    """Greedy policy, no exploration, on episodes seed .. seed+episodes-1.

    All episodes run side by side so the network is asked once per step for
    every episode at the same time. Returns episode averages; waits are over
    every report delivered, in seconds.
    """
    envs = [DFRCEnv(scenario) for _ in range(episodes)]
    obs = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
    totals = np.zeros(episodes)
    for _ in range(config.EPISODE_STEPS):
        actions, _ = model.predict(obs, deterministic=True)
        results = [env.step(a) for env, a in zip(envs, actions)]
        obs = np.stack([r[0] for r in results])
        totals += [r[1] for r in results]
    infos = [r[4] for r in results]

    def share(sent, created):
        s, c = sum(i[sent] for i in infos), sum(i[created] for i in infos)
        return s / c if c else float("nan")

    def wait(key):
        w = [x for i in infos for x in i[key]]
        return float(np.mean(w)) * config.STEP_SECONDS if w else float("nan")

    return {
        "return": float(totals.mean()),
        "return_std": float(totals.std()),
        "miss_rate": float(np.mean([i["miss_rate"] for i in infos])),
        "throughput": float(np.mean([i["throughput"] for i in infos])),
        "routine_throughput": float(np.mean([i["routine_throughput"] for i in infos])),
        "radar_fraction": float(np.mean([i["radar_fraction"] for i in infos])),
        "info_delivered": share("info_sent", "info_created"),
        "urgent_delivered": share("urgent_sent", "urgent_created"),
        "info_wait_s": wait("info_waits"),
        "urgent_wait_s": wait("urgent_waits"),
    }


def evaluate(model, scenario: config.Scenario, episodes: int = 200,
             seed: int = 9_000) -> dict:
    """The final test: 200 fresh episodes. This is what the figures report."""
    return run_greedy(model, scenario, episodes, seed)


# --- The learning curve: the same test drive at every checkpoint -------------
#
# The raw training return mixes three things: how good the policy is, the
# random exploration still switched on, and which hazards happened to occur in
# that episode. Only the first is learning. So every CURVE_EVERY steps the
# current policy, greedy, drives the SAME CURVE_EPISODES episodes - same seeds,
# so the same weather, traffic, hazards and channel every time. Any change in
# the curve is then a change in the policy and nothing else, and once the
# policy stops changing the curve is flat. (This is how the DQN paper, Mnih et
# al. 2015, reported its curves: periodic evaluation, not training episodes.)
#
# The draws in a step do not depend on the action (env.py always draws the
# event, arrivals, conditions and channel in the same order), so fixed seeds
# really do give every checkpoint an identical world.
CURVE_EVERY = 2_000          # steps = 10 training episodes between points
CURVE_EPISODES = 50
CURVE_SEED = 20_000          # apart from the final evaluation's 9_000

# The last CURVE_TAIL of the curve is where "settled" is judged.
CURVE_TAIL = 0.25


def settled(returns) -> tuple[float, float]:
    """Mean and spread (max - min) over the last quarter of a curve."""
    r = np.asarray(returns, dtype=float)
    tail = r[-max(1, int(len(r) * CURVE_TAIL)):]
    return float(tail.mean()), float(tail.max() - tail.min())


class LearningCurve(BaseCallback):
    """Records the fixed test drive every CURVE_EVERY steps, and before training."""

    def __init__(self, scenario: config.Scenario):
        super().__init__()
        self.scenario = scenario
        self.rows: list[dict] = []

    def _record(self) -> None:
        point = run_greedy(self.model, self.scenario, CURVE_EPISODES, CURVE_SEED)
        self.rows.append({"step": self.num_timesteps,
                          "episode": self.num_timesteps // config.EPISODE_STEPS,
                          "return": point["return"],
                          "miss_rate": point["miss_rate"],
                          "throughput": point["throughput"]})

    def _on_training_start(self) -> None:
        self._record()

    def _on_step(self) -> bool:
        if self.num_timesteps % CURVE_EVERY == 0:
            self._record()
        return True


def policy_table(model, scenario: config.Scenario) -> list[dict]:
    """What the trained agent does in each condition state, channel and kind of
    packet waiting, at a mid queue.

    Compared against src/analysis.py, this is how we tell a converged policy
    from a plausible-looking reward curve.
    """
    flags = {"routine": (0, 0), "info": (1, 0), "urgent": (0, 1)}
    rows = []
    for e, v, m in itertools.product((0, 1), repeat=3):
        for c in (0, 1):
            for waiting, (i, u) in flags.items():
                obs = np.array([5 / config.QUEUE_CAPACITY, c, e, v, m, i, u],
                               dtype=np.float32)
                action, _ = model.predict(obs, deterministic=True)
                rows.append({"e": e, "v": v, "m": m, "channel_bad": c,
                             "waiting": waiting,
                             "action": "radar" if int(action) == 1 else "comm"})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def train(scenario: config.Scenario, steps: int, seed: int,
          out: Path | None = None, tag: str | None = None) -> dict:
    """Train one DQN and test it. Returns the final test plus how the curve settled.

    With `out`, the model, curve, policy table and training log are written
    there as <tag>_*. Without it (the reward search) nothing is written.
    """
    tag = tag or scenario.name
    inner = DFRCEnv(scenario)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        # Monitor sits OUTSIDE the scaler, so the logged training returns stay
        # in real reward units.
        inner = Monitor(inner, filename=str(out / f"{tag}_monitor"))
    model = DQN(env=ScaleReward(inner), seed=seed, **HYPERPARAMS)

    t0 = time.time()
    curve = LearningCurve(scenario)
    model.learn(total_timesteps=steps, callback=curve, progress_bar=False)
    mins = (time.time() - t0) / 60

    result = evaluate(model, scenario)
    result["curve_settled"], result["curve_spread"] = settled(
        [row["return"] for row in curve.rows])
    result["minutes"] = round(mins, 2)

    if out is not None:
        model.save(out / f"{tag}_dqn")
        write_csv(out / f"{tag}_curve.csv", curve.rows)
        write_csv(out / f"{tag}_policy.csv", policy_table(model, scenario))
    return result


# --- Several seeds -----------------------------------------------------------
# One DQN run on this problem sometimes switches between two nearly equal
# policies late in training (on the highway: hold routine packets, or send
# them). Tested 2026-09-29: neither the reward values, a lower final learning
# rate, bigger batches, a slower target network nor Polyak averaging stops it.
# So every figure reports SEEDS independent runs - their mean, with the
# min-max range as a band - which is standard practice in RL reporting.
SEEDS = (0, 1, 2, 3, 4)


def _job(name: str, steps: int, seed: int) -> dict:
    import torch
    torch.set_num_threads(1)
    r = train(config.SCENARIOS[name], steps, seed, out=RUNS, tag=f"{name}_s{seed}")
    return {"name": name, "seed": seed, **r}


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--workers", type=int, default=12,
                    help="trainings at a time, one CPU thread each")
    args = ap.parse_args()

    names = list(config.SCENARIOS) if args.scenario == "all" else [args.scenario]
    jobs = [(n, args.steps, s) for n in names for s in SEEDS]
    print(f"training {len(jobs)} runs ({len(SEEDS)} seeds per scenario), "
          f"{args.steps:,} steps each, in parallel...")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        results = list(pool.map(_job, *zip(*jobs)))

    for r in results:
        print(f"  {r['name']:<8} seed {r['seed']} | return {r['return']:7.1f} | "
              f"miss {r['miss_rate']:.3f} | throughput {r['throughput']:.2f} | "
              f"urgent wait {r['urgent_wait_s']:.2f} s | "
              f"spread {100 * r['curve_spread'] / abs(r['curve_settled']):.1f}%")
    means = []
    for name in names:
        rs = [r for r in results if r["name"] == name]
        mean = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]
                if k not in ("name", "seed")}
        means.append({"name": name, "seed": "mean", **mean})
        print(f"  {name:<8} MEAN   | return {mean['return']:7.1f} | "
              f"miss {mean['miss_rate']:.3f} | throughput {mean['throughput']:.2f} | "
              f"urgent wait {mean['urgent_wait_s']:.2f} s")

    write_csv(RUNS / "summary.csv", results + means)
    print(f"\nwrote {RUNS / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
