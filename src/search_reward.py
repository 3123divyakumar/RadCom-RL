"""Brute-force search for the reward values: train a DQN for every combination.

Asked for by Divya, 2026-09-29. The reward FORM stays the paper's Eq. (3) with
the priority weight w (config.py). What is searched is its numbers:

    r1   comm on a good channel      grid per round
    r2   comm on a bad channel       always r1 / 2 (the paper's ratio, and the
                                     4-vs-2 packets the two channels carry)
    r3   miss penalty                grid per round
    r4   detection reward            grid per round
    r5   urgent report held          grid per round (added in round 2)
    r6   road-info report held       grid per round (added in round 3)

One reward for both scenarios. Every combination is trained on both.

--- What "best" means ---------------------------------------------------------
Returns cannot be compared across reward functions: each one scores in its own
units. So every trained policy is judged on what it DOES, against requirements
taken from the V2X standards. A reward qualifies when, in BOTH scenarios, it
meets every requirement its round names:

  stable      the learning curve moves by at most MAX_SPREAD of its level over
              the last quarter of training
  urgent      at least MIN_URGENT_DELIVERED of urgent reports are delivered,
              waiting on average at most MAX_URGENT_WAIT_S in the buffer.
              3GPP TS 22.185 sets 100 ms as the maximum latency for V2X
              messages.
  status      routine status messages go out at MIN_ROUTINE_HZ or more.
              ETSI EN 302 637-2 lets the CAM rate fall to 1 Hz, and no lower.
  info        at least MIN_INFO_DELIVERED of road-info reports are delivered,
              waiting on average at most MAX_INFO_WAIT_S (round 3, below).

Among the rewards that qualify, the best misses the fewest hazards (the mean
of the two scenarios' miss rates); a tie goes to the higher throughput. With
several seeds, each scenario is judged on the mean of its runs, which is what
the figures report.

Results are appended to <round folder>/results.csv one training at a time, so
an interrupted search resumes where it stopped, and a different rule for
"best" can be applied later without training anything again (--rank).

    .venv\\Scripts\\python.exe -m src.search_reward            # run, then rank
    .venv\\Scripts\\python.exe -m src.search_reward --rank     # rank only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

RUNS = Path(__file__).resolve().parent.parent / "runs"

# Round 1 - the paper's form, r1..r4, 100 000 steps. Result: nothing
# qualified. Only r1 = 12, r3 = 25 sent urgent reports within 100 ms in both
# scenarios, and it missed 43 % of hazards; the safest rewards held urgent
# reports 0.5-3 s. Also, at 100 000 steps the curves had not settled to the
# 2 % test (median spread 2-4 %; the 200 000-step run gave 1.4 %).
#
# Round 2 - adds r5, the cost of holding an urgent report (config.py), and
# trains at the full 200 000 steps. r4 stays the paper's 5: in round 1 it
# moved the results least. Chosen: r1 4, r3 100, r5 20.
#
# Round 3 (2026-10-23, Divya's two changes): adds r6, the cost of holding a
# road-info report, and a catch is no longer charged r5 (config.py). Only r6
# is searched; r1..r5 stay at round 2's choice. Three seeds per value, since
# one run can settle on either of two close policies (train.py, SEEDS).
# Stability is not required: it failed in every round for every reward, it
# turned out to be a training effect rather than a reward one, and it is
# handled by reporting several seeds. The road-info requirement replaces it.
ROUNDS = {
    1: dict(out=RUNS / "search", steps=100_000, seeds=(0,),
            r1=(2, 3, 4, 6, 8, 10, 12), r3=(25, 50, 100), r4=(2.5, 5, 10), r5=(0,), r6=(0,),
            require=("stable", "urgent", "status")),
    2: dict(out=RUNS / "search_wait", steps=200_000, seeds=(0,),
            r1=(4, 6, 8), r3=(50, 100), r4=(5,), r5=(0, 2, 5, 10, 20), r6=(0,),
            require=("stable", "urgent", "status")),
    3: dict(out=RUNS / "search_info", steps=200_000, seeds=(0, 1, 2),
            r1=(4,), r3=(100,), r4=(5,), r5=(20,), r6=(0, 2, 5, 10, 15, 20),
            require=("urgent", "status", "info")),
}
ROUND = 3
OUT = ROUNDS[ROUND]["out"]
RESULTS = OUT / "results.csv"
SEARCH_STEPS = ROUNDS[ROUND]["steps"]
WORKERS = 12                      # of 16 logical cores, one thread each

MAX_SPREAD = 0.02
MIN_URGENT_DELIVERED = 0.99
MAX_URGENT_WAIT_S = 0.10          # 3GPP TS 22.185: 100 ms
MIN_ROUTINE_HZ = 1.0              # ETSI EN 302 637-2: CAM at 1 Hz or more

# ASSUMED (Claude, 2026-10-23; Divya: "you decide, as it should be below
# urgent"). Road-info ranks between urgent reports and routine status, so its
# limit sits between theirs: urgent must wait at most 0.1 s (3GPP), routine
# status may go out as seldom as once a second (ETSI CAM floor). 0.5 s is
# five times looser than urgent and twice as tight as status. Delivered: 95 %,
# not urgent's 99 %, because a report made in the last 0.5 s of a 20 s
# episode cannot go out by design (about 2.5 % at a 0.5 s wait). Rejected:
# urgent's own 0.1 s / 99 % (Divya: road-info ranks below urgent), and 1 s
# (no tighter than routine status).
MIN_INFO_DELIVERED = 0.95
MAX_INFO_WAIT_S = 0.5

KEYS = ["r1", "r2", "r3", "r4", "r5", "r6"]


def apply(r1: float, r3: float, r4: float, r5: float, r6: float) -> None:
    config.REWARD_COMM_GOOD = float(r1)
    config.REWARD_COMM_BAD = float(r1) / 2
    config.PENALTY_MISS = float(r3)
    config.REWARD_DETECT = float(r4)
    config.PENALTY_URGENT_WAIT = float(r5)
    config.PENALTY_INFO_WAIT = float(r6)


def job(r1: float, r3: float, r4: float, r5: float, r6: float,
        scenario: str, seed: int) -> dict:
    """One training, in its own process. Runs on one CPU thread."""
    import torch
    torch.set_num_threads(1)
    from .train import train

    apply(r1, r3, r4, r5, r6)
    result = train(config.SCENARIOS[scenario], SEARCH_STEPS, seed)
    return {"r1": r1, "r2": r1 / 2, "r3": r3, "r4": r4, "r5": r5, "r6": r6,
            "scenario": scenario, "seed": seed, **result}


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    for col, at in (("r5", 4), ("r6", 5)):     # rounds 1-2 predate them
        if col not in df:
            df.insert(at, col, 0.0)
    if "seed" not in df:                        # rounds 1-2 ran seed 0 only
        df.insert(df.columns.get_loc("scenario") + 1, "seed", 0)
    return df


def done_jobs() -> set[tuple]:
    if not RESULTS.exists():
        return set()
    df = load()
    return {(float(a), float(b), float(c), float(d), float(e), s, int(n))
            for a, b, c, d, e, s, n in zip(df.r1, df.r3, df.r4, df.r5, df.r6,
                                           df.scenario, df.seed)}


def search(workers: int = WORKERS) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    finished = done_jobs()
    g = ROUNDS[ROUND]
    todo = [(float(r1), float(r3), float(r4), float(r5), float(r6), s, seed)
            for r1, r3, r4, r5, r6 in itertools.product(g["r1"], g["r3"], g["r4"],
                                                          g["r5"], g["r6"])
            for s in config.SCENARIOS for seed in g["seeds"]
            if (float(r1), float(r3), float(r4), float(r5), float(r6), s, seed)
            not in finished]
    total = len(todo) + len(finished)
    print(f"{len(todo)} trainings to run ({len(finished)} of {total} already done), "
          f"{workers} at a time, {SEARCH_STEPS:,} steps each")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, *t) for t in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            new = not RESULTS.exists()
            with open(RESULTS, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            mins = (time.time() - t0) / 60
            print(f"  [{n:>3}/{len(todo)}] {mins:5.1f} min  "
                  f"r5={row['r5']:<4g} r6={row['r6']:<4g} "
                  f"{row['scenario']:<8} seed {row['seed']}  "
                  f"miss {row['miss_rate']:.3f}  "
                  f"urgent wait {row['urgent_wait_s']:.2f} s  "
                  f"road-info wait {row['info_wait_s']:.2f} s, "
                  f"delivered {row['info_delivered']:.2f}",
                  flush=True)


def rank() -> pd.DataFrame:
    """Apply the round's requirements and order the rewards. Writes ranking.csv."""
    g = ROUNDS[ROUND]
    df = load()
    df["spread_share"] = df.curve_spread / df.curve_settled.abs()
    metrics = ["miss_rate", "throughput", "routine_throughput", "radar_fraction",
               "urgent_delivered", "urgent_wait_s", "info_delivered", "info_wait_s",
               "spread_share"]
    by = df.groupby(KEYS + ["scenario"])
    m = by[metrics].mean()
    m = m[by.size() == len(g["seeds"])]          # every seed finished

    checks = {
        "stable": m.spread_share <= MAX_SPREAD,
        "urgent": ((m.urgent_delivered >= MIN_URGENT_DELIVERED)
                   & (m.urgent_wait_s <= MAX_URGENT_WAIT_S)),
        "status": (m.routine_throughput / config.STEP_SECONDS) >= MIN_ROUTINE_HZ,
        "info": ((m.info_delivered >= MIN_INFO_DELIVERED)
                 & (m.info_wait_s <= MAX_INFO_WAIT_S)),
    }
    m["meets"] = np.logical_and.reduce([checks[k] for k in g["require"]])

    wide = m.unstack("scenario")
    wide.columns = [f"{s}_{c}" for c, s in wide.columns]
    wide = wide.dropna(subset=[f"{s}_miss_rate" for s in config.SCENARIOS])
    wide["qualifies"] = (wide[[f"{s}_meets" for s in config.SCENARIOS]]
                         .fillna(False).astype(bool).all(axis=1))
    wide["mean_miss"] = wide[[f"{s}_miss_rate" for s in config.SCENARIOS]].mean(axis=1)
    wide["mean_throughput"] = wide[[f"{s}_throughput" for s in config.SCENARIOS]].mean(axis=1)

    wide = wide.sort_values(["qualifies", "mean_miss", "mean_throughput"],
                            ascending=[False, True, False]).reset_index()
    wide.to_csv(OUT / "ranking.csv", index=False)

    q = wide[wide.qualifies]
    print(f"\n{len(q)} of {len(wide)} reward functions meet every requirement "
          f"({', '.join(g['require'])}) in both scenarios, mean of "
          f"{len(g['seeds'])} seed(s).")
    cols = KEYS + ["qualifies", "mean_miss", "mean_throughput",
                   "highway_urgent_wait_s", "city_urgent_wait_s",
                   "highway_info_wait_s", "city_info_wait_s",
                   "highway_info_delivered", "city_info_delivered"]
    print(wide.head(12)[cols].to_string(index=False, float_format="%.3f"))
    print("\nwhy the others fail (per scenario): "
          + ", ".join(f"{k} {int((~checks[k]).sum())}" for k in g["require"]))
    return wide


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", action="store_true", help="rank existing results only")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="trainings at a time, one CPU thread each")
    args = ap.parse_args()
    if not args.rank:
        search(args.workers)
    rank()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
