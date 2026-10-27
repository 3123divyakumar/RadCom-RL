"""Prove the environment is right before any learning touches it.

Three fixed policies, both scenarios, checked against arithmetic worked out
independently of the simulation. If always-comm's miss rate is not 1.0, or the
observed event or bad-channel rate does not match the analytic one, the
environment is wrong - and finding that here costs seconds instead of a wasted
training run.

The packet-priority rules are checked directly first: the buffer must send
urgent first and drop routine first. Then every hazard the radar detects must
become exactly one report, with the urgent share the arithmetic predicts.

    .venv\\Scripts\\python.exe -m src.sanity
"""

from __future__ import annotations

import itertools

import numpy as np

from . import config
from .env import (COMM, RADAR, DFRCEnv, channel_bad_probability, drop_by_priority,
                  event_probability, priority_weight, reward, send_by_priority)


def analytic_event_rate(s: config.Scenario) -> float:
    """Expected event probability per step, summed over all 8 condition states.

    Computed straight from the tau and p values without touching the simulation,
    so it is a genuinely independent check on both the world generation and the
    noisy-OR combination.
    """
    total = 0.0
    for e, v, m in itertools.product((0, 1), repeat=3):
        weight = ((1 - s.tau_e) if e else s.tau_e) \
               * ((1 - s.tau_v) if v else s.tau_v) \
               * ((1 - s.tau_m) if m else s.tau_m)
        total += weight * event_probability(s, e, v, m)
    return total


def analytic_channel_bad_rate(s: config.Scenario) -> float:
    """Expected share of steps with a bad channel, over the gap and speed states.

    Same idea as the event rate: straight from tau and the channel equation,
    never touching the simulation.
    """
    total = 0.0
    for g, v in itertools.product((0, 1), repeat=2):
        weight = ((1 - s.tau_g) if g else s.tau_g) \
               * ((1 - s.tau_v) if v else s.tau_v)
        total += weight * channel_bad_probability(s, g, v)
    return total


def analytic_urgent_share(s: config.Scenario) -> float:
    """Share of hazards that arise with b >= URGENT_MIN_BAD, i.e. become urgent.

    Under always-radar every hazard is detected and reported, so this is also
    the urgent share of all reports.
    """
    urgent = total = 0.0
    for e, v, m in itertools.product((0, 1), repeat=3):
        weight = ((1 - s.tau_e) if e else s.tau_e) \
               * ((1 - s.tau_v) if v else s.tau_v) \
               * ((1 - s.tau_m) if m else s.tau_m)
        rate = weight * event_probability(s, e, v, m)
        total += rate
        if e + v + m >= config.URGENT_MIN_BAD:
            urgent += rate
    return urgent / total


def check_priority_rules() -> list[str]:
    """The buffer's two fixed rules, on hand-built cases."""
    failures = []

    buffer = {"urgent": 1, "info": 2, "routine": 5}
    sent = send_by_priority(buffer, 2)
    if (sent != {"urgent": 1, "info": 1, "routine": 0}
            or buffer != {"urgent": 0, "info": 1, "routine": 5}):
        failures.append(f"send order: sent {sent}, left {buffer} - "
                        f"expected the urgent and one road-info out first")

    buffer = {"urgent": 1, "info": 1, "routine": 10}
    dropped = drop_by_priority(buffer, config.QUEUE_CAPACITY)
    if (dropped != {"urgent": 0, "info": 0, "routine": 2}
            or buffer != {"urgent": 1, "info": 1, "routine": 8}):
        failures.append(f"drop order: dropped {dropped}, left {buffer} - "
                        f"expected two routine dropped")

    buffer = {"urgent": 11, "info": 0, "routine": 0}
    dropped = drop_by_priority(buffer, config.QUEUE_CAPACITY)
    if dropped["urgent"] != 1 or sum(buffer.values()) != config.QUEUE_CAPACITY:
        failures.append(f"all-urgent overflow: dropped {dropped}, left {buffer}")

    return failures


def check_reward_rules() -> list[str]:
    """Every row of the paper-form reward, and the priority weight, by hand."""
    r1, r2 = config.REWARD_COMM_GOOD, config.REWARD_COMM_BAD
    r3, r4 = config.PENALTY_MISS, config.REWARD_DETECT
    cases = [
        # action, channel_bad, event, b, w, expected, what it is
        (COMM, 0, False, 1, 1, r1, "comm, good channel, routine"),
        (COMM, 0, False, 1, 3, 3 * r1, "comm, good channel, urgent"),
        (COMM, 1, False, 1, 2, 2 * r2, "comm, bad channel, road-info"),
        (COMM, 0, False, 1, 0, 0.0, "comm, nothing to send"),
        (COMM, 0, True, 2, 3, -r3, "comm during an event"),
        (RADAR, 1, True, 2, 1, r4 * 3, "radar, event, b = 2"),
        (RADAR, 0, False, 3, 3, 0.0, "radar, no event"),
    ]
    failures = []
    for action, c, event, b, w, expected, what in cases:
        got = reward(action, c, event, b, w)
        if got != expected:
            failures.append(f"reward, {what}: got {got}, expected {expected}")

    # r5 and r6: a radar step that finds nothing is charged for the most
    # important report it holds back; a catch is paid in full; comm is never
    # charged. While r6 is still 0 a stand-in value is used, so the r6 rows
    # test something.
    r5 = config.PENALTY_URGENT_WAIT
    saved_r6 = config.PENALTY_INFO_WAIT
    config.PENALTY_INFO_WAIT = r6 = saved_r6 or 7.0
    try:
        for action, event, b, w, urgent, info, expected, what in [
                (RADAR, False, 3, 3, True, False, -r5, "radar, no event, urgent waiting"),
                (RADAR, True, 3, 3, True, False, r4 * 4, "radar, event, urgent waiting"),
                (COMM, False, 3, 3, True, False, 3 * r1, "comm, urgent waiting"),
                (RADAR, False, 1, 2, False, True, -r6, "radar, no event, road-info waiting"),
                (RADAR, True, 1, 2, False, True, r4 * 2, "radar, event, road-info waiting"),
                (COMM, False, 1, 2, False, True, 2 * r1, "comm, road-info waiting"),
                (RADAR, False, 3, 3, True, True, -r5, "radar, no event, both waiting")]:
            got = reward(action, 0, event, b, w, urgent_waiting=urgent, info_waiting=info)
            if got != expected:
                failures.append(f"reward, {what}: got {got}, expected {expected}")
    finally:
        config.PENALTY_INFO_WAIT = saved_r6

    for buffer, expected in [({"urgent": 0, "info": 0, "routine": 0}, 0),
                             ({"urgent": 0, "info": 0, "routine": 4}, 1),
                             ({"urgent": 0, "info": 1, "routine": 4}, 2),
                             ({"urgent": 1, "info": 1, "routine": 4}, 3)]:
        if priority_weight(buffer) != expected:
            failures.append(f"priority weight of {buffer}: "
                            f"got {priority_weight(buffer)}, expected {expected}")
    return failures


def run(scenario: config.Scenario, policy, episodes: int, seed: int) -> dict:
    env = DFRCEnv(scenario)
    totals = {"reward": 0.0, "events": 0, "misses": 0, "sent": 0,
              "steps": 0, "radar": 0, "dmax": 0, "dmin": 10**9,
              "bad_channel": 0, "reports": 0, "urgent": 0}

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            totals["bad_channel"] += env.c    # the channel the agent sees now
            action = policy(env)
            obs, r, terminated, truncated, info = env.step(action)
            totals["reward"] += r
            totals["dmax"] = max(totals["dmax"], env.d)
            totals["dmin"] = min(totals["dmin"], env.d)
            done = terminated or truncated
        totals["events"] += info["events"]
        totals["misses"] += info["misses"]
        totals["reports"] += info["info_created"] + info["urgent_created"]
        totals["urgent"] += info["urgent_created"]
        totals["sent"] += info["events"] * 0  # placeholder, replaced below
        totals["steps"] += env.t
        totals["radar"] += info["radar_fraction"] * env.t
        totals["sent"] += env.packets_sent

    return {
        "return_per_episode": totals["reward"] / episodes,
        "event_rate": totals["events"] / totals["steps"],
        "channel_bad_rate": totals["bad_channel"] / totals["steps"],
        "miss_rate": totals["misses"] / totals["events"] if totals["events"] else 0.0,
        "throughput": totals["sent"] / totals["steps"],
        "radar_fraction": totals["radar"] / totals["steps"],
        "queue_min": totals["dmin"],
        "queue_max": totals["dmax"],
        "detected": totals["events"] - totals["misses"],
        "reports": totals["reports"],
        "urgent_share": (totals["urgent"] / totals["reports"]
                         if totals["reports"] else 0.0),
    }


POLICIES = {
    "always-radar": lambda env: RADAR,
    "always-comm": lambda env: COMM,
    "random": lambda env: int(env.np_random.random() < 0.5),
}


def main(episodes: int = 300, seed: int = 0) -> int:
    failures = check_priority_rules()
    print("Buffer rules (send urgent first, drop routine first): "
          + ("FAILED" if failures else "OK"))
    reward_failures = check_reward_rules()
    print("Reward table (every row, and the priority weight):   "
          + ("FAILED" if reward_failures else "OK"))
    failures += reward_failures

    for s in config.SCENARIOS.values():
        predicted = analytic_event_rate(s)
        predicted_bad = analytic_channel_bad_rate(s)
        predicted_urgent = analytic_urgent_share(s)
        print(f"\n=== {s.name.upper()} "
              f"(predicted: event {predicted:.4f}/step, "
              f"channel bad {predicted_bad:.4f}/step, "
              f"urgent share {predicted_urgent:.3f}) "
              f"{'=' * 4}")
        header = (f"{'policy':<14}{'return':>10}{'event/step':>12}"
                  f"{'bad ch':>9}{'urgent':>8}{'miss rate':>11}{'thruput':>9}"
                  f"{'radar':>8}{'queue':>9}")
        print(header)
        print("-" * len(header))

        for name, policy in POLICIES.items():
            r = run(s, policy, episodes, seed)
            print(f"{name:<14}{r['return_per_episode']:>10.1f}"
                  f"{r['event_rate']:>12.4f}{r['channel_bad_rate']:>9.4f}"
                  f"{r['urgent_share']:>8.3f}{r['miss_rate']:>11.3f}"
                  f"{r['throughput']:>9.2f}{r['radar_fraction']:>8.2f}"
                  f"{r['queue_min']:>4}-{r['queue_max']:<4}")

            # --- the checks ---------------------------------------------------
            tag = f"{s.name}/{name}"
            if abs(r["event_rate"] - predicted) > 0.006:
                failures.append(f"{tag}: event rate {r['event_rate']:.4f} "
                                f"!= predicted {predicted:.4f}")
            if abs(r["channel_bad_rate"] - predicted_bad) > 0.006:
                failures.append(f"{tag}: bad-channel rate "
                                f"{r['channel_bad_rate']:.4f} "
                                f"!= predicted {predicted_bad:.4f}")
            if r["reports"] != r["detected"]:
                failures.append(f"{tag}: {r['reports']} hazard reports != "
                                f"{r['detected']} hazards detected")
            if not (0 <= r["queue_min"] and r["queue_max"] <= config.QUEUE_CAPACITY):
                failures.append(f"{tag}: queue left 0-{config.QUEUE_CAPACITY}")

            if name == "always-radar":
                if r["misses" if "misses" in r else "miss_rate"] != 0.0:
                    failures.append(f"{tag}: radar-only must never miss an event")
                if r["throughput"] != 0.0:
                    failures.append(f"{tag}: radar-only must send nothing")
                if abs(r["urgent_share"] - predicted_urgent) > 0.015:
                    failures.append(f"{tag}: urgent share "
                                    f"{r['urgent_share']:.3f} != predicted "
                                    f"{predicted_urgent:.3f}")
            if name == "always-comm":
                if abs(r["miss_rate"] - 1.0) > 1e-9:
                    failures.append(f"{tag}: comm-only must miss every event "
                                    f"(got {r['miss_rate']:.3f})")
                if abs(r["throughput"] - config.ARRIVAL_RATE) > 0.05:
                    failures.append(f"{tag}: comm-only throughput "
                                    f"{r['throughput']:.2f} should track the "
                                    f"arrival rate {config.ARRIVAL_RATE}")
            if name == "random" and not (0.35 < r["miss_rate"] < 0.65):
                failures.append(f"{tag}: random should miss about half "
                                f"(got {r['miss_rate']:.3f})")

    print()
    if failures:
        print("FAILED")
        for f in failures:
            print("  " + f)
        return 1
    print("All checks passed. The environment behaves as the arithmetic says it should.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
