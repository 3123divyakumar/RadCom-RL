"""What the reward implies, worked out without running anything.

For every condition state, each channel the agent can see, and each kind of
packet waiting, this computes the expected value of each action and says which
wins. It is the reference the trained agent gets compared against: if the DQN's
policy disagrees with this table, either the agent has not converged or it has
found something the myopic view misses.

What the agent sees is taken as it sees it, not averaged: the channel c, and
the flags that say whether the next send would carry an urgent, a road-info or
only a routine packet - which sets w in the reward.

Myopic is the caveat. This ignores what happens next: a report left waiting can
still go out later, and a detection now creates a report worth sending. Where a
margin is small (marked *), the trained agent can reasonably land either way.

    .venv\\Scripts\\python.exe -m src.analysis
"""

from __future__ import annotations

import itertools

from . import config
from .env import channel_bad_probability, event_probability

# What can be at the front of the buffer, lowest priority first.
WAITING = ("routine", "info", "urgent")

# A margin smaller than this, in reward units, is marked as a near-tie.
NEAR_TIE = 1.0


def state_probability(s: config.Scenario, e: int, v: int, m: int) -> float:
    return (((1 - s.tau_e) if e else s.tau_e)
            * ((1 - s.tau_v) if v else s.tau_v)
            * ((1 - s.tau_m) if m else s.tau_m))


def channel_bad_given_speed(s: config.Scenario, v: int) -> float:
    """Chance the channel is bad at this speed, averaged over the gap factor.

    The agent never sees the gap flag, so this is what it faces at a given v.
    """
    return ((1 - s.tau_g) * channel_bad_probability(s, 1, v)
            + s.tau_g * channel_bad_probability(s, 0, v))


def comm_reward(channel_bad: int) -> float:
    return config.REWARD_COMM_BAD if channel_bad else config.REWARD_COMM_GOOD


def comm_value(p_event: float, channel_bad: int, w: int) -> float:
    """The paper's form: the comm reward is paid only if no event occurs."""
    return ((1 - p_event) * comm_reward(channel_bad) * w
            - p_event * config.PENALTY_MISS)


def radar_value(p_event: float, b: int) -> float:
    return p_event * config.REWARD_DETECT * (b + 1)


def verdict(p_event: float, channel_bad: int, w: int, b: int) -> str:
    """Which action wins, with * where the margin is a near-tie."""
    margin = comm_value(p_event, channel_bad, w) - radar_value(p_event, b)
    return ("comm" if margin > 0 else "radar") + ("*" if abs(margin) < NEAR_TIE else "")


def report(s: config.Scenario) -> dict[str, float]:
    bad_rate = sum(((1 - s.tau_v) if v else s.tau_v) * channel_bad_given_speed(s, v)
                   for v in (0, 1))

    print(f"\n=== {s.name.upper()} ".ljust(80, "="))
    print(f"channel bad {bad_rate * 100:.1f}% of steps   "
          f"comm pays {config.REWARD_COMM_GOOD:g} good / "
          f"{config.REWARD_COMM_BAD:g} bad, times w   "
          f"miss penalty {config.PENALTY_MISS:g}   "
          f"detect bonus {config.REWARD_DETECT:g}x(b+1)")
    print()

    head = (f"{'e v m':<8}{'channel':>8}{'P(event)':>10}{'b':>3}"
            f"{'routine':>10}{'road-info':>11}{'urgent':>9}{'% steps':>10}")
    print(head)
    print("-" * len(head))

    rows = []
    for e, v, m in itertools.product((0, 1), repeat=3):
        p = event_probability(s, e, v, m)
        b = e + v + m
        ws = state_probability(s, e, v, m)
        q = channel_bad_given_speed(s, v)
        for c in (0, 1):
            rows.append((p, c, e, v, m, b, ws * (q if c else 1 - q)))
    rows.sort()

    comm_share = {k: 0.0 for k in WAITING}
    for p, c, e, v, m, b, share in rows:
        cells = {k: verdict(p, c, config.PRIORITY_WEIGHT[k], b) for k in WAITING}
        for k in WAITING:
            if cells[k].startswith("comm"):
                comm_share[k] += share
        print(f"{e} {v} {m}   {'bad' if c else 'good':>8}{p:>10.4f}{b:>3}"
              f"{cells['routine']:>10}{cells['info']:>11}{cells['urgent']:>9}"
              f"{share * 100:>9.1f}%")

    print()
    print("  share of steps where transmitting beats radar, by what is waiting:")
    print(f"    routine {comm_share['routine'] * 100:5.1f}%    "
          f"road-info {comm_share['info'] * 100:5.1f}%    "
          f"urgent {comm_share['urgent'] * 100:5.1f}%")
    print("  (with nothing to send, radar always wins)")
    return comm_share


def main() -> int:
    print("Myopic best action per condition state, channel and what is waiting.")
    print(f"* = margin under {NEAR_TIE:g} reward units: a near-tie.")
    for s in config.SCENARIOS.values():
        report(s)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
