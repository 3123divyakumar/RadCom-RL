"""The driving environment: one vehicle, one shared front-end, one choice per step.

Every 100 ms the vehicle either SENSES (radar) or TRANSMITS (communication).
It cannot do both. Choosing to transmit leaves it blind for that step, so an
unexpected event occurring in that step goes undetected.

The agent sees seven things and must decide which risk to take. What it never
sees is whether an event is happening right now - only the radar reveals that,
and only by giving up the step.

Ordering within a step matters and is easy to get wrong:

    1. the event is drawn for the state the agent has ALREADY observed
    2. the agent acts, without knowing the outcome of that draw
    3. reward is settled, revealing whether the gamble paid off
    4. the world moves on - the buffer sends (urgent first) if we transmitted,
       new packets arrive (including a report of any hazard the radar found),
       then new conditions, then the channel drawn from them

Draw the event after the action and the agent becomes clairvoyant. Draw it from
next step's conditions and it can never learn anything. Step 1 has to come first.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config

COMM = 0
RADAR = 1

# Packet classes, highest priority first. Sending walks this order; dropping
# walks it backwards, so routine goes first and urgent last.
PRIORITY = ("urgent", "info", "routine")


def send_by_priority(buffer: dict[str, int], capacity: int) -> dict[str, int]:
    """Take up to `capacity` packets out of the buffer, highest priority first.

    Updates `buffer` in place and returns how many of each class were sent.
    """
    sent = {}
    for cls in PRIORITY:
        n = min(capacity, buffer[cls])
        buffer[cls] -= n
        capacity -= n
        sent[cls] = n
    return sent


def drop_by_priority(buffer: dict[str, int], limit: int) -> dict[str, int]:
    """Drop packets until the buffer holds at most `limit`, lowest priority first.

    Updates `buffer` in place and returns how many of each class were dropped.
    """
    excess = sum(buffer.values()) - limit
    dropped = {}
    for cls in reversed(PRIORITY):
        n = min(max(excess, 0), buffer[cls])
        buffer[cls] -= n
        excess -= n
        dropped[cls] = n
    return dropped


def priority_weight(buffer: dict[str, int]) -> int:
    """w in the reward: the weight of the most important packet a send carries."""
    for cls in PRIORITY:
        if buffer[cls]:
            return config.PRIORITY_WEIGHT[cls]
    return config.PRIORITY_WEIGHT["none"]


def event_probability(scenario: config.Scenario, e: int, v: int, m: int) -> float:
    """Chance an unexpected event occurs, given the conditions actually observed.

    This is the project's reading of the paper's equation (1). The paper averages
    each factor over its prior, which returns the same number every step - the
    same risk in a storm as on a clear day - so the conditions in the state would
    carry no information. Here each factor contributes the risk of the state it
    is actually in.

    They are combined by noisy-OR: the probability that AT LEAST ONE factor
    produces an event. Adding them, as the paper does, exceeds 1.0 at the top of
    the p_v_1 sweep, and a value above 1 is not a probability.
    """
    p_e = scenario.p_e_1 if e else scenario.p_e_0
    p_v = scenario.p_v_1 if v else scenario.p_v_0
    p_m = scenario.p_m_1 if m else scenario.p_m_0

    rule = config.COMBINE_RULE
    if rule == "noisy_or":
        return 1.0 - (1.0 - p_e) * (1.0 - p_v) * (1.0 - p_m)
    if rule == "clamp":
        return min(1.0, p_e + p_v + p_m)
    if rule == "sum":
        return p_e + p_v + p_m
    raise ValueError(f"unknown COMBINE_RULE: {rule!r}")


def channel_bad_probability(scenario: config.Scenario, g: int, v: int) -> float:
    """Chance the channel is bad this step: the channel equation.

    Built like equation (1): yes/no factors, one probability each, combined
    by noisy-OR. Traffic interference (q_base) is always on; being in a gap
    between roadside units (g) and driving fast (v) each add their own chance.
    """
    q_gap = config.CHANNEL_BAD_IN_GAP if g else 0.0
    q_fast = config.CHANNEL_BAD_WHEN_FAST if v else 0.0
    return 1.0 - (1.0 - scenario.q_base) * (1.0 - q_gap) * (1.0 - q_fast)


def reward(action: int, channel_bad: int, event: bool, b: int, w: int,
           urgent_waiting: bool = False, info_waiting: bool = False) -> float:
    """What the step was worth: the paper's Eq. (3), with packet priority.

    Five cases, and only five. `b` is how many of the three conditions are
    unfavourable (0-3), so the detection bonus scales 1x to 4x - paying more for
    catching an event when conditions were bad. `w` is the priority of the most
    important packet sent (0 nothing, 1 routine, 2 road-info, 3 urgent) and
    scales the comm reward the same way.

    Two additions, for a radar step that finds nothing while a report is held
    back another step: r5 (config.PENALTY_URGENT_WAIT) if an urgent report
    waits, else r6 (config.PENALTY_INFO_WAIT) if a road-info report does. Only
    the most important waiting report is charged. A radar step that catches
    a hazard is paid in full, whatever waits (Divya, 2026-10-23).
    """
    if action == COMM:
        if event:
            return -config.PENALTY_MISS                       # blind at the worst moment
        r_comm = config.REWARD_COMM_BAD if channel_bad else config.REWARD_COMM_GOOD
        return r_comm * w                                     # worth what it carried
    # RADAR
    if event:
        return config.REWARD_DETECT * (b + 1)                 # caught it
    if urgent_waiting:
        return -config.PENALTY_URGENT_WAIT                    # nothing there, urgent held
    if info_waiting:
        return -config.PENALTY_INFO_WAIT                      # nothing there, road-info held
    return 0.0                                                # nothing there, step spent


class DFRCEnv(gym.Env):
    """Radar-versus-communication mode selection for one vehicle.

    Observation (7 values, all scaled to 0-1 so the queue does not dominate the
    network input):

        d  queue occupancy, 0-10 packets of any class, divided by capacity
        c  channel, 0 good / 1 bad
        e  adverse conditions - rain and/or slippery road
        v  speed above 60 km/h
        m  object nearby, creating a blind spot
        i  a road-info packet is waiting
        u  an urgent packet is waiting

    The channel c comes from the channel equation: traffic interference, a gap
    between roadside units (g), and speed (v). The gap flag is drawn every step
    but not observed - the agent sees the channel itself, which is what the
    decision and the reward use.

    The buffer holds three classes of packet (config.py, "Packet priority"):
    routine status messages, and road-info or urgent reports of hazards the
    radar detected. It always sends urgent first and drops routine first.

    Action: 0 = communicate, 1 = use radar.
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario: config.Scenario | str = config.HIGHWAY,
                 episode_steps: int | None = None):
        super().__init__()
        if isinstance(scenario, str):
            scenario = config.SCENARIOS[scenario]
        self.scenario = scenario
        self.episode_steps = episode_steps or config.EPISODE_STEPS

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    # --- world generation ----------------------------------------------------

    def _sample_conditions(self) -> tuple[int, int, int]:
        """Roll the three environment factors. tau is P(factor is FAVOURABLE)."""
        s = self.scenario
        e = int(self.np_random.random() >= s.tau_e)
        v = int(self.np_random.random() >= s.tau_v)
        m = int(self.np_random.random() >= s.tau_m)
        return e, v, m

    def _sample_gap(self) -> int:
        """1 if the car is out of range of every roadside unit. tau_g is P(in range)."""
        return int(self.np_random.random() >= self.scenario.tau_g)

    def _sample_channel(self) -> int:
        """Draw the channel from this step's gap and speed - call after both."""
        p_bad = channel_bad_probability(self.scenario, self.g, self.v)
        return int(self.np_random.random() < p_bad)

    @property
    def d(self) -> int:
        """Packets waiting, all classes together."""
        return sum(self.buffer.values())

    def _observation(self) -> np.ndarray:
        return np.array(
            [self.d / config.QUEUE_CAPACITY, self.c, self.e, self.v, self.m,
             int(self.buffer["info"] > 0), int(self.buffer["urgent"] > 0)],
            dtype=np.float32,
        )

    # --- gym API -------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.buffer = {cls: 0 for cls in PRIORITY}
        self.e, self.v, self.m = self._sample_conditions()
        self.g = self._sample_gap()
        self.c = self._sample_channel()
        self.t = 0

        # episode counters, reported in `info` at the end
        self.events = 0
        self.misses = 0
        self.packets_sent = 0
        self.packets_dropped = 0
        self.radar_steps = 0
        self.created = {cls: 0 for cls in PRIORITY}
        self.sent = {cls: 0 for cls in PRIORITY}
        self.dropped = {cls: 0 for cls in PRIORITY}

        # When each waiting report was created, oldest first, so the time it
        # spends in the buffer can be measured. Routine is not tracked: a newer
        # status message replaces an old one, so its wait means nothing.
        self.born = {"urgent": deque(), "info": deque()}
        self.waits = {"urgent": [], "info": []}

        return self._observation(), {}

    def step(self, action: int):
        action = int(action)
        s = self.scenario

        # 1. the event is drawn for the conditions the agent has already seen
        p_event = event_probability(s, self.e, self.v, self.m)
        event = bool(self.np_random.random() < p_event)
        b = self.e + self.v + self.m

        # 2 & 3. the agent has acted; settle what it was worth. w is read before
        #    the buffer sends, and the send serves the most important first.
        w = priority_weight(self.buffer)
        r = reward(action, self.c, event, b, w,
                   self.buffer["urgent"] > 0, self.buffer["info"] > 0)

        if event:
            self.events += 1
        if action == RADAR:
            self.radar_steps += 1
        elif event:
            self.misses += 1          # transmitting while an event occurred

        # 4. the world moves on. The buffer only drains if we transmitted,
        #    urgent first. Then new packets arrive: routine status messages,
        #    plus a report of any hazard the radar just found. If that
        #    overflows the buffer, routine is dropped first.
        if action == COMM:
            capacity = (config.PACKETS_BAD_CHANNEL if self.c
                        else config.PACKETS_GOOD_CHANNEL)
            for cls, n in send_by_priority(self.buffer, capacity).items():
                self.sent[cls] += n
                self.packets_sent += n
                # Wait = steps spent in the buffer before this send. A report
                # created in step t can first go out in step t+1, which counts
                # as a wait of 0.
                for _ in range(n if cls in self.born else 0):
                    self.waits[cls].append(self.t - self.born[cls].popleft() - 1)

        arrivals = {cls: 0 for cls in PRIORITY}
        arrivals["routine"] = int(self.np_random.poisson(config.ARRIVAL_RATE))
        if action == RADAR and event:
            arrivals["urgent" if b >= config.URGENT_MIN_BAD else "info"] = 1
        for cls, n in arrivals.items():
            self.buffer[cls] += n
            self.created[cls] += n
            if cls in self.born:
                self.born[cls].extend([self.t] * n)
        for cls, n in drop_by_priority(self.buffer, config.QUEUE_CAPACITY).items():
            self.dropped[cls] += n
            self.packets_dropped += n
            for _ in range(n if cls in self.born else 0):
                self.born[cls].pop()      # the newest copy goes; which one is moot

        self.e, self.v, self.m = self._sample_conditions()
        self.g = self._sample_gap()
        self.c = self._sample_channel()

        self.t += 1
        truncated = self.t >= self.episode_steps

        info = {}
        if truncated:
            info = {
                "events": self.events,
                "misses": self.misses,
                "miss_rate": self.misses / self.events if self.events else 0.0,
                "throughput": self.packets_sent / self.t,
                "packets_dropped": self.packets_dropped,
                "radar_fraction": self.radar_steps / self.t,
                "info_created": self.created["info"],
                "info_sent": self.sent["info"],
                "info_dropped": self.dropped["info"],
                "urgent_created": self.created["urgent"],
                "urgent_sent": self.sent["urgent"],
                "urgent_dropped": self.dropped["urgent"],
                "routine_throughput": self.sent["routine"] / self.t,
                "urgent_waits": list(self.waits["urgent"]),
                "info_waits": list(self.waits["info"]),
            }

        return self._observation(), r, False, truncated, info
