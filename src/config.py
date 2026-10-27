"""Every number the simulation uses, in one place.

Each value is tagged with where it came from:

    PAPER    the iRDRC paper states it.
    DATA     taken or derived from a real-world source, cited on the line.
    ASSUMED  Claude's choice with a stated reason. Divya can overrule any of these.
    DECIDE   still open. Left as None so nothing runs on a guess.

--- What differs from the paper, and why -------------------------------------

1. Road and weather are ONE factor `e` ("adverse conditions"), not two.
   Rain makes a road slippery, so treating them as independent is wrong.
   Consequence: the state is 5 numbers instead of 6, and `b` (the count of bad
   factors) maxes at 3, so the reward multiplier (b+1) runs 1..4, not 1..5.

2. Equation (1) uses the conditions actually observed this step, not the paper's
   printed form, which averages them away and gives the same risk in a storm as
   on a clear day.

3. The four risks are combined with noisy-OR, not addition. See COMBINE_RULE.

4. The channel is not one coin flip at the paper's 0.1. It follows a second
   equation of the same form as equation (1) - see "The channel" below - so it
   is worse between roadside units, at speed, and in dense traffic.

5. Packets have priorities - routine, road-info, urgent - and each hazard the
   radar detects becomes a report to send. See "Packet priority" below.

--- What one time step means -------------------------------------------------

STEP_SECONDS below is the anchor for every probability in this file. Without it
"probability per step" has no physical meaning and no real-world number can be
converted into one.
"""

from dataclasses import dataclass, fields


# --- Time base ---------------------------------------------------------------
# ASSUMED: 100 ms per step. Three reasons, in order of weight:
#   1. V2X safety messages (DSRC BSM / C-V2X CAM) are broadcast at 10 Hz, so one
#      message per 100 ms. That makes the paper's arrival rate of exactly
#      1 packet/step fall out of the standard instead of being arbitrary.
#   2. An automotive FMCW radar frame is roughly 20-50 ms, so a 100 ms step
#      comfortably contains one complete radar measurement.
#   3. At 100 km/h the vehicle travels 2.8 m per step - fine enough granularity
#      for a collision-avoidance decision.
STEP_SECONDS = 0.1


# --- Shared world constants (PAPER) ------------------------------------------
QUEUE_CAPACITY = 10       # PAPER: D, packets the buffer holds
ARRIVAL_RATE = 1.0        # PAPER: lambda_d, Poisson arrivals per step (= 10 Hz, see above)
PACKETS_GOOD_CHANNEL = 4  # PAPER: nu_1, sent per step on a good channel
PACKETS_BAD_CHANNEL = 2   # PAPER: nu_2, sent per step on a bad channel

# ASSUMED: 200 steps = 20 seconds of driving at 100 ms/step. The paper never
# states an episode length; its converged return of ~+300 implies something near
# 100-150. 200 gives roughly 10-16 events per episode - enough for the reward
# signal to be stable - at negligible extra cost.
EPISODE_STEPS = 200


# --- Reward: the paper's Eq. (3), extended with packet priority --------------
#
#     r = r1 * w       comm, good channel, no event
#         r2 * w       comm, bad channel, no event
#        -r3           comm while an event occurred (missed)
#         r4 * (b+1)   radar, event detected - in full, whatever is waiting
#         0            radar, no event, no report waiting
#       - r5           radar, no event, an urgent report waits (added 29 Sep)
#       - r6           radar, no event, a road-info report waits and no urgent
#                      one does (added 23 Oct)
#
# w is the priority of the most important packet sent (PRIORITY_WEIGHT, below).
# It scales the comm reward the way the paper's (b+1) scales detection. Where
# the three equations enter: equation (1) decides event / no event and b, the
# channel equation decides c, packet priority decides w.
#
# Values chosen 2026-09-29 by brute-force search (src/search_reward.py, Claude,
# free hand from Divya), replacing 6 / 3 / 50 / 5 of 24 Sep. Every candidate
# was trained, and judged on what its policy does, against the V2X standards:
# urgent reports delivered within 100 ms (3GPP TS 22.185), status messages at
# 1 Hz or more (ETSI EN 302 637-2); among those, the fewest missed hazards.
#   * r3 = 100, twice the paper's. Misses fall from ~9 % to 3.5 % (highway)
#     and 1.6 % (city), consistently over seeds. Throughput falls from
#     0.6-0.75 to 0.26-0.32 packets per step; status still goes out at
#     1.5-2 Hz.
#   * r1 = 4, r2 = 2: the paper's 2 : 1 ratio, which is also the 4-vs-2 packets
#     a good and a bad channel carry. Of 4, 6 and 8 it missed fewest.
#   * r4 = 5, the paper's. It moved the results least in round 1.
#   * r5 = 20, see PENALTY_URGENT_WAIT.
# Full results: runs/search/ (round 1) and runs/search_wait/ (round 2).
REWARD_COMM_GOOD = 4.0    # r1, comm on a good channel, no event (paper: 2)
REWARD_COMM_BAD = 2.0     # r2, comm on a bad channel, no event (paper: 1)
PENALTY_MISS = 100.0      # r3, comm while an event occurred (subtracted) (paper: 50)
REWARD_DETECT = 5.0       # PAPER: r4, radar during an event, scaled by (b+1)

# r5, added 2026-09-29 (Claude, free hand from Divya): the cost of each step an
# urgent report is left waiting while the agent uses radar. The paper's form
# has no such cost, so nothing tells the agent that an urgent report is
# time-critical - it held them ~0.3 s on the highway, and a brute-force search
# over r1..r4 (runs/search/) found no values that send them within 3GPP's
# 100 ms without missing far more hazards. With r5 = 10 or more they go out
# at once (<= 0.03 s) in every combination tried; 20 is the searched value
# with the fewest misses. 0 gives back the paper's form.
#
# Changed 2026-10-23 (Divya): a radar step that CATCHES a hazard is no longer
# charged r5. The catch is paid in full; r5 is charged only when the radar
# step finds nothing.
PENALTY_URGENT_WAIT = 20.0

# r6, added 2026-10-23 (Divya's decision; value searched, src/search_reward.py
# round 3): the cost of a radar step that finds nothing while a road-info
# report waits. Under 29 Sep's reward road-info reports waited 0.52 s
# (highway) and 1.41 s (city), and 1 in 10 city reports was never delivered,
# because holding one cost nothing - the problem r5 fixed for urgent reports.
# Charged only when no urgent report waits: the cost is the most important
# waiting report's, the way w takes the most important packet sent (Divya).
# 0 gives back 29 Sep's form. 2 is the searched value: the smallest that
# brings road-info under 0.5 s and 95 % delivered in both scenarios (search
# round 3); 5 and above miss more hazards. runs/search_info/.
PENALTY_INFO_WAIT = 2.0


# --- How the three risks combine into one probability ------------------------
# ASSUMED: noisy-OR, i.e. 1 - (1-p_e)(1-p_v)(1-p_m). Reasons:
#   * It cannot exceed 1.0. Plain addition reaches ~1.18 at the top of the
#     p_v_1 sweep, which is not a probability, and the sweep is exactly where
#     figures (b), (c) and (d) live. Clamping would paper over that with a
#     flat ceiling and silently distort the top of every curve.
#   * It means something: "at least one factor causes an event", which is the
#     standard way to combine independent causes.
#   * At the paper's own risk levels it is numerically close to addition
#     (0.250 vs 0.276 in the all-bad case), so we stay near the paper's
#     behaviour in the regime it actually reported, and only diverge where the
#     paper's own form breaks.
COMBINE_RULE = "noisy_or"


# --- The channel: a second equation, same form as equation (1) ---------------
# The paper flips one coin for the channel - bad with p_c = 0.1 every step,
# everywhere. Here the channel has its own equation, built like equation (1):
# yes/no factors, one probability each, combined by noisy-OR.
#
#     P(channel bad) = 1 - (1 - q_base) * (1 - q_gap) * (1 - q_fast)
#
#     q_base   traffic interference, per scenario, always on - the floor
#     q_gap    CHANNEL_BAD_IN_GAP if the car is in a gap between two roadside
#              units, out of range of both, else 0
#     q_fast   CHANNEL_BAD_WHEN_FAST if speed is above 60 km/h - the same v
#              the agent already sees - else 0
#
# Set by Divya, 2026-09-24: distance to the roadside unit, speed and traffic
# density are in. Object-nearby blocking and rain were left out.
#
# Band: 5.9 GHz, India's V2X band (DoT G.S.R. 466(E), 10 June 2026, which
# de-licenses C-V2X on-board units in 5875-5905 MHz). With rain left out, the
# band changes none of the numbers below.

# ASSUMED: 0.5. Out of range of both units, the link fails about as often as
# it works.
CHANNEL_BAD_IN_GAP = 0.5

# ASSUMED: 0.05, kept small. At speed the channel changes while a packet is
# still arriving, so the receiver's estimate of it goes stale - the reason
# IEEE 802.11bd added midambles (Anwar et al., PIMRC 2019). The unit is fixed,
# so only our own speed counts, which keeps the effect modest.
CHANNEL_BAD_WHEN_FAST = 0.05


# --- Packet priority ----------------------------------------------------------
# The buffer (QUEUE_CAPACITY packets) holds three classes, highest priority
# first:
#
#     urgent     a hazard the radar detected in the worst conditions
#     road-info  any other hazard the radar detected, reported to others
#     routine    status messages - the paper's Poisson arrivals, unchanged
#
# Two fixed rules, not learned - the agent still only picks radar or comm:
#   * sending serves urgent first, then road-info, then routine
#   * when the buffer is full, routine is dropped first, then road-info,
#     urgent last. Repeated status messages are the ones worth deleting; a
#     newer one replaces them anyway.
#
# A missed hazard creates no report: missing it also means nobody is warned.
# This mirrors real V2X traffic - periodic status messages (CAM, ETSI EN
# 302 637-2) and event-triggered warnings (DENM, ETSI EN 302 637-3).
#
# Set by Divya, 2026-09-24.

# ASSUMED, built on the paper's own severity count: a detected hazard is urgent
# when all three conditions are bad (b = 3). The paper already scales its
# detection reward by (b+1), so this adds no new number. It makes about 15% of
# hazards urgent on the highway and 5% in the city.
URGENT_MIN_BAD = 3

# w in the reward: what a transmit step is worth, by the most important packet
# it sends. Nothing to send pays nothing - the paper paid r1 even for an empty
# buffer, which rewarded transmitting nothing.
PRIORITY_WEIGHT = {"none": 0, "routine": 1, "info": 2, "urgent": 3}


@dataclass
class Scenario:
    """One driving environment. Highway and city differ only in these numbers."""

    name: str

    # Risk that an unexpected event occurs while a factor is in a given state.
    # These are NOT crash probabilities - see the note at the bottom of the file.
    p_e_0: float = 0.005   # PAPER: floor value used for every "good" state
    p_e_1: float = 0.0     # per scenario below
    p_v_0: float = 0.005   # PAPER
    p_v_1: float = 0.0     # per scenario below; this is the sweep axis
    p_m_0: float = 0.005   # PAPER: floor
    p_m_1: float = 0.0     # per scenario below

    # How often each factor sits in its GOOD state. This is what makes a highway
    # a highway and a city a city.
    tau_e: float = 0.0
    tau_v: float = 0.0
    tau_m: float = 0.0

    # The channel equation's two per-scenario numbers.
    #   tau_g   how often the car is within range of a roadside unit - the
    #           GOOD state of the gap factor, like the tau values above
    #   q_base  chance the channel is bad from traffic interference alone
    tau_g: float = 0.0
    q_base: float = 0.0

    # What "an object is nearby" means, in metres. Only used to document how
    # tau_m was derived; the simulation never reads it.
    nearby_metres: float = 0.0

    # Distance between roadside units, in metres. Only used to document how
    # tau_g was derived; the simulation never reads it.
    unit_spacing_metres: float = 0.0

    def missing(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is None]


HIGHWAY = Scenario(
    name="highway",

    # DATA/ASSUMED: rain raises crash risk ~38% on average (Texas 2006-2021,
    # Nature Sci. Rep. 2025), and wet road after a dry spell adds ~39.5% more.
    # The paper's rain-only value was 0.046; merging road slipperiness in and
    # accounting for high-speed loss of grip (longer stopping, hydroplaning)
    # puts the merged adverse state higher.
    p_e_1=0.08,

    # PAPER: 0.1 for speed above 60 km/h, kept unchanged. A highway is built for
    # this speed - grade separated, no cross traffic - so 60+ km/h here is
    # normal operation, not misuse.
    p_v_1=0.1,

    # ASSUMED: 0.03. An object nearby still creates a blind spot, but highways
    # are access-controlled with no side roads, so there are few places for a
    # hidden vehicle to emerge from. Lower than the city for that reason.
    p_m_1=0.03,

    # DATA: Jodhpur has 40-49 rainy days/year (Weather Spark / weather-and-climate),
    # i.e. 11-13% of days, giving 0.87. Divya chose the by-day figure rather than
    # the by-hour one (~0.98) because the merged factor is not only rain - a road
    # stays slippery after the rain stops, so adverse conditions persist well
    # beyond the few hours of actual precipitation.
    tau_e=0.87,

    # ASSUMED: 0.25, i.e. ~75% of vehicles above 60 km/h. The Bengaluru roadside
    # study's "rural" figure (0.55) is NOT a highway - those were roadside
    # locations around a metro region, not access-controlled expressways, which
    # are limited at 100-120 km/h. Marked assumed at Divya's instruction.
    tau_v=0.25,

    # ASSUMED: 0.14. "Nearby" = within 100 m (Divya's choice - at highway speed
    # you need the longer warning distance). Indian inter-urban highway density
    # runs 0-85 veh/km/lane (Springer, Transportation in Developing Economies);
    # taking 20 veh/km/lane as typical free-flow gives 2.0 vehicles expected
    # within 100 m, so P(none) = exp(-2.0) = 0.14 under a Poisson placement
    # assumption.
    tau_m=0.14,
    nearby_metres=100.0,

    # DATA/ASSUMED: 0.35, i.e. in a gap 65% of the time. 3GPP's highway V2X
    # layout spaces sites 1732 m apart (TR 37.885, Table 6.1.3-1). A unit
    # reaches about 300 m each way - the range NHTSA uses for 5.9 GHz safety
    # messages (Readiness of V2V Technology, DOT HS 812 014, 2014) - so 600 m
    # of every 1732 m is covered: 600 / 1732 = 0.35.
    tau_g=0.35,
    unit_spacing_metres=1732.0,

    # ASSUMED: 0.02. The city keeps the paper's 0.1 (below); the highway is
    # scaled down by traffic density, 20 against 120 vehicles/km/lane - the
    # same densities tau_m uses: 0.1 x 20/120 = 0.017, rounded to 0.02.
    q_base=0.02,
)

CITY = Scenario(
    name="city",

    # DATA/ASSUMED: 0.055. Same rain evidence as above, but lower speeds limit
    # how much reduced grip actually costs, so the merged adverse state sits
    # below the highway's.
    p_e_1=0.055,

    # ASSUMED: 0.15, raised above the paper's 0.1. Urban limits are 50-70 km/h
    # against 100-120 on highways (Wikipedia, Speed limits in India), so 60+
    # km/h is normal on a highway but well over the limit in a city, among
    # intersections, pedestrians and cross traffic.
    p_v_1=0.15,

    # ASSUMED: 0.09, three times the highway. The paper's own motivating picture
    # is a car emerging from a side road hidden behind a truck - that is an
    # intersection conflict, and city geometry is dense with intersections,
    # crossings and driveways. An occluding object in a city hides far more
    # places a vehicle can come from.
    p_m_1=0.09,

    # DATA: same as highway. Weather does not care what road you are on, and
    # Divya set tau_e equal across both scenarios.
    tau_e=0.87,

    # DATA: 0.80. Bengaluru roadside study of 172,164 vehicles at 25 locations,
    # May-Oct 2022 (Journal of Road Safety): urban median 46 km/h with an
    # interquartile range of 38-58, so more than 75% of vehicles are below the
    # 60 km/h threshold.
    tau_v=0.80,

    # ASSUMED: 0.09. "Nearby" = within 20 m (Divya's choice - in a city only
    # close objects meaningfully block the view). Congested Indian urban
    # arterials with mixed traffic run far denser than highways; taking
    # 120 veh/km/lane gives 2.4 vehicles expected within 20 m, so
    # P(none) = exp(-2.4) = 0.09 under the same Poisson assumption.
    tau_m=0.09,
    nearby_metres=20.0,

    # DATA/ASSUMED: 1.0, never in a gap. 3GPP's urban V2X layout spaces sites
    # 500 m apart (TR 37.885, Table 6.1.3-1); at about 300 m each way their
    # ranges overlap. Gaps are a highway problem.
    tau_g=1.0,
    unit_spacing_metres=500.0,

    # PAPER value, our reading: 0.1 is the paper's p_c, kept here as the
    # interference floor - dense traffic means many vehicles share the channel.
    q_base=0.10,
)

SCENARIOS = {s.name: s for s in (HIGHWAY, CITY)}


def check() -> None:
    """Raise unless every value has been decided. Call before any run."""
    problems = []
    if COMBINE_RULE is None:
        problems.append("COMBINE_RULE (sum | clamp | noisy_or)")
    for s in SCENARIOS.values():
        problems += [f"{s.name}.{n}" for n in s.missing()]
    if problems:
        raise ValueError(
            "These values have not been decided yet:\n  " + "\n  ".join(problems)
        )


# --- What the p values actually mean, and why they are not crash rates -------
#
# Read literally, p_v_1 = 0.1 per 100 ms step is one event every second. Real
# crashes happen roughly once per million vehicle-kilometres, so these cannot be
# crash probabilities and the paper never claims they are.
#
# They are the probability that a SAFETY-RELEVANT SITUATION arises in this step -
# an object entering the region the camera cannot see, which would become a
# collision only if it went undetected. That is a near-miss / conflict rate, and
# it is orders of magnitude more common than a crash.
#
# So the absolute scale is inherited from the paper, and real-world evidence is
# used for the RATIOS between states and between scenarios. Anchoring the scale
# to crash statistics instead would make events essentially never happen, and the
# agent would have nothing to learn from.

if __name__ == "__main__":
    try:
        check()
        print("All values decided.")
    except ValueError as e:
        print(e)
