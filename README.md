# RadCom-RL: radar vs. communication time-sharing for a connected vehicle

AI course CSL3090 (2026), Assignment 2: Mini Project [RL].
Divya Kumar (B23EE1019), IIT Jodhpur. Solo team.

A car has one shared radio front-end. Every 100 ms it must choose one job:
**sense** with the radar, or **send** V2X data. A DQN agent learns from road
hazard, channel quality and queue state to minimise missed hazards and dropped
urgent messages.

## Where this code comes from (please read first)

- **My B.Tech project.** The environment and the DQN training in `src/` come
  from my own B.Tech project, *"Intelligent Joint Radar-Communication System
  for Automotive Vehicles"* (supervisor: Prof. Pradyumna Kumar Bishoyi,
  Electrical Engineering, IIT Jodhpur), presented on 27 Oct 2026. That project
  was not a git repository, so the **first commit here imports it as it was**.
  The later commits add what was made for this course.
- **The base paper.** The model extends **iRDRC** (Hieu et al., IEEE Wireless
  Communications Letters, 2020). The idea of a DQN choosing radar or
  communication every step is theirs. What this project changes is listed in
  `src/config.py` (top of the file) and in the report.
- **The DQN library.** The DQN is **Stable-Baselines3** (Raffin et al., JMLR
  2021), installed with `pip` and imported, the library my supervisor pointed
  me to. No code from the Stable-Baselines3 repository, or from any other
  repository, is copied here.

Some code comments point to B.Tech scripts that are not part of this course
submission (the parameter sweeps for the B.Tech figures).

## How to run

Python 3.11 on Windows; any OS should work.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.sanity            # checks the environment against the maths
python -m src.train             # trains 5 DQN seeds per scenario
python -m src.figures           # learning-curve figures
```
