"""Figure (a): the DQN's learning curve, one panel per scenario.

The DQN is not compared with anything (Divya, 2026-09-29), so each panel holds
one curve: the average return of the current greedy policy on the same 50 test
episodes, measured every 10 training episodes (src/train.py, "The learning
curve"). Because the test episodes never change, a run's curve moves only when
its policy does. The line is the mean of the training runs (src/train.py,
SEEDS) and the band their range.

    .venv\\Scripts\\python.exe -m src.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .train import CURVE_EPISODES, SEEDS, settled

RUNS = Path(__file__).resolve().parent.parent / "runs"

# Validated categorical slot 1 (blue) plus neutral inks for annotation.
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"


def curves(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Episode axis, and one row of returns per seed."""
    frames = [pd.read_csv(RUNS / f"{name}_s{s}_curve.csv") for s in SEEDS]
    return frames[0]["episode"].values, np.stack([f["return"].values for f in frames])


def main() -> int:
    """One figure per scenario: runs/fig_a_<scenario>.png."""
    RUNS.mkdir(exist_ok=True)
    for name in config.SCENARIOS:
        x, runs = curves(name)
        r = runs.mean(axis=0)
        mean, spread = settled(r)

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.fill_between(x, runs.min(axis=0), runs.max(axis=0), color=SERIES,
                        alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, r, color=SERIES, linewidth=2.0, zorder=3,
                solid_capstyle="round")

        ax.annotate(f"settles at {mean:.0f}",
                    xy=(x[-1], r[-1]), xytext=(-6, 12), textcoords="offset points",
                    ha="right", fontsize=9, color=INK_2)

        ax.set_xlabel("training episode", fontsize=9.5, color=INK_2)
        ax.set_ylabel("average return per episode", fontsize=9.5, color=INK_2)
        ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.margins(x=0.02)

        fig.suptitle(f"(a) DQN learning curve — {name.capitalize()}", fontsize=13,
                     color=INK, x=0.012, ha="left", y=0.975, fontweight="bold")
        fig.text(0.012, 0.865,
                 f"Greedy policy on the same {CURVE_EPISODES} test episodes "
                 f"({config.EPISODE_STEPS} steps each), every 10 training episodes\n"
                 f"line: mean of {len(SEEDS)} training runs  ·  band: their range",
                 fontsize=8.5, color=INK_2, ha="left", linespacing=1.4)
        fig.tight_layout(rect=(0, 0, 1, 0.84))

        out = RUNS / f"fig_a_{name}.png"
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {out}  (last quarter: mean {mean:.1f}, spread {spread:.1f}, "
              f"{100 * spread / abs(mean):.1f}% of mean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
