"""Compare the trained DQN with three baseline policies; draw the report figures.

Added for the AI course (CSL3090, A2). The baselines are the three fixed
policies src/sanity.py already uses to check the environment: always use
radar, always communicate, and a fair coin each step. Every policy drives the
same 200 test episodes as the DQN's final test in src/train.py (seeds
9000-9199), so every number is measured on the same roads, weather, traffic,
hazards and channel. The draws in a step do not depend on the action
(src/env.py), so the world really is identical for every policy.

    .venv\\Scripts\\python.exe -m src.evaluate

Needs the trained models in runs/ (python -m src.train). Writes
runs/comparison.csv, runs/fig_report_curve.png and runs/fig_report_tradeoff.png.
"""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

from . import config
from .env import COMM, RADAR
from .figures import GRID, INK, INK_2, SERIES, SURFACE, curves
from .train import RUNS, SEEDS, evaluate, settled

METRICS = ("return", "miss_rate", "throughput", "routine_throughput",
           "urgent_delivered", "urgent_wait_s", "info_delivered", "info_wait_s",
           "radar_fraction")


class Baseline:
    """A fixed policy with the same predict() call as a trained SB3 model."""

    def __init__(self, choose):
        self.choose = choose
        self.rng = np.random.default_rng(0)

    def predict(self, obs, deterministic=True):
        return self.choose(len(obs), self.rng), None


def baselines() -> dict[str, Baseline]:
    """Fresh policies, so the coin flips are the same for every scenario."""
    return {
        "always-radar": Baseline(lambda n, rng: np.full(n, RADAR)),
        "always-comm": Baseline(lambda n, rng: np.full(n, COMM)),
        "random": Baseline(lambda n, rng: rng.integers(0, 2, n)),
    }


def compare() -> list[dict]:
    """One row per DQN seed, the DQN mean, and one row per baseline."""
    rows = []
    for name, scenario in config.SCENARIOS.items():
        dqn = []
        for seed in SEEDS:
            # The learning-rate schedule was saved as a function of src.train
            # run as __main__, so it cannot be read back here. Only training
            # uses it; the greedy test never does.
            model = DQN.load(RUNS / f"{name}_s{seed}_dqn.zip",
                             custom_objects={"learning_rate": 0.0,
                                             "lr_schedule": lambda _: 0.0})
            dqn.append({"scenario": name, "policy": "DQN", "seed": seed,
                        **evaluate(model, scenario)})
        rows += dqn
        rows.append({"scenario": name, "policy": "DQN", "seed": "mean",
                     **{k: float(np.nanmean([r[k] for r in dqn])) for k in METRICS}})
        for policy, model in baselines().items():
            rows.append({"scenario": name, "policy": policy, "seed": "-",
                         **evaluate(model, scenario)})
    return rows


def write(rows: list[dict]) -> None:
    keys = ["scenario", "policy", "seed", *METRICS]
    with open(RUNS / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def show(rows: list[dict]) -> None:
    def pct(x):
        return "   -  " if np.isnan(x) else f"{100 * x:5.1f}%"

    def secs(x):
        return "   -  " if np.isnan(x) else f"{x:5.2f} s"

    print(f"{'scenario':<9}{'policy':<14}{'return':>9}{'missed':>9}{'sent/step':>11}"
          f"{'urgent ok':>11}{'urgent wait':>13}{'radar':>8}")
    for r in rows:
        if r["policy"] == "DQN" and r["seed"] != "mean":
            continue
        label = "DQN (mean)" if r["policy"] == "DQN" else r["policy"]
        print(f"{r['scenario']:<9}{label:<14}{r['return']:>9.1f}{pct(r['miss_rate']):>9}"
              f"{r['throughput']:>11.2f}{pct(r['urgent_delivered']):>11}"
              f"{secs(r['urgent_wait_s']):>13}{pct(r['radar_fraction']):>8}")


# --- Report figures: sized for one column of a two-column page ---------------

WIDTH, HEIGHT = 3.5, 1.75
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(title, color=INK, loc="left", fontweight="bold")


def figure_curve() -> None:
    """DQN learning curve, both scenarios: mean of the seeds, band = their range."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes, config.SCENARIOS):
        x, runs = curves(name)
        mean = runs.mean(axis=0)
        _style(ax, name.capitalize())
        ax.fill_between(x, runs.min(axis=0), runs.max(axis=0), color=SERIES,
                        alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, mean, color=SERIES, linewidth=1.5, zorder=3)
        level, _ = settled(mean)
        ax.annotate(f"settles at {level:.0f}", xy=(x[-1], mean[-1]),
                    xytext=(-2, -14), textcoords="offset points", ha="right",
                    color=INK_2, fontsize=6.5)
        # The untrained policy scores far below everything after it; zoom on
        # the learning and say where the curve starts instead of drawing it.
        ax.text(0.97, 0.2, f"untrained start: −{abs(mean[0]):.0f}\n(below the axis)",
                transform=ax.transAxes, ha="right", color=INK_2, fontsize=6)
        ax.set_xlabel("training episode", color=INK_2)
    axes[0].set_ylim(-40, 370)
    axes[0].set_yticks([0, 100, 200, 300])
    axes[0].set_ylabel("test return / episode", color=INK_2)
    fig.tight_layout(pad=0.4)
    fig.savefig(RUNS / "fig_report_curve.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)


def figure_tradeoff(rows: list[dict]) -> None:
    """Hazards missed against packets sent: every policy, both scenarios.

    Top-left is better: few hazards missed, much data sent. The DQN is the one
    series in colour; the baselines are reference points in grey, each labelled.
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT + 0.15), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    nudge = {"always-radar": (4, 4), "always-comm": (-4, -11), "random": (0, 6)}
    for ax, name in zip(axes, config.SCENARIOS):
        _style(ax, name.capitalize())
        mine = [r for r in rows if r["scenario"] == name]
        seeds = [r for r in mine if r["policy"] == "DQN" and r["seed"] != "mean"]
        mean = next(r for r in mine if r["policy"] == "DQN" and r["seed"] == "mean")
        ax.scatter([100 * r["miss_rate"] for r in seeds],
                   [r["throughput"] for r in seeds], s=10, color=SERIES,
                   alpha=0.45, linewidths=0, zorder=3)
        ax.scatter(100 * mean["miss_rate"], mean["throughput"], s=32, color=SERIES,
                   edgecolors=SURFACE, linewidths=1.2, zorder=4)
        ax.annotate("DQN", xy=(100 * mean["miss_rate"], mean["throughput"]),
                    xytext=(6, -2), textcoords="offset points", color=INK,
                    fontweight="bold", va="center")
        for r in mine:
            if r["policy"] == "DQN":
                continue
            ax.scatter(100 * r["miss_rate"], r["throughput"], s=22, color=INK_2,
                       marker="s", edgecolors=SURFACE, linewidths=1.0, zorder=4)
            dx, dy = nudge[r["policy"]]
            ax.annotate(r["policy"], xy=(100 * r["miss_rate"], r["throughput"]),
                        xytext=(dx, dy), textcoords="offset points", color=INK_2,
                        ha="right" if dx < 0 else "left", fontsize=6.5)
        ax.set_xlim(-8, 108)
        ax.set_ylim(-0.08, 1.25)
        ax.set_xlabel("hazards missed (%)", color=INK_2)
    axes[0].set_ylabel("packets sent / step", color=INK_2)
    fig.tight_layout(pad=0.4)
    fig.savefig(RUNS / "fig_report_tradeoff.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)


# --- The report's numbers, written from the results, never typed by hand -----

REPORT = RUNS.parent / "report"
SHORT = {"highway": "hw", "city": "city"}


def settles_at(name: str, share: float = 0.9) -> int:
    """First training episode where the mean curve reaches `share` of its level."""
    x, runs = curves(name)
    mean = runs.mean(axis=0)
    level, _ = settled(mean)
    return int(x[np.argmax(mean >= share * level)])


def write_tex(rows: list[dict]) -> None:
    """report/numbers.tex (one macro per number) and report/table.tex."""
    lines = []
    for name, short in SHORT.items():
        m = next(r for r in rows if r["scenario"] == name
                 and r["policy"] == "DQN" and r["seed"] == "mean")
        lines += [
            rf"\newcommand{{\{short}miss}}{{{100 * m['miss_rate']:.1f}}}",
            rf"\newcommand{{\{short}catch}}{{{100 * (1 - m['miss_rate']):.1f}}}",
            rf"\newcommand{{\{short}tput}}{{{m['throughput']:.2f}}}",
            rf"\newcommand{{\{short}urg}}{{{100 * m['urgent_delivered']:.1f}}}",
            rf"\newcommand{{\{short}wait}}{{{m['urgent_wait_s']:.2f}}}",
            rf"\newcommand{{\{short}radar}}{{{100 * m['radar_fraction']:.0f}}}",
        ]
    episodes = max(settles_at(name) for name in config.SCENARIOS)
    lines.append(rf"\newcommand{{\settle}}{{{int(round(episodes, -1))}}}")
    (REPORT / "numbers.tex").write_text("\n".join(lines) + "\n")

    def cell(x, fmt, scale=1.0):
        return "--" if np.isnan(x) else fmt.format(scale * x)

    body = []
    for name in config.SCENARIOS:
        body.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{name.capitalize()}}}}}\\")
        for r in rows:
            if r["scenario"] != name or (r["policy"] == "DQN" and r["seed"] != "mean"):
                continue
            label = r"\textbf{DQN} (5 seeds)" if r["policy"] == "DQN" else r["policy"]
            body.append(" & ".join([
                label, cell(r["return"], "{:.0f}"),
                cell(r["miss_rate"], "{:.1f}", 100), cell(r["throughput"], "{:.2f}"),
                cell(r["urgent_delivered"], "{:.0f}", 100),
                cell(r["urgent_wait_s"], "{:.2f}")]) + r"\\")
    (REPORT / "table.tex").write_text("\n".join([
        r"\begin{table}[t]",
        r"\caption{Test results: 200 identical drives per policy}",
        r"\label{tab:res}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Policy & Return & Missed & Sent & Urgent & Urgent \\",
        r" & /episode & hazards \% & /step & sent \% & wait (s) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]) + "\n")


def main() -> int:
    rows = compare()
    write(rows)
    show(rows)
    figure_curve()
    figure_tradeoff(rows)
    write_tex(rows)
    print(f"\nwrote {RUNS / 'comparison.csv'}, fig_report_curve.png, "
          f"fig_report_tradeoff.png, report/numbers.tex, report/table.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
