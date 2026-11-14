"""Shared chart style so every figure in the project looks like one system."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

# Categorical colours (colour-blind checked order): identity, not decoration.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOR_LEGIT, COLOR_FRAUD = BLUE, ORANGE
COLOR_LOGREG, COLOR_LGBM = BLUE, ORANGE
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def set_style() -> None:
    """Apply the project chart style (thin marks, recessive grid, no chart junk)."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "text.color": INK, "lines.linewidth": 2,
        "legend.frameon": False,
    })


def save_fig(fig, name: str) -> None:
    """Save a figure under reports/figures/ as PNG."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png")


def pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    """Format an axis holding proportions (0-1) as percentages."""
    fmt = mpl.ticker.PercentFormatter(xmax=1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
