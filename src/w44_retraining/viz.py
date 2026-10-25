"""Shared plotting style for the notebooks (same system as W37).

Categorical: the first 3 slots of the reference palette (validated all-pairs, colour-blind safe; aqua is below
3:1 contrast, so every chart that uses it carries direct labels or sits next to its table). Beyond 3 series:
one highlighted series against neutral grey, or small multiples, never a 4th hue.
Sequential: one hue (blue), light to dark. Diverging: blue <-> red, grey midpoint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)
TEXT, TEXT_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "w44_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "w44_div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948", "#a32a2a"])


def setup() -> None:
    """Matplotlib theme: recessive axes and grid, neutral ink for text."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "lines.linewidth": 2, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=list(SERIES)), "svg.fonttype": "none",
    })


def exporter(folder: str | Path = "../figures"):
    """Return export(fig, name): saves an SVG next to the block's README."""
    out = Path(folder)
    out.mkdir(exist_ok=True)

    def export(fig: plt.Figure, name: str) -> None:
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")

    return export
