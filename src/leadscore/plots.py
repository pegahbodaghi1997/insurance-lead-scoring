"""Small matplotlib helpers so every chart in charts/ looks the same."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#8a8985"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = [BLUE, ORANGE, AQUA]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "font.size": 9,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def pct_axis(ax, axis="y"):
    from matplotlib.ticker import PercentFormatter
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(PercentFormatter(1.0, decimals=0))
