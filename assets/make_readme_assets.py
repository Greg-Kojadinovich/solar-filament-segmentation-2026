"""Draw the README figures: the public-score bar chart and the two-pipeline architecture diagram.

Run from the repository root:  python assets/make_readme_assets.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
BACKGROUND, PANEL, TEXT, MUTED = "#0d1117", "#161b22", "#f0f6fc", "#8b949e"
ORANGE, BLUE, GREEN, PURPLE, RED = "#f78166", "#58a6ff", "#3fb950", "#bc8cff", "#ff7b72"


def wrap(text, width=19):
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\n".join(lines)


def score_chart():
    runs = [
        ("V1", "ResNet18 U-Net", 0.29, MUTED),
        ("V2", "U-Net, 1024 px + CLAHE", 0.32, MUTED),
        ("V3", "YOLO11s-seg + crop refiner", 0.35, BLUE),
        ("V5", "5-fold YOLO ensemble", 0.36, BLUE),
        ("V7", "YOLO only, no refiner", 0.26, RED),
        ("V11", "YOLO + solar filters + 8-view voting", 0.37, ORANGE),
        ("V13", "PyTorch Mask R-CNN + U-Net refiner", 0.37, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=200)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    positions = list(range(len(runs)))
    bars = ax.bar(positions, [r[2] for r in runs], color=[r[3] for r in runs], width=0.66, zorder=3)
    for bar, (_, _, value, color) in zip(bars, runs):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.2f}", ha="center", va="bottom",
                color=TEXT, fontsize=15, fontweight="bold", zorder=6,
                bbox=dict(facecolor=BACKGROUND, edgecolor="none", pad=1.5))
        if color in (ORANGE, GREEN):
            bar.set_edgecolor(TEXT)
            bar.set_linewidth(2)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{r[0]}\n{wrap(r[1])}" for r in runs], color=TEXT, fontsize=9.5)
    ax.set_ylim(0, 0.46)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.tick_params(axis="y", colors=MUTED, labelsize=10)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.grid(axis="y", color="#30363d", linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#30363d")
    ax.axhline(0.34, color=PURPLE, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(4, 0.334, "human vs human\nagreement: 0.34", color=PURPLE, fontsize=10, fontweight="bold",
            ha="center", va="top", linespacing=1.2)
    ax.set_title("Public leaderboard score (Panoptic Quality) by experiment", color=TEXT, fontsize=17,
                 fontweight="bold", pad=18, loc="left")
    ax.set_ylabel("Public Kaggle score", color=MUTED, fontsize=11)
    fig.text(0.07, 0.012, "Two very different architectures, YOLO (V11) and Mask R-CNN (V13), both landed at 0.37.",
             color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "score_chart.png", facecolor=BACKGROUND)
    plt.close(fig)


def box(ax, x, y, w, h, title, detail, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=PANEL, edgecolor=color, linewidth=2.2, zorder=3))
    ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", color=TEXT, fontsize=10.5,
            fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.36, detail, ha="center", va="center", color=MUTED, fontsize=8.2, zorder=4,
            linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=color,
                                 linewidth=2, zorder=2))


def architecture():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.955, "Two pipeline solutions, one goal: find every solar filament and outline it", color=TEXT,
            fontsize=20, fontweight="bold", va="center")
    ax.text(0.02, 0.915, "2048 x 2048 GONG H-alpha frame in, one mask per filament out, scored by Panoptic Quality",
            color=MUTED, fontsize=11.5, va="center")

    lanes = [
        ("YOLO PIPELINE  |  V11  |  public score 0.37", ORANGE, 0.53, [
            ("Solar-disk fit", "find the Sun's rim,\nflatten brightness,\ngray out the sky"),
            ("3 filter channels", "normalized frame,\nblurred copy,\nblack top-hat ridges"),
            ("YOLO11s-seg", "1280 px, 30 epochs,\nrotation + flip\naugmentation"),
            ("ResNet34 U-Net", "256 px crop per filament,\ndetector mask as\n4th input channel"),
            ("8-view voting", "4 rotations x flips,\nvote on instances\nand on pixels"),
            ("Clean-up", "largest piece only,\nminimum area,\ndisjoint masks"),
        ]),
        ("PYTORCH PIPELINE  |  V13  |  public score 0.37", GREEN, 0.09, [
            ("Resize", "whole frame to\n1024 px grayscale"),
            ("Mask R-CNN", "ResNet50-FPN V2,\nanchors 16-256 px,\n10-15 epochs, 2 GPUs"),
            ("4 flipped views", "up to 10 candidate\nfilaments per view"),
            ("Vote down", "group views at IoU 0.5,\ndrop weak groups,\nkeep the best 7-10"),
            ("ResNet18 U-Net", "256 px crop,\n4 flipped copies\naveraged"),
            ("Clean-up", "largest piece,\nmerge overlaps,\nvalid RLE masks"),
        ]),
    ]
    for label, color, y, steps in lanes:
        ax.add_patch(FancyBboxPatch((0.012, y - 0.03), 0.976, 0.355, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    facecolor="#0b0f14", edgecolor="#30363d", linewidth=1.2, zorder=1))
        ax.text(0.03, y + 0.30, label, color=color, fontsize=13.5, fontweight="bold", va="center", zorder=4)
        width, gap, x0 = 0.134, 0.0225, 0.03
        for index, (title, detail) in enumerate(steps):
            x = x0 + index * (width + gap)
            box(ax, x, y + 0.04, width, 0.20, title, detail, color)
            if index < len(steps) - 1:
                arrow(ax, x + width + 0.003, y + 0.14, x + width + gap - 0.003, y + 0.14, color)
    ax.add_patch(FancyBboxPatch((0.20, 0.435), 0.60, 0.05, boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=PANEL, edgecolor=PURPLE, linewidth=2, zorder=3))
    ax.text(0.50, 0.46, "Both end the same way: a validated COCO-RLE submission.csv, scored by PQ", color=TEXT,
            fontsize=11.5, fontweight="bold", ha="center", va="center", zorder=4)
    fig.savefig(OUT / "architecture_both_pipelines.png", facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


if __name__ == "__main__":
    score_chart()
    architecture()
    print("wrote", OUT / "score_chart.png", OUT / "architecture_both_pipelines.png")
