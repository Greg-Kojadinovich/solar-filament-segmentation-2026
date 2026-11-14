"""Rebuild visual-draft figures from checked-in result and EDA evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import seaborn as sns


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent.parent
COMPETITION = ROOT / "Kaggle Competition"
COLORS = ["#2963A2", "#DB8A24", "#21856A", "#8057A6"]
INK = "#203040"


def load_results(path):
    results = json.loads(Path(path).read_text())
    runs = results["runs"]
    if [row["run"] for row in runs] != ["V1", "V2", "V3", "V5"]:
        raise ValueError("Expected the four principal experiments V1, V2, V3, V5.")
    for row in runs:
        score = row["public_pq"]
        if row["status"] == "pending":
            if score is not None or row["submission_id"] is not None:
                raise ValueError("Pending results must not contain a score or submission ID.")
        elif row["status"] == "complete":
            if (type(score) not in (int, float) or not np.isfinite(score)
                    or not 0 <= score <= 1 or type(row["submission_id"]) is not int):
                raise ValueError("Completed results need a finite PQ and integer submission ID.")
        else:
            raise ValueError("Unknown competition measurement status.")
    return results


def setup_style():
    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans",
                  rc={"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                      "xtick.labelsize": 8, "ytick.labelsize": 8,
                      "text.color": INK, "axes.labelcolor": INK,
                      "svg.fonttype": "none", "pdf.fonttype": 42,
                      "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig, output, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{extension}", bbox_inches="tight",
                    facecolor="white", dpi=220)
    plt.close(fig)


def score_chart(results, output):
    fig, ax = plt.subplots(figsize=(7.1, 2.45), layout="constrained")
    runs = results["runs"]
    for index, row in enumerate(runs):
        score = row["public_pq"]
        if score is None:
            ax.axvspan(index - .36, index + .36, facecolor=COLORS[index], alpha=.06)
            ax.text(index, .16, "PENDING\nNo scored CSV yet", ha="center", va="center",
                    color=COLORS[index], weight="bold", fontsize=9)
        else:
            ax.bar(index, score, width=.57, color=COLORS[index], zorder=3)
            ax.text(index, score + .012, f"{score:.2f}", ha="center",
                    weight="bold", fontsize=11)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        first, second = runs[a]["public_pq"], runs[b]["public_pq"]
        if first is not None and second is not None:
            ax.annotate(f"{second-first:+.2f}", xy=((a+b)/2, .395),
                        ha="center", fontsize=9, color=INK)
    ax.text(-.43, .445, "Measured progression; not a controlled single-factor ablation",
            fontsize=9, color=INK)
    ax.set(ylim=(0, max(.47, max(row["public_pq"] or 0 for row in runs) + .12)),
           xlim=(-.55, 3.55), ylabel="Public Kaggle PQ (higher is better)",
           xticks=range(4),
           xticklabels=[f"{row['run']}\n{row['chart_label']}" for row in runs])
    ax.grid(axis="x", visible=False)
    save_figure(fig, output, "score_progression")


def box(ax, x, y, width, height, text, color="#EAF0F6", edge="#7B8C9D", size=9):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015",
                               facecolor=color, edgecolor=edge, linewidth=.9))
    ax.text(x + width/2, y + height/2, text, ha="center", va="center",
            fontsize=size, color=INK, linespacing=1.35)


def arrow(ax, start, end, color="#596C7A"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                linewidth=1.1, color=color))


def workflow(output):
    fig, ax = plt.subplots(figsize=(7.1, 2.65))
    ax.set(xlim=(0, 12), ylim=(0, 5.2))
    ax.axis("off")
    box(ax, .1, 4.1, 3.3, .9, "2048 x 2048 H-alpha\nAnnotator-specific polygons", size=8)
    box(ax, 4, 4.1, 3.5, .9, "Group by observing day\nTrain / inner tune / evaluation")
    box(ax, 8.1, 4.1, 3.7, .9, "Unlabeled competition images\nInference only; never tune", size=8)
    arrow(ax, (3.45, 4.55), (3.95, 4.55))
    box(ax, .1, 2.45, 3.3, 1.05, "V1 / V2: semantic U-Net\nForeground probabilities",
        color="#E8F0FB", edge=COLORS[0], size=8)
    box(ax, .1, .95, 3.3, 1.05, "Resize; threshold\nConnected components\nbecome separate instances",
        color="#E8F0FB", edge=COLORS[0], size=8)
    arrow(ax, (4.0, 4.05), (1.75, 3.55))
    arrow(ax, (1.75, 2.4), (1.75, 2.05))
    box(ax, 4, 2.45, 3.5, 1.05, "V3 / V5: YOLO11s-seg\nSeparate instance proposals",
        color="#E5F3ED", edge=COLORS[2], size=8)
    box(ax, 4, .95, 3.5, 1.05, "256px crops -> U-Net\nResNet18 (V3)\nResNet34 (V5)",
        color="#E5F3ED", edge=COLORS[2], size=8)
    arrow(ax, (5.75, 4.05), (5.75, 3.55))
    arrow(ax, (5.75, 2.4), (5.75, 2.05))
    box(ax, 8.1, 2.45, 3.7, 1.05, "Inner-selected settings\nV5 only: fuse predictions\nfrom all five model pairs",
        color="#F1EAF8", edge=COLORS[3], size=8)
    box(ax, 8.1, .95, 3.7, 1.05, "Disjoint masks; RLE checks\nCSV -> Kaggle public PQ", size=8)
    arrow(ax, (3.45, 1.1), (3.7, 1.1))
    arrow(ax, (3.7, 1.1), (3.7, .7))
    arrow(ax, (3.7, .7), (7.8, .7))
    arrow(ax, (7.8, .7), (7.8, 2.98))
    arrow(ax, (7.55, 1.48), (7.8, 1.48))
    arrow(ax, (7.8, 1.48), (7.8, 2.98))
    arrow(ax, (7.8, 2.98), (8.05, 2.98))
    arrow(ax, (9.95, 4.05), (9.95, 3.55))
    arrow(ax, (9.95, 2.4), (9.95, 2.05))
    ax.text(6, .2, "Two alternative model paths; test masks remain unobserved.",
            ha="center", fontsize=8)
    fig.tight_layout(pad=.5)
    save_figure(fig, output, "workflow")


def folds_and_fusion(output):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3),
                             gridspec_kw={"width_ratios": [1.05, 1.25]})
    ax = axes[0]
    matrix = np.zeros((5, 5), dtype=int)
    np.fill_diagonal(matrix, 1)
    sns.heatmap(matrix, ax=ax, cmap=ListedColormap(["#A9C9E6", "#E6AD62"]),
                vmin=0, vmax=1, cbar=False, linewidths=2, linecolor="white",
                annot=np.where(matrix, "OUT", "IN"), fmt="", annot_kws={"fontsize": 8},
                xticklabels=["A", "B", "C", "D", "E"],
                yticklabels=[f"Fold {i}" for i in range(5)])
    ax.set(title="Day-grouped outer partitions", xlabel="Schematic day groups", ylabel="")
    ax.tick_params(axis="y", rotation=0)
    fig.legend(handles=[Patch(color="#A9C9E6", label="IN: training + inner tune"),
                        Patch(color="#E6AD62", label="OUT: frozen evaluation")],
               loc="lower center", bbox_to_anchor=(.5, -.01), frameon=False, fontsize=7,
               ncol=2)
    ax = axes[1]
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    ax.set_title("Same test image -> five predictions", fontsize=9)
    for i in range(5):
        x = i * 2 + .05
        box(ax, x, 4.6, 1.72, .9, f"Pair {i}",
            color="#F1EAF8", edge=COLORS[3], size=7)
        arrow(ax, (x + .86, 4.5), (5, 3.7))
    box(ax, 1, 2.45, 8, 1.15, "One-to-one instance matching\nIoU >= 0.5; one vote per fold",
        size=7)
    arrow(ax, (5, 2.35), (5, 1.9))
    box(ax, 1, .6, 8, 1.2, "3/5 instance and pixel votes\nDisjoint masks -> validated CSV",
        color="#E5F3ED", edge=COLORS[2], size=7)
    fig.subplots_adjust(left=.085, right=.985, bottom=.28, top=.86, wspace=.28)
    save_figure(fig, output, "folds_and_fusion")


def training_and_geometry(summary, v2_history, output):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05))
    history = pd.read_json(v2_history)
    history = history[history["train_loss"].notna()]
    sns.lineplot(data=history, x="additional_epoch", y="tune_loss",
                 ax=axes[0], color=COLORS[1], linewidth=1.7)
    best = summary["training"]["v2_best"]["additional_epoch"]
    axes[0].axvline(best, color=INK, linestyle="--", linewidth=.8)
    axes[0].set(title="V2: best epoch 13", xlabel="Additional epoch",
                ylabel="Tuning Dice/BCE loss")
    refiner = pd.DataFrame(summary["training"]["refiner_history"])
    sns.lineplot(data=refiner, x="epoch", y="tune", marker="o",
                 ax=axes[1], color=COLORS[2], linewidth=1.7)
    axes[1].set(title="V3: refiner tuning", xlabel="Completed epoch",
                ylabel="Tuning Dice/BCE loss", xticks=[1, 2, 3, 4])
    runs = list(summary["runs"])
    counts = [summary["runs"][name]["instances"] for name in runs]
    areas = [summary["runs"][name]["foreground_pixels"] / 1e6 for name in runs]
    axes[2].bar(runs, counts, color=COLORS[:3], alpha=.85)
    axes[2].set(title="Test geometry (not accuracy)", ylabel="Predicted instances", ylim=(0, 1700))
    for i, count in enumerate(counts):
        axes[2].text(i, count+30, str(count), ha="center", fontsize=7)
    axes[2].text(.5, .1, "Foreground pixels (millions)\n" +
                 " / ".join(f"{area:.2f}" for area in areas),
                 ha="center", transform=axes[2].transAxes, fontsize=7,
                 bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    fig.tight_layout(pad=.8, w_pad=1.2)
    save_figure(fig, output, "training_and_geometry")


def mask_examples(output):
    # Re-layout the existing, audited EDA panels without requiring restricted data downloads.
    from PIL import Image

    source = COMPETITION / "reports/v1-v3-assets/detail-1.png"
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("RGB"))
    if pixels.shape[:2] != (600, 2100):
        raise ValueError("EDA panel layout changed; inspect it before re-layout.")
    panels = [(97, 130, 419, 550), (592, 130, 914, 550),
              (1087, 130, 1409, 550), (1582, 130, 1904, 550)]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.05))
    for ax, bounds, title in zip(axes, panels, [
        "Original H-alpha", "V1: two instances", "V2: no detection", "V3: one instance",
    ]):
        x0, y0, x1, y1 = [round(value * scale) for value, scale in
                           zip(bounds, [2100/2000, 600/571, 2100/2000, 600/571])]
        ax.imshow(pixels[y0:y1, x0:x1])
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=.2, w_pad=.45)
    save_figure(fig, output, "mask_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER / "figures-v2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_style()
    result_path = PAPER / "paper_v2_results.json"
    summary_path = COMPETITION / "reports/v1-v3-assets/summary.json"
    history_path = COMPETITION / "V2Output/results/v2_focused_1024_clahe_from_v1/training_history.json"
    results = load_results(result_path)
    summary = json.loads(summary_path.read_text())
    score_chart(results, args.output)
    workflow(args.output)
    folds_and_fusion(args.output)
    training_and_geometry(summary, history_path, args.output)
    mask_examples(args.output)
    evidence = [result_path, summary_path, history_path,
                COMPETITION / "reports/v1-v3-assets/detail-1.png", Path(__file__)]
    manifest = dict(
        generator=str(Path(__file__).resolve().relative_to(ROOT)),
        versions=dict(matplotlib=matplotlib.__version__, seaborn=sns.__version__,
                      numpy=np.__version__, pandas=pd.__version__),
        sources={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in evidence},
        measurements=results,
        figures=["score_progression", "workflow", "folds_and_fusion",
                 "training_and_geometry", "mask_comparison"],
        note="SVG/PDF are vector exports; raster observation panels remain embedded images. "
             "SVGs are editable/importable artwork, not native Visio .vsdx documents. "
             "No test ground truth, confidence intervals, or V5 score are fabricated.",
    )
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {len(manifest['figures'])} figures in PDF/SVG/PNG:", args.output)


if __name__ == "__main__":
    main()
