"""Generate high-level diagrams for the four scored pipelines."""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_paper_v2 import COLORS, arrow, box, save_figure, setup_style


PAPER = Path(__file__).resolve().parent
PIPELINES = [
    ("V1", [
        "Grayscale image\n512 px",
        "ResNet18 encoder\nU-Net decoder",
        "Semantic mask\nthreshold",
        "Connected components\ninstances",
    ]),
    ("Focused V2", [
        "Grayscale image\n1024 px + CLAHE",
        "V1 ResNet18 U-Net\nfine-tuned",
        "Semantic mask\nthreshold",
        "Connected components\ninstances",
    ]),
    ("Repaired V3", [
        "Grayscale image\n1024 px",
        "YOLO11s-seg\ninstance proposals",
        "256 px crops\nResNet18 U-Net",
        "Disjoint-mask repair\ninstances",
    ]),
    ("V5 ensemble", [
        "Grayscale image\n1024 px",
        "5 YOLO11s-seg models\ninstance proposals",
        "Per fold: YOLO-only or\nResNet34 U-Net refiner",
        "Match + 3-of-5 vote\ndisjoint masks",
    ]),
]


def main():
    setup_style()
    output = PAPER / "figures-v3"
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.1, 1.8))
    ax.set(xlim=(0, 14.8), ylim=(0, 4.4))
    ax.axis("off")
    for index, (name, stages) in enumerate(PIPELINES):
        y = 3.45 - index * 1.05
        ax.text(1.95, y + .38, name, ha="right", va="center",
                color=COLORS[index], fontsize=9, weight="bold")
        for stage, text in enumerate(stages):
            x = 2.15 + stage * 3.13
            box(ax, x, y, 2.8, .76, text, color="#F6F8FA",
                edge=COLORS[index], size=7.6)
            if stage < len(stages) - 1:
                arrow(ax, (x + 2.83, y + .38), (x + 3.10, y + .38))
    fig.tight_layout(pad=.15)
    save_figure(fig, output, "pipeline_architectures")
    print(f"Generated pipeline_architectures in {output}")


if __name__ == "__main__":
    main()
