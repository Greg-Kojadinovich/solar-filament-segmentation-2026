"""Generate a review of saved training histories and test prediction agreement."""

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
import sklearn
from sklearn.metrics import f1_score, jaccard_score


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent.parent
COMPETITION = ROOT / "Kaggle Competition"
OUTPUT = PAPER / "results-review"
SHAPE = (2048, 2048)
RUNS = ("V1", "V2", "V3", "V5")
CSV_PATHS = {
    "V1": COMPETITION / "cv_outputs/kaggle_baseline_v1/submission.csv",
    "V2": COMPETITION / "V2Output/results/submission.csv",
    "V3": COMPETITION / "cv_outputs/kaggle_v3/submission_disjoint.csv",
    "V5": COMPETITION / "V5_SubmissionFile/submission.csv",
}
V1_HISTORY = COMPETITION / "cv_outputs/kaggle_baseline_v1/unet_resnet18_gray_512_seed2026/training_history.json"
V1_HOLDOUT = COMPETITION / "cv_outputs/kaggle_baseline_v1/unet_resnet18_gray_512_seed2026/holdout_summary.json"
V2_HISTORY = COMPETITION / "V2Output/results/v2_focused_1024_clahe_from_v1/training_history.json"
SUMMARY = COMPETITION / "reports/v1-v3-assets/summary.json"
RESULTS = PAPER / "paper_v2_results.json"
TEST_IMAGES = COMPETITION / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_unions(path, test_stems):
    groups = {}
    seen = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected submission columns in {path}")
        for row in reader:
            identity = row["filament_id"]
            stem, separator, number = identity.rpartition("_")
            if (not separator or not number.isdigit() or int(number) < 1
                    or stem not in test_stems or identity in seen):
                raise ValueError(f"Invalid or duplicate ID {identity!r} in {path}")
            seen.add(identity)
            rle = {"size": list(SHAPE), "counts": row["segmentation_rle"].encode("ascii")}
            if int(mask_utils.area(rle)) == 0:
                raise ValueError(f"Empty mask {identity!r} in {path}")
            groups.setdefault(stem, []).append(rle)
    if not groups:
        raise ValueError(f"No masks in {path}")
    return {stem: mask_utils.merge(masks) for stem, masks in groups.items()}


def foreground_agreement(first_area, second_area, intersection, total_pixels):
    if not 0 <= intersection <= min(first_area, second_area):
        raise ValueError("Invalid intersection area")
    union = first_area + second_area - intersection
    if union > total_pixels:
        raise ValueError("Invalid foreground union")
    if union == 0:
        return None
    # These four weighted samples represent every background/foreground pair.
    weights = [total_pixels - union, second_area - intersection,
               first_area - intersection, intersection]
    truth = [0, 0, 1, 1]
    predicted = [0, 1, 0, 1]
    return {
        "f1": float(f1_score(truth, predicted, sample_weight=weights, zero_division=0)),
        "jaccard": float(jaccard_score(truth, predicted, sample_weight=weights, zero_division=0)),
    }


def agreement(unions, test_stems):
    results = []
    for left, right in combinations(RUNS, 2):
        measurements = []
        excluded = 0
        for stem in sorted(test_stems):
            first = unions[left].get(stem)
            second = unions[right].get(stem)
            a = int(mask_utils.area(first)) if first is not None else 0
            b = int(mask_utils.area(second)) if second is not None else 0
            both = int(mask_utils.area(mask_utils.merge([first, second], intersect=True))) if first is not None and second is not None else 0
            metrics = foreground_agreement(a, b, both, SHAPE[0] * SHAPE[1])
            if metrics is None:
                excluded += 1
            else:
                measurements.append(metrics)
        if not measurements:
            raise ValueError(f"No nonempty comparisons for {left} and {right}")
        results.append({
            "left": left, "right": right, "images": len(measurements),
            "both_empty_excluded": excluded,
            "f1": float(np.mean([m["f1"] for m in measurements])),
            "jaccard": float(np.mean([m["jaccard"] for m in measurements])),
        })
    return results


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def loss_figure(v1, v2, training):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, rows, epoch, title in (
        (axes[0, 0], v1, "epoch", "V1: full-frame U-Net (512 px)"),
        (axes[0, 1], v2, "additional_epoch", "V2: fine-tuning (1024 px)"),
    ):
        for key, label in (("train_loss", "Training"), ("tune_loss", "Tuning")):
            available = [row for row in rows if row.get(key) is not None]
            ax.plot([row[epoch] for row in available], [row[key] for row in available],
                    marker=".", label=label)
        ax.set(title=title, xlabel="Epoch" if epoch == "epoch" else "Additional epoch (0 = inherited)",
               ylabel="0.5 BCE + 0.5 Dice loss")
        ax.legend()
    refiner = training["refiner_history"]
    ax = axes[1, 0]
    for key, label in (("train", "Training"), ("tune", "Tuning")):
        ax.plot([row["epoch"] for row in refiner], [row[key] for row in refiner],
                marker="o", label=label)
    ax.set(title="V3: 256 px crop refiner", xlabel="Completed epoch",
           ylabel="0.5 BCE + 0.5 Dice loss", xticks=[r["epoch"] for r in refiner])
    ax.legend()
    ax = axes[1, 1]
    detector = [row for row in training["yolo_epoch_losses"] if row["full_training_epoch"]]
    for key, label in (("box_loss", "Box"), ("mask_loss", "Mask")):
        ax.plot([row["epoch"] for row in detector], [row[key] for row in detector], label=label)
    ax.set(title="V3: detector training components", xlabel="Completed epoch",
           ylabel="Ultralytics reported loss")
    ax.legend()
    fig.suptitle("Saved loss histories; objectives and data differ across panels")
    save(fig, "loss_by_epoch")


def agreement_figure(pairs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), layout="constrained")
    for ax, metric, title in zip(axes, ("f1", "jaccard"), ("Foreground F1", "Foreground Jaccard (IoU)")):
        matrix = np.eye(len(RUNS))
        for pair in pairs:
            i, j = RUNS.index(pair["left"]), RUNS.index(pair["right"])
            matrix[i, j] = matrix[j, i] = pair[metric]
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=RUNS, yticklabels=RUNS,
               title=title, xlabel="Prediction", ylabel="Prediction")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i, j]:.3f}" if i != j else "self",
                        ha="center", va="center", color="white" if matrix[i, j] > .65 else "black")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("Mean per-image prediction agreement — NOT ground-truth accuracy")
    save(fig, "prediction_agreement")


def holdout_figure(holdout):
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    x = np.arange(len(RUNS))
    for offset, key, label, color in ((-.17, "dice", "F1 / Dice", "#2963A2"),
                                       (.17, "iou", "Jaccard / IoU", "#21856A")):
        value = holdout["scores"][key]
        ax.bar(x[0] + offset, value, width=.32, label=label, color=color)
        ax.text(x[0] + offset, value + .025, f"{value:.3f}", ha="center")
    for position in x[1:]:
        ax.text(position, .12, "Not available", rotation=90, ha="center", va="bottom",
                color="#6a6a6a")
    ax.set(xticks=x, xticklabels=RUNS, ylim=(0, 1),
           ylabel="Mean labeled foreground score",
           title="V1 local holdout: annotators averaged within each image")
    ax.legend(loc="upper right")
    save(fig, "labeled_holdout")


def main():
    test_stems = {path.stem for path in TEST_IMAGES.glob("*.jpeg")}
    if len(test_stems) != 180:
        raise ValueError(f"Expected 180 test images at {TEST_IMAGES}, found {len(test_stems)}")
    summary = json.loads(SUMMARY.read_text())
    scored = json.loads(RESULTS.read_text())
    hashes = {run: sha256(path) for run, path in CSV_PATHS.items()}
    for run in RUNS[:3]:
        if hashes[run] != summary["provenance"][run]["sha256"]:
            raise ValueError(f"{run} submission hash does not match saved EDA provenance")
    if hashes["V5"] != next(row["csv_sha256"] for row in scored["runs"] if row["run"] == "V5"):
        raise ValueError("V5 submission hash does not match scored submission provenance")
    unions = {run: load_unions(path, test_stems) for run, path in CSV_PATHS.items()}
    pairs = agreement(unions, test_stems)
    holdout = json.loads(V1_HOLDOUT.read_text())["unet"]
    if holdout["image_count"] != 105 or holdout["annotation_set_count"] != 166:
        raise ValueError("Unexpected V1 holdout population")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loss_figure(json.loads(V1_HISTORY.read_text()), json.loads(V2_HISTORY.read_text()), summary["training"])
    agreement_figure(pairs)
    holdout_figure(holdout)
    evidence = list(CSV_PATHS.values()) + [V1_HISTORY, V1_HOLDOUT, V2_HISTORY, SUMMARY, RESULTS, Path(__file__).resolve()]
    report = {
        "test_image_count": len(test_stems),
        "description": "Unweighted image means; both-empty pairs excluded. Prediction agreement, not accuracy.",
        "pairs": pairs,
        "v1_labeled_holdout": holdout,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in evidence},
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__, "scikit-learn": sklearn.__version__},
    }
    (OUTPUT / "measurements.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
