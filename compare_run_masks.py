"""Reproduce prediction-only V1/V2/V3 EDA; never train or submit."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pycocotools.mask as masks
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from submission_masks import SUBMISSION_FIELDS, submission_groups


SHAPE = (2048, 2048)
RUN_PATHS = {
    "V1": "cv_outputs/kaggle_baseline_v1/submission.csv",
    "V2": "V2Output/results/submission.csv",
    "V3": "cv_outputs/kaggle_v3/submission_disjoint.csv",
}


def describe_instances(instances, shape=SHAPE):
    coverage = np.zeros(shape, dtype=np.uint16)
    instance_rows = []
    for rle in instances:
        mask = masks.decode(rle)
        if mask.shape != shape or not mask.any():
            raise ValueError("Unexpected empty or wrong-size mask.")
        coverage += mask
        x, y, width, height = masks.toBbox(rle).astype(int)
        crop = mask[y:y + height, x:x + width]
        _, components = ndimage.label(crop, structure=np.ones((3, 3)))
        area = int(mask.sum())
        instance_rows.append(dict(
            area=area, bbox_fill=area / (width * height),
            bbox_elongation=max(width, height) / min(width, height),
            components=int(components),
        ))
    return coverage > 0, dict(
        count=len(instances), foreground_pixels=int(np.count_nonzero(coverage)),
        overlap_pixels=int(np.count_nonzero(coverage > 1)),
        disconnected_instances=sum(r["components"] > 1 for r in instance_rows),
    ), instance_rows


def agreement(first, second, union_first, union_second):
    intersection = np.count_nonzero(union_first & union_second)
    union = np.count_nonzero(union_first | union_second)
    matches = 0
    if first and second:
        matrix = masks.iou(first, second, [0] * len(second))
        a, b = linear_sum_assignment(-(matrix > 0.5).astype(float))
        matches = int(np.count_nonzero(matrix[a, b] > 0.5))
    denominator = len(first) + len(second)
    return dict(
        union_iou=float(intersection / union) if union else None,
        instance_agreement=float(2 * matches / denominator) if denominator else None,
        matches_iou_gt_half=matches,
        both_empty=not bool(denominator),
    )


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training_figures(root, output):
    v1 = json.loads((root / "cv_outputs/kaggle_baseline_v1/"
                    "unet_resnet18_gray_512_seed2026/training_history.json").read_text())
    v2 = json.loads((root / "V2Output/results/"
                    "v2_focused_1024_clahe_from_v1/training_history.json").read_text())
    log_path = root / "cv_outputs/kaggle_v3/solar-filament-cv-v3.log"
    events = json.loads(log_path.read_text())
    validation, refiner, losses = [], [], []
    for event in events:
        text = event["data"]
        match = re.fullmatch(r"\s*all\s+161\s+1199\s+([\d.\s]+)\s*", text)
        if match:
            values = [float(v) for v in match[1].split()]
            if len(values) != 8:
                raise ValueError("Unexpected YOLO metric columns.")
            validation.append(values)
        if text.startswith("REFINER:"):
            import ast
            refiner.append(ast.literal_eval(text.split("REFINER:", 1)[1].strip()))
        clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).split("\r")[-1]
        match = re.search(r"(\d+)/\d+\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)", clean)
        if match:
            losses.append(dict(epoch=int(match[1]), box_loss=float(match[2]),
                               mask_loss=float(match[3]), full_training_epoch="207/207" in clean))
    if not refiner or not validation or not losses:
        raise ValueError("Expected training evidence missing in V3 log.")
    # The final all-row revalidates best.pt; it is not another training epoch.
    if len(validation) != len(losses) + 1:
        raise ValueError("Cannot align YOLO validation summaries to completed epochs.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, history, key, title in [
        (axes[0, 0], v1, "epoch", "V1 semantic U-Net"),
        (axes[0, 1], v2, "additional_epoch", "V2 fine-tuning (epoch 0 excluded)"),
    ]:
        history = [r for r in history if r["train_loss"] is not None]
        ax.plot([r[key] for r in history], [r["train_loss"] for r in history], label="train")
        ax.plot([r[key] for r in history], [r["tune_loss"] for r in history], label="tune")
        ax.set(title=title, xlabel="Completed epoch", ylabel="Dice/BCE loss")
        ax.legend()
    ax = axes[1, 0]
    ax.plot(range(1, len(losses) + 1), [r[6] for r in validation[:-1]], label="mask mAP50")
    ax.plot(range(1, len(losses) + 1), [r[7] for r in validation[:-1]], label="mask mAP50-95")
    ax.set(title="V3 detector validation (not PQ)", xlabel="Epoch (48 budget-interrupted)")
    ax.legend()
    ax = axes[1, 1]
    for key in ["train", "tune"]:
        ax.plot([r["epoch"] for r in refiner], [r[key] for r in refiner], marker="o", label=key)
    ax.set(title="V3 refiner: budget-truncated", xlabel="Completed epoch", ylabel="Dice/BCE loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=150)
    plt.close(fig)
    return dict(yolo_logged_epochs=len(losses),
                yolo_full_training_epochs=sum(r["full_training_epoch"] for r in losses),
                yolo_epoch_losses=losses,
                yolo_validation_by_epoch=validation[:-1],
                yolo_best_checkpoint_revalidation=validation[-1], refiner_history=refiner,
                v1_best=min(v1, key=lambda r: r["tune_loss"]),
                v2_best=min(v2, key=lambda r: r["tune_loss"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("reports/v1-v3-assets"))
    args = parser.parse_args()
    root, output = args.root, args.output
    images = sorted((root / "data/MAGFiLO_1.0_Kaggle_2026/test").rglob("*.jpeg"))
    if len(images) != 180 or len({p.stem for p in images}) != 180:
        raise ValueError("Expected exactly 180 distinct physical test images.")
    groups, provenance = {}, {}
    for run, relative in RUN_PATHS.items():
        path = root / relative
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != SUBMISSION_FIELDS:
                raise ValueError(f"{run}: unexpected columns.")
            groups[run] = submission_groups(list(reader), {p.stem for p in images}, SHAPE)
        provenance[run] = dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    image_rows, instance_rows, pair_rows = [], [], []
    for path in images:
        unions, instances = {}, {}
        for run in RUN_PATHS:
            instances[run] = [rle for _, rle in groups[run].get(path.stem, [])]
            unions[run], row, details = describe_instances(instances[run])
            if row["overlap_pixels"]:
                raise ValueError(f"{run}/{path.stem}: overlaps in submitted predictions.")
            image_rows.append(dict(run=run, image=path.name, **row))
            instance_rows.extend(dict(run=run, image=path.name, **r) for r in details)
        for a, b in [("V1", "V2"), ("V2", "V3"), ("V1", "V3")]:
            pair_rows.append(dict(
                pair=f"{a}-{b}", image=path.name,
                **agreement(instances[a], instances[b], unions[a], unions[b]),
            ))
    write_table(output / "per_image.csv", image_rows)
    write_table(output / "per_instance.csv", instance_rows)
    write_table(output / "agreement.csv", pair_rows)
    summary = {}
    for run in RUN_PATHS:
        rows = [r for r in image_rows if r["run"] == run]
        details = [r for r in instance_rows if r["run"] == run]
        areas = np.array([r["area"] for r in details])
        summary[run] = dict(
            instances=len(details), empty_images=[r["image"] for r in rows if not r["count"]],
            mean_count=float(np.mean([r["count"] for r in rows])),
            foreground_pixels=sum(r["foreground_pixels"] for r in rows),
            mean_frame_coverage_percent=float(np.mean([r["foreground_pixels"] for r in rows])
                                              / np.prod(SHAPE) * 100),
            area_quantiles=dict(zip(["p10", "p50", "p90"], np.quantile(areas, [.1, .5, .9]).tolist())),
            small_instances_lt256=int(np.count_nonzero(areas < 256)),
            disconnected_instances=sum(r["components"] > 1 for r in details),
            median_bbox_fill=float(np.median([r["bbox_fill"] for r in details])),
        )
    pairs = {}
    for pair in ["V1-V2", "V2-V3", "V1-V3"]:
        rows = [r for r in pair_rows if r["pair"] == pair]
        pairs[pair] = dict(
            mean_union_iou_nonempty=float(np.mean([r["union_iou"] for r in rows
                                                  if r["union_iou"] is not None])),
            mean_instance_agreement_nonempty=float(np.mean([r["instance_agreement"] for r in rows
                                                           if r["instance_agreement"] is not None])),
            both_empty=sum(r["both_empty"] for r in rows),
        )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#2563eb", "#d97706", "#059669"]
    count_bins = np.arange(0, max(r["count"] for r in image_rows) + 3, 2)
    area_bins = np.geomspace(min(r["area"] for r in instance_rows),
                            max(r["area"] for r in instance_rows) * 1.01, 40)
    for run, color in zip(RUN_PATHS, colors):
        counts = [r["count"] for r in image_rows if r["run"] == run]
        areas = [r["area"] for r in instance_rows if r["run"] == run]
        axes[0].hist(counts, bins=count_bins, histtype="step", label=run, color=color)
        axes[1].hist(areas, bins=area_bins, histtype="step", label=run, color=color)
        axes[2].bar(run, summary[run]["mean_frame_coverage_percent"], color=color)
    axes[0].set(xlabel="Instances per image", ylabel="Images")
    axes[1].set(xlabel="Mask area (original pixels)", ylabel="Instances", xscale="log")
    axes[2].set(ylabel="Mean foreground / full frame (%)")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "distributions.png", dpi=150)
    plt.close(fig)
    disagreements = sorted(
        [r for r in pair_rows if r["pair"] == "V2-V3" and r["union_iou"] is not None],
        key=lambda r: (r["union_iou"], r["image"]),
    )
    counts = {(r["run"], r["image"]): r["count"] for r in image_rows}
    largest_gap = max(images, key=lambda p: (
        abs(counts["V3", p.name] - counts["V2", p.name]), p.name,
    ))
    examples = [
        ("Lowest V2/V3 foreground agreement", disagreements[0]["image"]),
        ("Largest V2/V3 instance-count gap", largest_gap.name),
        ("Median V2/V3 foreground agreement", disagreements[len(disagreements) // 2]["image"]),
        ("Shared zero-detection observation", summary["V3"]["empty_images"][0]),
    ]
    image_lookup = {p.name: p for p in images}
    for index, (reason, name) in enumerate(examples, 1):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        with Image.open(image_lookup[name]) as image:
            gray = np.asarray(image).copy()
        axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original H-alpha")
        rendered_labels = {}
        for ax, run in zip(axes[1:], RUN_PATHS):
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            entries = groups[run].get(Path(name).stem, [])
            labels = np.zeros(SHAPE, dtype=np.uint16)
            for identity, (_, rle) in enumerate(entries, 1):
                labels[masks.decode(rle).astype(bool)] = identity
            rendered_labels[run] = labels
            ax.imshow(np.ma.masked_equal(labels, 0), cmap="turbo", vmin=1,
                      vmax=max(2, len(entries)), interpolation="nearest", alpha=.65)
            ax.set_title(f"{run}: {len(entries)} instances")
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"{reason}: {name}\nColors identify predicted instances, not correctness.")
        fig.tight_layout()
        fig.savefig(output / f"example-{index}.png", dpi=150)
        plt.close(fig)
        if index <= 2:
            difference = (rendered_labels["V2"] > 0) ^ (rendered_labels["V3"] > 0)
            components, count = ndimage.label(difference, structure=np.ones((3, 3)))
            if not count:
                raise ValueError("Selected disagreement example has no differing pixels.")
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            ys, xs = np.where(components == sizes.argmax())
            x0, x1 = max(0, int(xs.min()) - 64), min(2048, int(xs.max()) + 65)
            y0, y1 = max(0, int(ys.min()) - 64), min(2048, int(ys.max()) + 65)
            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            for ax in axes:
                ax.imshow(gray[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
                ax.set_axis_off()
            axes[0].set_title("Original crop")
            for ax, run in zip(axes[1:], RUN_PATHS):
                labels = rendered_labels[run]
                ax.imshow(np.ma.masked_equal(labels[y0:y1, x0:x1], 0), cmap="turbo",
                          vmin=1, vmax=max(2, int(labels.max())), interpolation="nearest", alpha=.65)
                ax.set_title(run)
            fig.suptitle(f"{name}: largest connected V2/V3 foreground disagreement\n"
                         f"Original bounds x=[{x0},{x1}), y=[{y0},{y1}); no ground truth")
            fig.tight_layout()
            fig.savefig(output / f"detail-{index}.png", dpi=150)
            plt.close(fig)
    result = dict(
        provenance=provenance, runs=summary, pairs=pairs, examples=examples,
        training=training_figures(root, output),
        limitations="Test prediction morphology/agreement only, no test ground truth. "
                     "No thresholds or models selected using this analysis.",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: result[k] for k in ["runs", "pairs", "examples"]}, indent=2))
    print("EDA COMPLETE:", output)


if __name__ == "__main__":
    main()
