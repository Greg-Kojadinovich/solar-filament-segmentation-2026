"""Disjoint instance masks and an offline repair command for saved V3 CSVs."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils


MASK_POLICY = "first-mask-wins-v1"
SUBMISSION_FIELDS = ["filament_id", "segmentation_rle"]


def decode_instance(rle, shape):
    if tuple(rle["size"]) != tuple(shape):
        raise ValueError(f"Instance shape {rle['size']} does not match {shape}.")
    mask = mask_utils.decode(rle)
    if mask.shape != tuple(shape) or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError("An instance must be a nonempty binary mask.")
    return mask.astype(bool)


def disjoint_instances(instances, shape=(2048, 2048), min_area=1):
    """Keep earlier masks unchanged; later masks may claim only unowned pixels."""
    if min_area < 1:
        raise ValueError("min_area must be positive.")
    occupied = np.zeros(shape, dtype=bool)
    result, kept_indices, dropped_indices = [], [], []
    overlap_pixels = 0
    for index, rle in enumerate(instances):
        mask = decode_instance(rle, shape)
        overlap_pixels += int(np.count_nonzero(mask & occupied))
        mask &= ~occupied
        if np.count_nonzero(mask) < min_area:
            dropped_indices.append(index)
            continue
        occupied |= mask
        result.append(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
        kept_indices.append(index)
    return result, dict(
        policy=MASK_POLICY, overlap_pixels_removed=overlap_pixels,
        kept_indices=kept_indices, dropped_indices=dropped_indices,
    )


def validate_disjoint(instances, shape=(2048, 2048)):
    occupied = np.zeros(shape, dtype=bool)
    for index, rle in enumerate(instances):
        mask = decode_instance(rle, shape)
        if np.any(occupied & mask):
            raise ValueError(f"Instance {index} overlaps an earlier instance.")
        occupied |= mask


def submission_groups(rows, test_stems, shape):
    groups, seen = defaultdict(list), set()
    for row in rows:
        if set(row) != set(SUBMISSION_FIELDS):
            raise ValueError("CSV must have exactly the two competition columns.")
        identity = row["filament_id"]
        stem, separator, number = identity.rpartition("_")
        if (not separator or not number.isdigit() or int(number) < 1
                or stem not in test_stems or identity in seen):
            raise ValueError(f"Unknown, invalid, or duplicate filament_id: {identity}")
        seen.add(identity)
        rle = dict(size=list(shape), counts=row["segmentation_rle"].encode("ascii"))
        groups[stem].append((row, rle))
    if not groups:
        raise ValueError("Submission has no predicted instances.")
    return groups


def validate_submission_rows(rows, test_stems, shape=(2048, 2048)):
    groups = submission_groups(rows, test_stems, shape)
    for stem, entries in groups.items():
        try:
            validate_disjoint([rle for _, rle in entries], shape)
        except ValueError as error:
            raise ValueError(f"{stem}: {error}") from error
    return dict(
        rows=len(rows), images_with_predictions=len(groups),
        zero_detection_images=sorted(set(test_stems) - set(groups)),
        overlapping_pixels=0,
    )


def repair_submission(source, destination, test_stems, min_area=20, shape=(2048, 2048)):
    source, destination = Path(source), Path(destination)
    audit_path = destination.with_suffix(".audit.json")
    if source.resolve() == destination.resolve():
        raise ValueError("Repair must not overwrite the original submission.")
    if destination.exists() or audit_path.exists():
        raise FileExistsError("Choose a new repair output path; existing evidence is preserved.")
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUBMISSION_FIELDS:
            raise ValueError("Unexpected submission columns or column order.")
        original_rows = list(reader)
    groups = submission_groups(original_rows, test_stems, shape)
    repaired_by_id, image_audits = {}, []
    for stem, entries in groups.items():
        masks, audit = disjoint_instances([rle for _, rle in entries], shape, min_area)
        for index, rle in zip(audit["kept_indices"], masks):
            identity = entries[index][0]["filament_id"]
            repaired_by_id[identity] = dict(
                filament_id=identity, segmentation_rle=rle["counts"].decode("ascii"),
            )
        image_audits.append(dict(
            image=stem, overlap_pixels_removed=audit["overlap_pixels_removed"],
            input_instances=len(entries), output_instances=len(masks),
            dropped_ids=[entries[i][0]["filament_id"] for i in audit["dropped_indices"]],
        ))
    rows = [repaired_by_id[r["filament_id"]] for r in original_rows
            if r["filament_id"] in repaired_by_id]
    validation = validate_submission_rows(rows, test_stems, shape)
    temporary = destination.with_suffix(".tmp.csv")
    if temporary.exists():
        raise FileExistsError(f"Temporary file already exists: {temporary}")
    try:
        with temporary.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUBMISSION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with temporary.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != SUBMISSION_FIELDS or list(reader) != rows:
                raise ValueError("Repaired CSV readback mismatch.")
        report = dict(
            policy=MASK_POLICY, min_area=min_area, validation=validation,
            input_csv=str(source), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
            input_rows=len(original_rows), dropped_rows=len(original_rows) - len(rows),
            images_with_overlap=sum(a["overlap_pixels_removed"] > 0 for a in image_audits),
            overlap_pixels_removed=sum(a["overlap_pixels_removed"] for a in image_audits),
            images=image_audits,
            note="Offline repair only. No retraining, recalibration, or submission. "
                 "Previous tuning scores do not measure these repaired masks.",
        )
        with audit_path.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=20)
    args = parser.parse_args()
    test_stems = {p.stem for p in args.test_dir.rglob("*.jpeg")}
    if not test_stems:
        parser.error("--test-dir contains no test JPEGs.")
    report = repair_submission(args.source, args.destination, test_stems, args.min_area)
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))
    print("REPAIRED CSV:", args.destination, "No competition upload performed.")


if __name__ == "__main__":
    main()
