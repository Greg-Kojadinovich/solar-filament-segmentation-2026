"""Day-grouped outer CV with inner tuning for the standalone YOLO/refiner notebook."""

import hashlib
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
from scipy.optimize import linear_sum_assignment


def cv_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def make_day_folds(names, seed=2026, folds=5, inner_fraction=0.1):
    names = sorted(names)
    if len(set(names)) != len(names) or folds != 5 or not 0 < inner_fraction < 1:
        raise ValueError("Use unique physical filenames, five folds, and 0 < inner_fraction < 1.")
    by_day = defaultdict(list)
    for name in names:
        datetime.strptime(name[:8], "%Y%m%d")
        by_day[name[:8]].append(name)
    if len(by_day) < 10:
        raise ValueError("Five-fold CV needs at least ten observation days.")
    days = sorted(by_day)
    random.Random(seed).shuffle(days)
    # Largest physical-image groups first; seeded order breaks equal-size ties.
    days.sort(key=lambda day: -len(by_day[day]))
    outer_days, sizes = [[] for _ in range(folds)], [0] * folds
    for day in days:
        fold = min(range(folds), key=lambda index: (sizes[index], index))
        outer_days[fold].append(day)
        sizes[fold] += len(by_day[day])
    plans = []
    for fold in range(folds):
        remaining = sorted(set(days) - set(outer_days[fold]))
        random.Random(seed + fold + 1).shuffle(remaining)
        n_inner = max(1, round(len(remaining) * inner_fraction))
        if n_inner >= len(remaining):
            raise ValueError("Inner tuning leaves no training days.")
        groups = dict(
            train=set(remaining[n_inner:]), tune=set(remaining[:n_inner]),
            holdout=set(outer_days[fold]),
        )
        split = {key: sorted(name for day in selected for name in by_day[day])
                 for key, selected in groups.items()}
        if any(not selected for selected in split.values()):
            raise ValueError("Every CV partition must be nonempty.")
        if set.union(*groups.values()) != set(by_day):
            raise ValueError("CV partitions do not cover the labeled dataset.")
        if any(groups[a] & groups[b] for a, b in
               [("train", "tune"), ("train", "holdout"), ("tune", "holdout")]):
            raise ValueError("Observation-day leakage.")
        plans.append(dict(fold=fold, splits=split))
    outer = [name for plan in plans for name in plan["splits"]["holdout"]]
    if len(outer) != len(names) or set(outer) != set(names):
        raise ValueError("Each physical image must appear in exactly one outer fold.")
    return plans


def save_cv_plan(path, plan):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Existing CV plan differs. Choose a new CV_RUN_NAME.")
    else:
        path.write_text(json.dumps(plan, indent=2, allow_nan=False))


def fold_records(plan, records_by_name):
    return {phase: [record for name in names for record in records_by_name[name]]
            for phase, names in plan["splits"].items()}


def validate_fold_result(result, plan, plan_hash):
    if (result["plan_sha256"] != plan_hash or result["fold"] != plan["fold"]
            or result["status"] != "complete"):
        raise ValueError("Fold result provenance/status mismatch.")
    rows = result["outer_rows"]
    expected = plan["splits"]["holdout"]
    actual = [row["filename"] for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("Outer rows must cover exactly this fold's physical images.")
    if not result["checkpoint_hashes"] or not result["selected_parameters"]:
        raise ValueError("Missing selected model provenance.")
    for row in rows:
        if row["annotator_sets"] < 1:
            raise ValueError("Missing outer annotator evaluation.")
        for key in ["pq", "dice", "iou"]:
            if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError("Invalid outer metric.")
    mean = float(np.mean([row["pq"] for row in rows]))
    if not math.isclose(mean, result["outer_mean"]["pq"], abs_tol=1e-12):
        raise ValueError("Outer PQ summary disagrees with per-image rows.")


def summarize_folds(results, plans, plan_hash):
    by_fold = {}
    for result in results:
        fold = result["fold"]
        if fold not in range(5) or fold in by_fold:
            raise ValueError("Unknown or duplicate completed fold.")
        validate_fold_result(result, plans[fold], plan_hash)
        by_fold[fold] = result
    complete = len(by_fold) == 5
    summary = dict(
        plan_sha256=plan_hash, complete=complete, completed_folds=sorted(by_fold),
        missing_folds=sorted(set(range(5)) - set(by_fold)),
        status="complete" if complete else "incomplete: no five-fold aggregate reported",
    )
    if complete:
        fold_pq = [by_fold[fold]["outer_mean"]["pq"] for fold in range(5)]
        rows = [row for fold in range(5) for row in by_fold[fold]["outer_rows"]]
        summary.update(
            fold_mean_pq=float(np.mean(fold_pq)),
            fold_sample_sd_pq=float(np.std(fold_pq, ddof=1)),
            pooled_oof_mean_pq=float(np.mean([row["pq"] for row in rows])),
            physical_images=len(rows),
            note="Per-image metrics average annotators first. Fold SD is descriptive, not a confidence interval. "
                 "These are reused labeled data, not a new untouched final test or leaderboard score.",
        )
    return summary


def collect_fold_results(roots, plans, plan_hash):
    results = {}
    for root in roots:
        for path in sorted(Path(root).rglob("fold_result.json")):
            result = json.loads(path.read_text())
            fold = result["fold"]
            if fold not in range(5):
                raise ValueError(f"Unknown fold in {path}")
            validate_fold_result(result, plans[fold], plan_hash)
            if fold in results and results[fold] != result:
                raise ValueError(f"Conflicting results for fold {fold}; do not select by outer score.")
            results[fold] = result
    return [results[fold] for fold in sorted(results)]


def evaluate_outer(names, parameters, namespace, detector, refiner):
    rows = []
    for name in namespace["tqdm"](names, desc="Outer evaluation: frozen parameters"):
        proposals = namespace["v4_proposals"](
            namespace["train_lookup"][name], detector, refiner, parameters["confidence"],
        )
        predicted, fallbacks = namespace["v4_instances"](proposals, parameters)
        namespace["validate_disjoint"](predicted, namespace["FULL_SHAPE"])
        scores = [namespace["local_scores"](predicted, namespace["gt_rles"](record["id"]))
                  for record in namespace["records_by_name"][name]]
        if not scores:
            raise ValueError("No annotator set for outer image.")
        rows.append(dict(
            filename=name, annotator_sets=len(scores), instances=len(predicted),
            refiner_fallbacks=fallbacks,
            **{key: float(np.mean([score[key] for score in scores])) for key in scores[0]},
        ))
    if not rows:
        raise ValueError("No outer images.")
    means = {key: float(np.mean([row[key] for row in rows]))
             for key in rows[0] if key not in ["filename", "annotator_sets"]}
    return means, rows


def run_cv_fold(namespace, plan, shared_plan, root):
    """Run one isolated fold; outer labels never enter training or calibration."""
    plan_hash = cv_digest(shared_plan)
    directory = Path(root) / f"fold_{plan['fold']}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "fold_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        validate_fold_result(result, plan, plan_hash)
        print("Reusing completed outer evaluation:", result_path)
        return result
    namespace["OUTPUT_DIR"] = directory
    namespace["split_names"] = plan["splits"]
    namespace["split_records"] = fold_records(plan, namespace["records_by_name"])
    namespace["manifest"] = dict(
        seed=namespace["CFG"]["seed"], grouping="observation-day",
        annotation_sha256=namespace["annotation_sha256"], splits=plan["splits"],
    )
    namespace["write_json"](directory / "split_manifest.json", namespace["manifest"])
    for flag, files in [
        ("RUN_YOLO_TRAINING", ["yolo_best.pt", "yolo_provenance.json"]),
        ("RUN_REFINER_TRAINING", ["refiner_best.pt", "refiner_runtime.json"]),
    ]:
        existing = [(directory / name).exists() for name in files]
        if any(existing) and not all(existing):
            raise ValueError(f"Incomplete checkpoint evidence in {directory}; recover it explicitly.")
        namespace[flag] = not all(existing)
    namespace["RUN_V4_CALIBRATION"] = not (directory / "v4_calibration.json").exists()
    detector = refiner = None
    try:
        detector = namespace["v4_train_yolo"]()
        refiner = namespace["v4_train_refiner"]()
        calibration = namespace["v4_calibrate"](detector, refiner)
        means, rows = evaluate_outer(
            plan["splits"]["holdout"], calibration["selected"]["parameters"],
            namespace, detector, refiner,
        )
        result = dict(
            fold=plan["fold"], plan_sha256=plan_hash, status="complete",
            smoke_test=shared_plan["smoke_test"],
            checkpoint_hashes=calibration["hashes"],
            selected_parameters=calibration["selected"]["parameters"],
            inner_selected_scores=calibration["selected"]["scores"],
            outer_mean=means, outer_rows=rows,
        )
        validate_fold_result(result, plan, plan_hash)
        namespace["write_json"](directory / "fold_result.json", result)
        return result
    finally:
        del detector, refiner
        namespace["gc"].collect()
        if namespace["torch"].cuda.is_available():
            namespace["torch"].cuda.empty_cache()


def export_cv_submission(namespace, plan, shared_plan, root):
    """Export a predeclared fold; never select a model by outer score."""
    directory = Path(root) / f"fold_{plan['fold']}"
    result = json.loads((directory / "fold_result.json").read_text())
    validate_fold_result(result, plan, cv_digest(shared_plan))
    namespace.update(
        OUTPUT_DIR=directory, split_names=plan["splits"],
        split_records=fold_records(plan, namespace["records_by_name"]),
        RUN_YOLO_TRAINING=False, RUN_REFINER_TRAINING=False, RUN_V4_CALIBRATION=False,
    )
    detector = refiner = None
    try:
        detector = namespace["v4_train_yolo"]()
        refiner = namespace["v4_train_refiner"]()
        calibration = namespace["v4_calibrate"](detector, refiner)
        if calibration["hashes"] != result["checkpoint_hashes"]:
            raise ValueError("Submission checkpoints differ from the completed fold.")
        return namespace["v4_export"](detector, refiner, calibration)
    finally:
        del detector, refiner
        namespace["gc"].collect()
        if namespace["torch"].cuda.is_available():
            namespace["torch"].cuda.empty_cache()


ENSEMBLE_POLICY = dict(
    method="equal-fold-instance-majority-v1", match_iou=0.5,
    required_fold_votes=3, pixel_votes=3, min_area=20,
)


def fuse_fold_instances(fold_instances, shape):
    """Match at most one instance per fold per cluster, then vote across five folds."""
    if len(fold_instances) != 5:
        raise ValueError("Ensembling requires predictions from exactly five folds.")
    clusters = []
    for instances in fold_instances:
        for rle in instances:
            if tuple(rle["size"]) != tuple(shape) or not mask_utils.area(rle):
                raise ValueError("Invalid ensemble instance shape or empty mask.")
        if not instances:
            continue
        representatives = []
        for cluster in clusters:
            votes = sum((mask_utils.decode(rle).astype(np.uint8) for rle in cluster),
                        np.zeros(shape, dtype=np.uint8))
            representatives.append(mask_utils.encode(np.asfortranarray(
                (votes * 2 >= len(cluster)).astype(np.uint8))))
        matched = set()
        if representatives:
            overlaps = mask_utils.iou(representatives, instances, [0] * len(instances))
            eligible = overlaps >= ENSEMBLE_POLICY["match_iou"]
            # Maximize valid match count first, then IoU; invalid pairs cannot steal matches.
            weights = eligible * (min(overlaps.shape) + 1 + overlaps)
            rows, columns = linear_sum_assignment(-weights)
            for row, column in zip(rows, columns):
                if eligible[row, column]:
                    clusters[row].append(instances[column])
                    matched.add(column)
        clusters.extend([rle] for index, rle in enumerate(instances) if index not in matched)
    fused = []
    for cluster in clusters:
        if len(cluster) < ENSEMBLE_POLICY["required_fold_votes"]:
            continue
        votes = sum((mask_utils.decode(rle).astype(np.uint8) for rle in cluster),
                    np.zeros(shape, dtype=np.uint8))
        mask = (votes >= ENSEMBLE_POLICY["pixel_votes"]).astype(np.uint8)
        if int(mask.sum()) >= ENSEMBLE_POLICY["min_area"]:
            fused.append((len(cluster), int(mask.sum()),
                          mask_utils.encode(np.asfortranarray(mask))))
    fused.sort(key=lambda item: (-item[0], -item[1]))
    return [rle for _, _, rle in fused]


def export_cv_ensemble(namespace, plans, shared_plan, root):
    """Reload all five models sequentially and export one validated voted submission."""
    if shared_plan["smoke_test"] or namespace["SMOKE_TEST"]:
        raise ValueError("Smoke runs cannot produce competition exports.")
    if sorted(plan["fold"] for plan in plans) != list(range(5)):
        raise ValueError("All five unique fold plans are required.")
    plans = sorted(plans, key=lambda plan: plan["fold"])
    root = Path(root)
    results = []
    for plan in plans:
        result = json.loads((root / f"fold_{plan['fold']}" / "fold_result.json").read_text())
        validate_fold_result(result, plan, cv_digest(shared_plan))
        if result.get("smoke_test", False):
            raise ValueError("Cannot ensemble smoke checkpoints.")
        results.append(result)
    test_paths = namespace["test_paths"]
    if not test_paths or len({path.stem for path in test_paths}) != len(test_paths):
        raise ValueError("Test image IDs must be nonempty and unique.")
    prediction_paths = []
    for plan, result in zip(plans, results):
        directory = root / f"fold_{plan['fold']}"
        namespace.update(
            OUTPUT_DIR=directory, split_names=plan["splits"],
            split_records=fold_records(plan, namespace["records_by_name"]),
            RUN_YOLO_TRAINING=False, RUN_REFINER_TRAINING=False, RUN_V4_CALIBRATION=False,
        )
        detector = refiner = None
        try:
            detector = namespace["v4_train_yolo"]()
            refiner = namespace["v4_train_refiner"]()
            calibration = namespace["v4_calibrate"](detector, refiner)
            if (calibration["hashes"] != result["checkpoint_hashes"]
                    or calibration["selected"]["parameters"] != result["selected_parameters"]):
                raise ValueError("Ensemble checkpoints/calibration differ from completed fold.")
            parameters = calibration["selected"]["parameters"]
            predictions = {}
            for path in namespace["tqdm"](test_paths, desc=f"Ensemble fold {plan['fold']} test"):
                proposals = namespace["v4_proposals"](
                    path, detector, refiner, parameters["confidence"])
                instances, _ = namespace["v4_instances"](proposals, parameters)
                namespace["validate_disjoint"](instances, namespace["FULL_SHAPE"])
                predictions[path.stem] = [
                    dict(size=list(rle["size"]), counts=rle["counts"].decode("ascii"))
                    for rle in instances
                ]
            prediction_path = directory / "ensemble_predictions.json"
            namespace["write_json"](prediction_path, predictions)
            prediction_paths.append(prediction_path)
        finally:
            del detector, refiner
            namespace["gc"].collect()
            if namespace["torch"].cuda.is_available():
                namespace["torch"].cuda.empty_cache()
    predictions = [json.loads(path.read_text()) for path in prediction_paths]
    rows, image_audits = [], []
    for path in test_paths:
        per_fold = [
            [dict(size=rle["size"], counts=rle["counts"].encode("ascii"))
             for rle in fold[path.stem]] for fold in predictions
        ]
        instances = fuse_fold_instances(per_fold, namespace["FULL_SHAPE"])
        instances, ownership = namespace["disjoint_instances"](
            instances, namespace["FULL_SHAPE"], ENSEMBLE_POLICY["min_area"])
        namespace["validate_disjoint"](instances, namespace["FULL_SHAPE"])
        for rle in instances:
            if mask_utils.encode(np.asfortranarray(mask_utils.decode(rle)))["counts"] != rle["counts"]:
                raise ValueError("Ensemble RLE roundtrip failed.")
        image_audits.append(dict(
            filename=path.name, fold_instances=[len(items) for items in per_fold],
            instances=len(instances), ownership=ownership,
        ))
        rows.extend(dict(filament_id=f"{path.stem}_{index}",
                         segmentation_rle=rle["counts"].decode("ascii"))
                    for index, rle in enumerate(instances, 1))
    audit = namespace["validate_submission_rows"](
        rows, {path.stem for path in test_paths}, namespace["FULL_SHAPE"])
    destination = root / "submission.csv"
    temporary = root / "submission.ensemble.tmp.csv"
    fields = ["filament_id", "segmentation_rle"]
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with temporary.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fields or list(reader) != rows:
                raise ValueError("Ensemble CSV readback mismatch.")
        namespace["write_json"](root / "ensemble_audit.json", dict(
            policy=ENSEMBLE_POLICY, plan_sha256=cv_digest(shared_plan),
            folds=[dict(fold=r["fold"], hashes=r["checkpoint_hashes"],
                        parameters=r["selected_parameters"]) for r in results],
            validation=audit, images=image_audits,
            submission_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
        ))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    kaggle_root = Path("/kaggle/working")
    if kaggle_root.is_dir() and destination.resolve() != (kaggle_root / "submission.csv").resolve():
        namespace["shutil"].copy2(destination, kaggle_root / "submission.csv")
    namespace["OUTPUT_DIR"] = root
    if audit["zero_detection_images"]:
        print("WARNING: zero-detection images omitted, no invented masks:",
              audit["zero_detection_images"])
    return destination
