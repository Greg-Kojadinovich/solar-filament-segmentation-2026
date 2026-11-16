"""V8: five-fold YOLO + crop-refiner execution, resumable inference, and audited export."""

import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import scipy.ndimage as ndi
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

from grouped_yolo_cv import cv_digest, fold_records, validate_fold_result
from solar_disk import normalize_file
from submission_masks import (
    SUBMISSION_FIELDS, disjoint_instances, validate_disjoint, validate_submission_rows,
)
from v6_crop_runtime import (
    CachedInstanceCrops, base_refiner, crop_bounds, loader_options, normalized_crop,
    parallel_refiner,
)
from v7_yolo_runtime import (
    atomic_json, decode_masks, encode_masks, file_hash, image_manifest,
    prepare_dataset, require_identity, score_predictions,
)

SAVE_EVERY = 8
MODES = ("yolo", "refined", "gated", "zoom")
YOLO_AUGMENT_DEFAULTS = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, fliplr=0.5, flipud=0.0,
                             mosaic=0.3, degrees=10.0, scale=0.3, close_mosaic=5)


def yolo_augmentation(settings):
    """Detector augmentation: V8-V10 defaults, overridable through settings["yolo_augment"].

    The solar disk is centered in the frame and everything outside it is neutral gray, so
    a rotation about the frame center is a rotation of the Sun about its own center.
    """
    custom = settings.get("yolo_augment", {})
    unknown = set(custom) - set(YOLO_AUGMENT_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown YOLO augmentation settings: {sorted(unknown)}")
    return {**YOLO_AUGMENT_DEFAULTS, **custom}


def rotate_batch(x, y, angles=None, scales=None, shifts=None):
    """Rotate (and optionally zoom and shift) each (image, mask) crop about its center.

    scales below 1 zoom in; shifts are fractions of the half-width. Images use reflection
    padding; masks use zero padding and are re-binarized.
    """
    count = x.shape[0]
    if angles is None:
        angles = torch.rand(count, device=x.device) * 2 * math.pi
    angles = angles.to(x.dtype)
    scale = torch.ones_like(angles) if scales is None else scales.to(x.dtype)
    shift = torch.zeros(count, 2, dtype=x.dtype, device=x.device) if shifts is None else shifts.to(x.dtype)
    cos, sin = angles.cos() * scale, angles.sin() * scale
    matrix = torch.stack([torch.stack([cos, -sin, shift[:, 0]], dim=1),
                          torch.stack([sin, cos, shift[:, 1]], dim=1)], dim=1)
    grid = F.affine_grid(matrix, list(x.shape), align_corners=False)
    rotated = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    mask = F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return rotated, (mask >= 0.5).to(y.dtype)


def jitter_batch(x, y):
    """Random rotation plus the zoom and shift an imperfect detector box would cause."""
    count = x.shape[0]
    scales = 0.75 + 0.4 * torch.rand(count, device=x.device)
    shifts = (torch.rand(count, 2, device=x.device) - 0.5) * 0.16
    return rotate_batch(x, y, scales=scales, shifts=shifts)


def simulate_coarse(y, generator=None):
    """Detector-like masks for a batch of ground-truth crop masks (N, 1, H, W).

    Half of the samples get a rough mask (blocky edges, shifted, grown or shrunk, sometimes
    only part of the filament) like YOLO's; the rest get a fine one like a first refinement
    pass. A fixed generator makes the validation inputs identical every epoch.
    """
    count, _, height, width = y.shape
    draws = torch.rand(count, 9, generator=generator).tolist()
    out = torch.empty_like(y)
    rows = torch.arange(height, device=y.device, dtype=y.dtype)[:, None].expand(height, width)
    columns = torch.arange(width, device=y.device, dtype=y.dtype)[None, :].expand(height, width)
    for index, u in enumerate(draws):
        truth = y[index:index + 1]
        fine = u[0] < 0.4
        cell = (2, 3, 4)[int(u[1] * 3)] if fine else (6, 8, 10, 14)[int(u[1] * 4)]
        pooled = F.avg_pool2d(truth, cell, ceil_mode=True)
        pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        mask = (pooled >= 0.35 + 0.3 * u[2]).to(y.dtype)
        reach = 2 if fine else 6
        mask = torch.roll(mask, shifts=(int(round((u[3] - 0.5) * 2 * reach)),
                                         int(round((u[4] - 0.5) * 2 * reach))), dims=(2, 3))
        radius = int(round((u[5] - 0.5) * (2 if fine else 7)))
        if radius > 0:
            mask = F.max_pool2d(mask, 2 * radius + 1, 1, radius)
        elif radius < 0:
            mask = 1 - F.max_pool2d(1 - mask, 2 * -radius + 1, 1, -radius)
        if not fine and u[6] < 0.15 and mask.sum() > 0:
            angle = u[7] * 2 * math.pi
            projection = math.cos(angle) * columns + math.sin(angle) * rows
            values = projection[mask[0, 0] > 0]
            keep = projection <= torch.quantile(values, 0.35 + 0.4 * u[8])
            mask = mask * keep[None, None].to(y.dtype)
        out[index] = mask[0] if mask.sum() >= 4 else truth[0]
    return out


def prepare_images(ns):
    """Write disk-normalized copies of every train/test frame and return their lookups.

    Geometry is unchanged (still 2048 x 2048), so labels, crop windows and the CSV
    coordinates are identical; only pixel values differ.
    """
    settings, root = ns["V8"], ns["SCRATCH_DIR"] / "_preprocessed"
    jobs = ([(path, root / "train" / name) for name, path in sorted(ns["train_lookup"].items())]
            + [(path, root / "test" / path.name) for path in ns["test_paths"]])
    started = time.monotonic()
    print(f"Disk normalization: starting {len(jobs)} frames", flush=True)
    margin, fits = settings["disk_margin"], []
    features = bool(settings.get("feature_channels", False))
    # Threads, not processes: numpy/scipy release the GIL, and notebooks cannot hang on spawn.
    with ThreadPoolExecutor(max_workers=settings["preprocess_workers"]) as pool:
        for done, fit in enumerate(pool.map(lambda job: normalize_file(job[0], job[1], margin, features), jobs), 1):
            fits.append(fit)
            if done % 100 == 0 or done == len(jobs):
                print(f"Disk normalization: {done}/{len(jobs)} frames "
                      f"({time.monotonic() - started:.0f}s)", flush=True)
    missing = [fit["name"] for fit in fits if not fit["disk_found"]]
    if len(missing) > 0.01 * len(fits):
        raise ValueError(f"Solar disk not found in {len(missing)} frames: {missing[:5]}")
    radii = np.array([fit["radius"] for fit in fits if fit["disk_found"]])
    report = dict(frames=len(fits), disk_not_found=missing, radius_min=float(radii.min()),
                  radius_median=float(np.median(radii)), radius_max=float(radii.max()),
                  margin=settings["disk_margin"], seconds=time.monotonic() - started)
    atomic_json(ns["CV_ROOT"] / "disk_fits.json", dict(report=report, fits=fits))
    print("Disk normalization:", report, flush=True)
    train_lookup = {name: destination for (_, destination), name in
                    zip(jobs[:len(ns["train_lookup"])], sorted(ns["train_lookup"]))}
    test_paths = [destination for _, destination in jobs[len(ns["train_lookup"]):]]
    return train_lookup, test_paths, report


def normalized_crop_rgb(image, bounds, size):
    array = np.asarray(image.crop(bounds).resize((size, size), Image.Resampling.BILINEAR),
                       dtype=np.float32) / 255
    return ((array - 0.5) / 0.5).transpose(2, 0, 1).copy()


def crop_tensor(image, bounds, size):
    """Refiner input for one window: three real channels, or one gray channel repeated."""
    return (normalized_crop_rgb(image, bounds, size) if image.mode == "RGB"
            else normalized_crop(image, bounds, size))


def images_are_rgb(ns):
    with Image.open(next(iter(ns["train_lookup"].values()))) as image:
        return image.mode == "RGB"


class CachedFeatureCrops(Dataset):
    """Instance crops of three-channel frames, cached as uint8 (3 image + 1 mask) arrays."""

    def __init__(self, entries, cache_dir, size=256, context=1.8, minimum=96, augment=False):
        self.size, self.context, self.minimum, self.augment = size, context, minimum, augment
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.entries = []
        for path, rle in entries:
            path = Path(path)
            stat = path.stat()
            counts = rle["counts"]
            if not isinstance(counts, bytes):
                raise TypeError("Cached crop entries require compressed byte-count RLEs.")
            identity = json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                                   list(rle["size"]), size, context, minimum,
                                   "rgb-uint8-bilinear-nearest-v1"]).encode() + counts
            self.entries.append((path, rle, directory / f"{hashlib.sha256(identity).hexdigest()}.npy"))

    def __len__(self):
        return len(self.entries)

    def _write(self, index, image):
        path, rle, cached = self.entries[index]
        mask = mask_utils.decode(rle)
        x0, y0, x1, y1 = crop_bounds(mask, self.context, self.minimum)
        pixels = np.asarray(image.crop((x0, y0, x1, y1)).resize(
            (self.size, self.size), Image.Resampling.BILINEAR), dtype=np.uint8)
        target = np.asarray(Image.fromarray(mask[y0:y1, x0:x1]).resize(
            (self.size, self.size), Image.Resampling.NEAREST), dtype=np.uint8)
        pair = np.concatenate([pixels.transpose(2, 0, 1), target[None]], axis=0)
        temporary = cached.with_name(f"{cached.stem}.{os.getpid()}.{index}.tmp.npy")
        try:
            np.save(temporary, pair, allow_pickle=False)
            temporary.replace(cached)
        finally:
            if temporary.exists():
                temporary.unlink()
        return pair

    def prewarm(self, threads=4):
        """Decode each frame once and write all of its missing crops."""
        groups = defaultdict(list)
        for index, (path, _, cached) in enumerate(self.entries):
            if not cached.exists():
                groups[path].append(index)

        def work(item):
            path, indices = item
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError(f"Expected a three-channel frame: {path}")
                image.load()
                for index in indices:
                    self._write(index, image)

        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(work, groups.items()))

    def __getitem__(self, index):
        path, rle, cached = self.entries[index]
        if cached.exists():
            pair = np.load(cached, allow_pickle=False)
        else:
            with Image.open(path) as image:
                pair = self._write(index, image.convert("RGB") if image.mode != "RGB" else image)
        if pair.shape != (4, self.size, self.size) or pair.dtype != np.uint8:
            raise ValueError(f"Invalid cached crop: {cached}")
        x = (pair[:3].astype(np.float32) / 255 - 0.5) / 0.5
        y = pair[3:4].astype(np.float32)
        if self.augment:
            turns = int(torch.randint(4, ()).item())
            x, y = np.rot90(x, turns, axes=(1, 2)), np.rot90(y, turns, axes=(1, 2))
            if torch.rand(()).item() < 0.5:
                x, y = np.flip(x, axis=2), np.flip(y, axis=2)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def window_mask(mask, bounds, size):
    """A full-frame binary mask cut to a window and resized to the refiner's input size."""
    x0, y0, x1, y1 = bounds
    return np.asarray(Image.fromarray(mask[y0:y1, x0:x1]).resize(
        (size, size), Image.Resampling.NEAREST), dtype=np.float32)


@torch.inference_mode()
def refine_image(image, masks, refiner, device, shape, batch_size=16, crop_size=256,
                 context=1.8, minimum=96, mask_channel=False):
    """Crop every detected mask from `image`, run the refiner, return per-crop probabilities.

    With mask_channel the detector's mask, cut to the same window, is the fourth input channel.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Crop inference batch size must be a positive integer.")
    if masks.ndim != 3 or masks.shape[1:] != tuple(shape):
        raise ValueError("Detector masks are not full-resolution masks.")
    if image.size != (shape[1], shape[0]):
        raise ValueError("Inference image does not match the mask resolution.")
    proposals = []
    for start in range(0, len(masks), batch_size):
        batch, metadata = [], []
        for mask in masks[start:start + batch_size]:
            coarse = (mask >= 0.5).astype(np.uint8)
            if not coarse.any():
                continue
            bounds = crop_bounds(coarse, context, minimum)
            crop = crop_tensor(image, bounds, crop_size)
            if mask_channel:
                crop = np.concatenate([crop, window_mask(coarse, bounds, crop_size)[None]], axis=0)
            batch.append(crop)
            metadata.append((mask_utils.encode(np.asfortranarray(coarse)), bounds))
        if not batch:
            continue
        tensor = torch.from_numpy(np.stack(batch)).to(device)
        probabilities = refiner(tensor).sigmoid()[:, 0].cpu().numpy()
        if probabilities.shape != (len(batch), crop_size, crop_size):
            raise ValueError("Refiner output does not match crop batch shape.")
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Nonfinite refinement probability.")
        proposals.extend(dict(coarse=rle, bounds=bounds, probability=probability)
                         for (rle, bounds), probability in zip(metadata, probabilities))
    return proposals


def segmentation_loss(logits, target):
    """Equal-weight BCE and soft Dice, as in the scored V1-V5 U-Net losses."""
    probabilities = logits.sigmoid()
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * dice


def refiner_channels(settings):
    """Three image channels, plus the detector mask as a fourth when enabled."""
    return 4 if settings.get("refiner_mask_channel", False) else 3


def build_refiner(settings, pretrained):
    return smp.Unet(
        encoder_name=settings["refiner_encoder"],
        encoder_weights="imagenet" if pretrained else None,
        in_channels=refiner_channels(settings), classes=1, activation=None,
    )


def load_refiner(ns, path):
    saved = torch.load(path, weights_only=True, map_location="cpu")
    model = build_refiner(ns["V8"], pretrained=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(ns["REFINER_DEVICE"]).eval()


def train_detector(ns, plan, shared_plan):
    """YOLO training as in V7, plus an explicit nominal batch size (nbs) and warmup length."""
    directory = ns["CV_ROOT"] / f"fold_{plan['fold']}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(plan_sha256=cv_digest(shared_plan), fold=plan["fold"])
    require_identity(directory / "fold_identity.json", identity)
    checkpoint, provenance = directory / "yolo_best.pt", directory / "yolo_provenance.json"
    if provenance.exists():
        saved = json.loads(provenance.read_text())
        if saved["identity"] != identity or saved["sha256"] != file_hash(checkpoint):
            raise ValueError("Saved detector checkpoint provenance mismatch.")
        print(f"Fold {plan['fold']}: reusing verified detector", flush=True)
        return YOLO(str(checkpoint))
    if checkpoint.exists():
        raise ValueError("Checkpoint publication was interrupted; preserve it and inspect provenance.")
    if not torch.cuda.is_available() and not ns["SMOKE_TEST"]:
        raise RuntimeError("Enable a GPU before full training.")
    settings = ns["V8"]
    data = prepare_dataset(ns, plan, directory)
    run = directory / "detector_runs" / "yolo"
    last, best = run / "weights" / "last.pt", run / "weights" / "best.pt"
    completed = False
    if last.exists():
        # Only read checkpoints generated by this identity-checked local run.
        saved = torch.load(last, map_location="cpu", weights_only=False)
        completed = saved["epoch"] < 0 or saved["epoch"] + 1 >= settings["epochs"]
        del saved
    started = time.monotonic()
    if not completed:
        if last.exists():
            print(f"Fold {plan['fold']}: resuming last completed YOLO epoch", flush=True)
            model = YOLO(str(last))
            # last.pt and best.pt are rewritten every epoch; per-epoch copies (save_period) fill the disk.
            model.train(resume=True, device=ns["YOLO_TRAIN_DEVICE"], workers=settings["workers"],
                        save_period=-1)
        else:
            print(f"Fold {plan['fold']}: starting YOLO (no saved epoch)", flush=True)
            model = YOLO(settings["weights"])
            model.train(
                data=str(data), epochs=settings["epochs"], imgsz=settings["image_size"],
                batch=settings["batch_size"], nbs=settings["nbs"],
                warmup_epochs=settings["warmup_epochs"], device=ns["YOLO_TRAIN_DEVICE"],
                workers=settings["workers"], patience=settings["patience"], time=None,
                optimizer="AdamW", lr0=settings["learning_rate"], momentum=0.9,
                weight_decay=0.0005, warmup_bias_lr=0.0, seed=ns["CFG"]["seed"],
                single_cls=True, mask_ratio=2, overlap_mask=False,
                **yolo_augmentation(settings),
                cache=False, save=True, save_period=-1, plots=False,
                project=str(run.parent), name=run.name, exist_ok=True,
            )
        del model
        gc.collect()
    if not best.is_file():
        raise FileNotFoundError(f"No best detector checkpoint: {best}")
    temporary = checkpoint.with_suffix(".tmp.pt")
    shutil.copyfile(best, temporary)
    temporary.replace(checkpoint)
    atomic_json(provenance, dict(
        identity=identity, sha256=file_hash(checkpoint), settings=settings,
        latest_invocation_seconds=time.monotonic() - started,
        note="Invocation time excludes earlier interrupted attempts; see YOLO results.csv.",
    ))
    return YOLO(str(checkpoint))


def refiner_entries(ns, records):
    return [(ns["train_lookup"][record["file_name"]], ns["ann_rle"](annotation))
            for record in records for annotation in ns["annotations_by_id"][record["id"]]]


def train_refiner(ns, plan, shared_plan):
    """Train the crop refiner on ground-truth instance crops; resume only at fold granularity."""
    settings, device = ns["V8"], ns["REFINER_DEVICE"]
    directory = ns["CV_ROOT"] / f"fold_{plan['fold']}"
    identity = dict(plan_sha256=cv_digest(shared_plan), fold=plan["fold"])
    checkpoint, provenance = directory / "refiner_best.pt", directory / "refiner_provenance.json"
    working = directory / "refiner_working_best.pt"
    if provenance.exists():
        saved = json.loads(provenance.read_text())
        if saved["identity"] != identity or saved["sha256"] != file_hash(checkpoint):
            raise ValueError("Saved refiner checkpoint provenance mismatch.")
        print(f"Fold {plan['fold']}: reusing verified refiner", flush=True)
        return load_refiner(ns, checkpoint)
    for stale in (working, checkpoint):
        if stale.exists():
            print(f"Fold {plan['fold']}: discarding unpublished refiner {stale.name}", flush=True)
            stale.unlink()

    torch.manual_seed(ns["CFG"]["seed"] + plan["fold"])
    records = fold_records(plan, ns["records_by_name"])
    crops = CachedFeatureCrops if images_are_rgb(ns) else CachedInstanceCrops
    datasets = {
        phase: crops(
            refiner_entries(ns, records[phase]), ns["CROP_CACHE_DIR"],
            size=settings["crop_size"], context=settings["crop_context"],
            minimum=settings["minimum_crop"], augment=phase == "train")
        for phase in ["train", "tune"]
    }
    if crops is CachedFeatureCrops:
        for dataset in datasets.values():
            dataset.prewarm(settings["preprocess_workers"])
    batch = settings["refiner_batch"]
    loaders = {
        phase: DataLoader(
            dataset, batch_size=batch, shuffle=phase == "train",
            # A one-sample tail (or a one-sample GPU chunk) breaks train-mode BatchNorm.
            drop_last=phase == "train" and 0 < len(dataset) % batch < 4,
            **loader_options(settings["refiner_workers"], device.type == "cuda"))
        for phase, dataset in datasets.items()
    }
    if any(len(loader) == 0 for loader in loaders.values()):
        raise ValueError("Empty crop dataset.")
    print(f"Fold {plan['fold']}: refiner crops "
          f"train={len(datasets['train'])} tune={len(datasets['tune'])}", flush=True)
    gpus = settings["refiner_gpus"] if device.type == "cuda" else []
    model = parallel_refiner(build_refiner(settings, pretrained=True).to(device), gpus)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["refiner_lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["refiner_epochs"])
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, significant_best, stale, history, status = float("inf"), float("inf"), 0, [], "max_epochs"
    started = time.monotonic()
    for epoch in range(settings["refiner_epochs"]):
        totals = {}
        for phase, loader in loaders.items():
            model.train(phase == "train")
            total, count = 0.0, 0
            # Validation draws restart from the same seed every epoch, so its loss is comparable.
            generator = torch.Generator().manual_seed(1234) if phase == "tune" else None
            with torch.set_grad_enabled(phase == "train"):
                for x, y in loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    if phase == "train" and settings.get("refiner_jitter", False):
                        x, y = jitter_batch(x, y)
                    elif phase == "train" and settings.get("refiner_rotate", False):
                        x, y = rotate_batch(x, y)
                    if settings.get("refiner_mask_channel", False):
                        coarse = simulate_coarse(y, generator)
                        x = torch.cat([x, coarse], dim=1)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        logits = model(x)
                    loss = segmentation_loss(logits.float(), y.float())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite crop loss.")
                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    total += loss.item() * len(x)
                    count += len(x)
            totals[phase] = total / count
        history.append(dict(epoch=epoch + 1, learning_rate=optimizer.param_groups[0]["lr"], **totals))
        print(f"Fold {plan['fold']} REFINER:", history[-1], flush=True)
        if totals["tune"] < best:
            best = totals["tune"]
            temporary = working.with_suffix(".tmp.pt")
            torch.save(dict(state_dict={k: v.detach().cpu()
                                        for k, v in base_refiner(model).state_dict().items()},
                            epoch=epoch + 1), temporary)
            temporary.replace(working)
        if totals["tune"] < significant_best - settings["refiner_min_delta"]:
            significant_best, stale = totals["tune"], 0
        else:
            stale += 1
        scheduler.step()
        if stale >= settings["refiner_patience"]:
            status = "early_stopped"
            break
    del optimizer, scheduler, scaler, model, loaders
    gc.collect()
    if not working.is_file():
        raise FileNotFoundError("Refiner produced no completed checkpoint.")
    temporary = checkpoint.with_suffix(".tmp.pt")
    shutil.copyfile(working, temporary)
    temporary.replace(checkpoint)
    atomic_json(directory / "refiner_history.json", history)
    atomic_json(provenance, dict(
        identity=identity, sha256=file_hash(checkpoint), settings=settings, status=status,
        completed_epochs=len(history), seconds=time.monotonic() - started,
    ))
    return load_refiner(ns, checkpoint)


def pass_mask(proposal, threshold, shape):
    """Full-frame mask of the first refiner pass (the detector's mask if it comes out empty)."""
    x0, y0, x1, y1 = proposal["bounds"]
    probability = np.asarray(Image.fromarray(proposal["probability"]).resize(
        (x1 - x0, y1 - y0), Image.Resampling.BILINEAR))
    keep = probability >= threshold
    if not keep.any():
        return mask_utils.decode(proposal["coarse"])
    full = np.zeros(shape, dtype=np.uint8)
    full[y0:y1, x0:x1] = keep
    return full


@torch.inference_mode()
def zoom_pass(image, proposals, refiner, device, shape, settings):
    """Zoom in on each refined filament and refine again with the first result as channel 4."""
    size, minimum = settings["crop_size"], settings["minimum_crop"]
    for threshold in settings["refiner_thresholds"]:
        firsts = [pass_mask(proposal, threshold, shape) for proposal in proposals]
        windows = [crop_bounds(first, settings["refiner_zoom_context"], minimum) for first in firsts]
        for start in range(0, len(proposals), settings["refiner_inference_batch"]):
            chunk = range(start, min(start + settings["refiner_inference_batch"], len(proposals)))
            batch = np.stack([np.concatenate([
                crop_tensor(image, windows[i], size),
                window_mask(firsts[i], windows[i], size)[None]], axis=0) for i in chunk])
            probabilities = refiner(torch.from_numpy(batch).to(device)).sigmoid()[:, 0].cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("Nonfinite zoom probability.")
            for i, probability in zip(chunk, probabilities):
                proposals[i].setdefault("zoom", {})[threshold] = dict(
                    bounds=windows[i], probability=probability)
    return proposals


def _yolo_source(image):
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return np.ascontiguousarray(array[..., ::-1])  # OpenCV order, as a file read would give


@torch.inference_mode()
def detect_proposals(ns, source, detector, refiner, confidence, refine):
    """YOLO instances for one image (a path or a PIL image); optionally add refiner probabilities."""
    settings, shape = ns["V8"], tuple(ns["FULL_SHAPE"])
    if isinstance(source, (str, Path)):
        with Image.open(source) as opened:
            opened.load()
            image = opened.copy()
        yolo_input = str(source)
    else:
        image, yolo_input = source, _yolo_source(source)
    result = detector.predict(
        source=yolo_input, imgsz=settings["image_size"], conf=confidence,
        device=ns["YOLO_INFERENCE_DEVICE"], agnostic_nms=True, retina_masks=True, verbose=False,
    )[0]
    if tuple(result.orig_shape) != shape:
        raise ValueError("Detector result dimensions differ from the image.")
    if result.masks is None or len(result.masks) == 0:
        return []
    masks = (result.masks.data > 0.5).to(torch.uint8).cpu().numpy()
    if masks.ndim != 3 or tuple(masks.shape[1:]) != shape:
        raise ValueError("Detector masks must retain original image dimensions.")
    if refine:
        proposals = refine_image(
            image, masks, refiner, ns["REFINER_DEVICE"], shape,
            batch_size=settings["refiner_inference_batch"], crop_size=settings["crop_size"],
            context=settings["crop_context"], minimum=settings["minimum_crop"],
            mask_channel=settings.get("refiner_mask_channel", False),
        )
        if settings.get("refiner_zoom", False):
            zoom_pass(image, proposals, refiner, ns["REFINER_DEVICE"], shape, settings)
        return proposals
    return [dict(coarse=mask_utils.encode(np.asfortranarray(mask)), bounds=None, probability=None)
            for mask in masks if mask.any()]


def largest_component(mask):
    """Keep the biggest 8-connected piece: every ground-truth filament is one polygon."""
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count <= 1:
        return mask
    return labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)


def build_instances(proposals, parameters, shape):
    """Turn proposals into disjoint instances under one yolo/refined/gated parameter set."""
    mode = parameters["mode"]
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    instances, fallbacks = [], 0
    for proposal in proposals:
        rle = proposal["coarse"]
        if mode == "zoom":
            x0, y0, x1, y1 = proposal["zoom"][parameters["refiner_threshold"]]["bounds"]
            probability = np.asarray(Image.fromarray(
                proposal["zoom"][parameters["refiner_threshold"]]["probability"]).resize(
                    (x1 - x0, y1 - y0), Image.Resampling.BILINEAR))
            keep = probability >= parameters["refiner_threshold"]
            if parameters.get("single_piece", False) and keep.any():
                keep = largest_component(keep)
            if keep.any():
                full = np.zeros(shape, dtype=np.uint8)
                full[y0:y1, x0:x1] = keep
                rle = mask_utils.encode(np.asfortranarray(full))
            else:  # fall back to the first pass
                fallbacks += 1
                rle = mask_utils.encode(np.asfortranarray(
                    pass_mask(proposal, parameters["refiner_threshold"], shape)))
        elif mode != "yolo":
            x0, y0, x1, y1 = proposal["bounds"]
            probability = np.asarray(Image.fromarray(proposal["probability"]).resize(
                (x1 - x0, y1 - y0), Image.Resampling.BILINEAR))
            keep = probability >= parameters["refiner_threshold"]
            if mode == "gated":
                window = mask_utils.decode(rle)[y0:y1, x0:x1].astype(bool)
                keep &= ndi.binary_dilation(window, iterations=parameters["gate_pixels"])
            if parameters.get("single_piece", False) and keep.any():
                keep = largest_component(keep)
            if keep.any():
                full = np.zeros(shape, dtype=np.uint8)
                full[y0:y1, x0:x1] = keep
                rle = mask_utils.encode(np.asfortranarray(full))
            else:
                fallbacks += 1
        if mask_utils.area(rle) >= parameters["min_area"]:
            instances.append(rle)
    instances, _ = disjoint_instances(instances, shape, parameters["min_area"])
    return instances, fallbacks


def candidate_parameters(settings, confidence):
    for area in settings["min_areas"]:
        base = dict(confidence=confidence, min_area=area)
        if settings.get("single_piece", False):
            base["single_piece"] = True
        yield dict(mode="yolo", refiner_threshold=None, gate_pixels=None, **base)
        for threshold in settings["refiner_thresholds"]:
            yield dict(mode="refined", refiner_threshold=threshold, gate_pixels=None, **base)
            for gate in settings["gate_pixels"]:
                yield dict(mode="gated", refiner_threshold=threshold, gate_pixels=gate, **base)
            if settings.get("refiner_zoom", False):
                yield dict(mode="zoom", refiner_threshold=threshold, gate_pixels=None, **base)


# Test-time voting: the 8 rotations/flips of a square frame are lossless, and V11 trains on all
# of them. Each view is processed by the full pipeline, its masks are mapped back, and the
# views vote on which instances exist and which pixels belong to them.
VIEWS = [(k, flip) for flip in (False, True) for k in range(4)]


def apply_view(array, k, flip):
    view = np.rot90(array, k, axes=(0, 1))
    if flip:
        view = view[:, ::-1]
    return np.ascontiguousarray(view)


def invert_mask(mask, k, flip):
    if flip:
        mask = mask[:, ::-1]
    return np.ascontiguousarray(np.rot90(mask, -k, axes=(0, 1)))


def tta_view_lists(ns, path, detector, refiner, parameters, count):
    """Instances of one image under `count` views, each mapped back to the original frame."""
    shape = tuple(ns["FULL_SHAPE"])
    with Image.open(path) as opened:
        opened.load()
        image = opened.copy()
    array = np.asarray(image)
    lists = []
    for k, flip in VIEWS[:count]:
        view = image if (k, flip) == (0, False) else Image.fromarray(apply_view(array, k, flip))
        proposals = detect_proposals(ns, view, detector, refiner, parameters["confidence"],
                                     parameters["mode"] != "yolo")
        instances, _ = build_instances(proposals, parameters, shape)
        if (k, flip) != (0, False):
            instances = [mask_utils.encode(np.asfortranarray(
                invert_mask(mask_utils.decode(rle), k, flip))) for rle in instances]
        lists.append(instances)
    return lists


def cluster_views(view_lists, match_iou=0.5):
    """Group instances across views: one-to-one IoU matching, at most one member per view."""
    clusters = []
    for view_index, instances in enumerate(view_lists):
        if not instances:
            continue
        matched = set()
        if clusters:
            overlaps = mask_utils.iou([c["rep"] for c in clusters], instances, [0] * len(instances))
            eligible = overlaps >= match_iou
            weights = eligible * (min(overlaps.shape) + 1 + overlaps)
            for row, column in zip(*linear_sum_assignment(-weights)):
                if eligible[row, column]:
                    clusters[row]["members"].append(instances[column])
                    matched.add(column)
        clusters.extend(dict(rep=rle, members=[rle])
                        for column, rle in enumerate(instances) if column not in matched)
    return clusters


def cluster_votes(clusters, shape, minimum_support):
    """Per-pixel vote counts (how many views include each pixel) for supported clusters."""
    votes = {}
    for index, cluster in enumerate(clusters):
        if len(cluster["members"]) >= minimum_support:
            total = np.zeros(shape, dtype=np.uint8)
            for rle in cluster["members"]:
                total += mask_utils.decode(rle)
            votes[index] = total
    return votes


def fuse_clusters(clusters, votes, instance_votes, pixel_votes, min_area, shape, single_piece=False):
    """Keep instances seen in >= instance_votes views; keep pixels in >= pixel_votes views."""
    kept = []
    for index, total in votes.items():
        support = len(clusters[index]["members"])
        if support < instance_votes:
            continue
        mask = total >= pixel_votes
        if single_piece and mask.any():
            mask = largest_component(mask)
        area = int(mask.sum())
        if area >= min_area:
            kept.append((support, area, mask))
    kept.sort(key=lambda item: (-item[0], -item[1]))
    rles = [mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) for _, _, mask in kept]
    instances, _ = disjoint_instances(rles, shape, min_area)
    return instances


def tta_grid(settings):
    return [(v, p) for v in settings["tta_instance_votes"] for p in settings["tta_pixel_votes"]]


def tta_parameters(parameters, choice):
    return dict(parameters, tta=dict(views=choice["views"], instance_votes=choice["instance_votes"],
                                     pixel_votes=choice["pixel_votes"]))


def select_tta(ns, detector, refiner, plan, identity, calibration):
    """Choose voting thresholds on the tuning photos; vote only if it beats the single view."""
    directory = ns["CV_ROOT"] / f"fold_{plan['fold']}"
    path = directory / "tta.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("Voting selection provenance mismatch.")
        return saved
    settings, shape = ns["V8"], tuple(ns["FULL_SHAPE"])
    count = settings["tta_views"]
    names = calibration["tuning_names"]
    parameters = calibration["selected"]["parameters"]
    cache_path = directory / "tta_tuning_views.json"
    expected = dict(identity=identity, parameters=parameters, views=count, names=names)
    cache = dict(identity=expected, views={})
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache["identity"] != expected:
            raise ValueError(f"Stale voting cache: {cache_path}")
    started = time.monotonic()
    for index, name in enumerate(names, 1):
        stem = Path(name).stem
        if stem in cache["views"]:
            continue
        lists = tta_view_lists(ns, ns["train_lookup"][name], detector, refiner, parameters, count)
        cache["views"][stem] = [encode_masks(instances) for instances in lists]
        if index % SAVE_EVERY == 0 or index == len(names):
            atomic_json(cache_path, cache)
            print(f"Voting views: {len(cache['views'])}/{len(names)} tuning photos "
                  f"({time.monotonic() - started:.0f}s)", flush=True)
    grid = tta_grid(settings)
    predictions = {combo: {} for combo in grid}
    lowest = min(v for v, _ in grid)
    for name in names:
        stem = Path(name).stem
        lists = [decode_masks(items) for items in cache["views"][stem]]
        clusters = cluster_views(lists)
        votes = cluster_votes(clusters, shape, lowest)
        for combo in grid:
            instances = fuse_clusters(clusters, votes, combo[0], combo[1], parameters["min_area"], shape,
                                      parameters.get("single_piece", False))
            predictions[combo][stem] = encode_masks(instances)
    trials = []
    for combo in grid:
        means, _ = score_predictions(ns, names, predictions[combo])
        trials.append(dict(instance_votes=combo[0], pixel_votes=combo[1], scores=means))
    best = max(trials, key=lambda trial: trial["scores"]["pq"])
    single = calibration["selected"]["scores"]["pq"]
    choice = dict(identity=identity, views=count, instance_votes=best["instance_votes"],
                  pixel_votes=best["pixel_votes"], enabled=best["scores"]["pq"] > single,
                  tuning_pq_voting=best["scores"]["pq"], tuning_pq_single=single, trials=trials)
    atomic_json(path, choice)
    print("VOTING (tuning only): best", (best["instance_votes"], best["pixel_votes"]),
          "PQ %.4f vs single view %.4f -> %s" % (best["scores"]["pq"], single,
                                                  "using voting" if choice["enabled"] else "single view"),
          flush=True)
    return choice


def tta_predictor(ns, detector, refiner, parameters, choice, counters):
    """Per-image voting predictor; a failure on one image falls back to its single view."""
    shape = tuple(ns["FULL_SHAPE"])

    def predict(path):
        try:
            lists = tta_view_lists(ns, path, detector, refiner, parameters, choice["views"])
            clusters = cluster_views(lists)
            votes = cluster_votes(clusters, shape, choice["instance_votes"])
            return fuse_clusters(clusters, votes, choice["instance_votes"], choice["pixel_votes"],
                                 parameters["min_area"], shape, parameters.get("single_piece", False))
        except Exception as error:  # keep the run alive; report it
            counters["fallbacks"] += 1
            print(f"WARNING: voting failed for {Path(path).name} ({error}); using the single view",
                  flush=True)
            proposals = detect_proposals(ns, path, detector, refiner, parameters["confidence"],
                                         parameters["mode"] != "yolo")
            return build_instances(proposals, parameters, shape)[0]

    return predict


def _cache_identity(identity, parameters, paths, shape):
    return dict(model=identity, parameters=parameters,
                images_sha256=cv_digest(image_manifest(paths)), shape=list(shape))


def _read_cache(cache_path, expected_identity, expected, shape):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return dict(identity=expected_identity, predictions={})
    cache = json.loads(cache_path.read_text())
    if cache["identity"] != expected_identity:
        raise ValueError(f"Stale prediction cache: {cache_path}")
    if not set(cache["predictions"]) <= expected:
        raise ValueError("Prediction cache contains unknown image IDs.")
    for items in cache["predictions"].values():
        validate_disjoint(decode_masks(items), shape)
    return cache


def predict_cached(ns, detector, refiner, paths, parameters, cache_path, identity, predictor=None):
    """Save every few images; an interrupted pass restarts only the missing ones."""
    paths, shape = list(paths), tuple(ns["FULL_SHAPE"])
    expected = {p.stem for p in paths}
    cache = _read_cache(cache_path, _cache_identity(identity, parameters, paths, shape), expected, shape)
    predictions = cache["predictions"]
    missing = [p for p in paths if p.stem not in predictions]
    print(f"{Path(cache_path).name}: {len(predictions)}/{len(paths)} saved", flush=True)
    for index, path in enumerate(missing, 1):
        if predictor is None:
            proposals = detect_proposals(ns, path, detector, refiner,
                                         parameters["confidence"], parameters["mode"] != "yolo")
            instances, _ = build_instances(proposals, parameters, shape)
        else:
            instances = predictor(path)
        validate_disjoint(instances, shape)
        predictions[path.stem] = encode_masks(instances)
        if index % SAVE_EVERY == 0 or index == len(missing):
            atomic_json(cache_path, cache)
            print(f"{Path(cache_path).name}: {len(predictions)}/{len(paths)} saved", flush=True)
    if set(predictions) != expected:
        raise ValueError("Incomplete inference coverage.")
    return predictions


def load_cached_predictions(ns, paths, parameters, cache_path, identity):
    paths, shape = list(paths), tuple(ns["FULL_SHAPE"])
    expected = {p.stem for p in paths}
    if not Path(cache_path).exists():
        raise FileNotFoundError(f"Missing test predictions: {cache_path}")
    cache = _read_cache(cache_path, _cache_identity(identity, parameters, paths, shape), expected, shape)
    if set(cache["predictions"]) != expected:
        raise ValueError(f"Incomplete test predictions: {cache_path}")
    return cache["predictions"]


def fold_model_identity(ns, plan, shared_plan):
    directory = ns["CV_ROOT"] / f"fold_{plan['fold']}"
    return dict(plan_sha256=cv_digest(shared_plan), fold=plan["fold"],
                checkpoint_sha256=file_hash(directory / "yolo_best.pt"),
                refiner_sha256=file_hash(directory / "refiner_best.pt"))


def checkpoint_hashes(identity):
    return {"yolo_best.pt": identity["checkpoint_sha256"],
            "refiner_best.pt": identity["refiner_sha256"]}


def calibrate(ns, detector, refiner, plan, identity):
    """Pick confidence/mode on the inner tuning days only; proposals are computed once per confidence."""
    directory = ns["CV_ROOT"] / f"fold_{plan['fold']}"
    path = directory / "calibration.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError("Calibration provenance mismatch.")
        return saved
    settings, shape = ns["V8"], tuple(ns["FULL_SHAPE"])
    names = sorted(plan["splits"]["tune"])
    random.Random(ns["CFG"]["seed"] + plan["fold"]).shuffle(names)
    names = names[:settings["tuning_images"]]
    if not names:
        raise ValueError("No tuning images.")
    trials = []
    for confidence in settings["confidences"]:
        started = time.monotonic()
        cache = {name: detect_proposals(ns, ns["train_lookup"][name], detector, refiner, confidence, True)
                 for name in names}
        for parameters in candidate_parameters(settings, confidence):
            predictions, fallbacks = {}, 0
            for name in names:
                instances, used = build_instances(cache[name], parameters, shape)
                predictions[Path(name).stem] = encode_masks(instances)
                fallbacks += used
            means, _ = score_predictions(ns, names, predictions)
            trials.append(dict(parameters=parameters, scores=means, refiner_fallbacks=fallbacks))
        del cache
        print(f"Fold {plan['fold']} tuning conf={confidence}: "
              f"{time.monotonic() - started:.0f}s", flush=True)
    best_by_mode = {
        mode: max((t for t in trials if t["parameters"]["mode"] == mode),
                  key=lambda t: t["scores"]["pq"])
        for mode in MODES if any(t["parameters"]["mode"] == mode for t in trials)
    }
    selected = max(trials, key=lambda t: t["scores"]["pq"])
    result = dict(identity=identity, selected=selected, best_by_mode=best_by_mode,
                  trials=trials, tuning_names=names)
    atomic_json(path, result)
    print("TUNING ONLY, not independent performance:",
          {mode: (t["parameters"], round(t["scores"]["pq"], 4)) for mode, t in best_by_mode.items()},
          "\nSELECTED:", selected["parameters"], flush=True)
    return result


def merge_holdout_into_training(plan):
    """Train on the holdout days too; only the tuning days stay out of training."""
    splits = plan["splits"]
    if set(splits["train"]) & set(splits["holdout"]):
        raise ValueError("Train and holdout days overlap.")
    return dict(plan, splits=dict(
        train=sorted(splits["train"] + splits["holdout"]), tune=list(splits["tune"]), holdout=[]))


def validate_result(result, plan, plan_hash):
    """validate_fold_result needs outer rows; without a holdout check the rest."""
    if plan["splits"]["holdout"]:
        return validate_fold_result(result, plan, plan_hash)
    if (result["plan_sha256"] != plan_hash or result["fold"] != plan["fold"]
            or result["status"] != "complete" or result["outer_rows"]
            or not result["checkpoint_hashes"] or not result["selected_parameters"]):
        raise ValueError("Fold result provenance/status mismatch.")


def run_fold(ns, plan, shared_plan):
    directory = ns["CV_ROOT"] / f"fold_{plan['fold']}"
    path = directory / "fold_result.json"
    plan_hash = cv_digest(shared_plan)
    if path.exists():
        result = json.loads(path.read_text())
        validate_result(result, plan, plan_hash)
        if result["checkpoint_hashes"] != checkpoint_hashes(fold_model_identity(ns, plan, shared_plan)):
            raise ValueError("Completed fold checkpoint changed.")
        print(f"Fold {plan['fold']}: reusing completed run", flush=True)
        return result
    started = time.monotonic()
    detector, refiner = train_detector(ns, plan, shared_plan), None
    try:
        refiner = train_refiner(ns, plan, shared_plan)
        identity = fold_model_identity(ns, plan, shared_plan)
        calibration = calibrate(ns, detector, refiner, plan, identity)
        parameters = calibration["selected"]["parameters"]
        names = plan["splits"]["holdout"]
        means, rows, yolo_means = None, [], None
        yolo_parameters = calibration["best_by_mode"]["yolo"]["parameters"]
        if names:
            outer_paths = [ns["train_lookup"][name] for name in names]
            predictions = predict_cached(ns, detector, refiner, outer_paths, parameters,
                                         directory / "outer_predictions.json", identity)
            means, rows = score_predictions(ns, names, predictions)
            # Matched diagnostic: same detector and split, crop refinement switched off.
            yolo_means = means
            if parameters["mode"] != "yolo":
                yolo_predictions = predict_cached(
                    ns, detector, refiner, outer_paths, yolo_parameters,
                    directory / "outer_predictions_yolo_only.json", identity)
                yolo_means, _ = score_predictions(ns, names, yolo_predictions)
        predict_cached(ns, detector, refiner, ns["test_paths"], parameters,
                       directory / "test_predictions.json", identity)
        choice = dict(enabled=False)
        counters = dict(fallbacks=0)
        if ns["V8"].get("tta_views", 1) > 1:
            choice = select_tta(ns, detector, refiner, plan, identity, calibration)
            if choice["enabled"]:
                predict_cached(ns, detector, refiner, ns["test_paths"], tta_parameters(parameters, choice),
                               directory / "test_predictions_tta.json", identity,
                               predictor=tta_predictor(ns, detector, refiner, parameters, choice, counters))
        result = dict(
            fold=plan["fold"], plan_sha256=plan_hash, status="complete",
            smoke_test=ns["SMOKE_TEST"], preprocessing=ns["PREPROCESSING"],
            checkpoint_hashes=checkpoint_hashes(identity),
            selected_parameters=parameters, inner_selected_scores=calibration["selected"]["scores"],
            outer_mean=means, outer_rows=rows,
            outer_yolo_only_parameters=yolo_parameters, outer_yolo_only_mean=yolo_means,
            tuning_pq_by_mode={mode: trial["scores"]["pq"]
                               for mode, trial in calibration["best_by_mode"].items()},
            training_photos=len(plan["splits"]["train"]),
            tta={k: v for k, v in choice.items() if k != "trials" and k != "identity"},
            tta_fallbacks=counters["fallbacks"],
            fold_seconds_latest_invocation=time.monotonic() - started,
        )
        validate_result(result, plan, plan_hash)
        atomic_json(path, result)
        return result
    finally:
        del detector, refiner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def crop_report(results):
    """Per-fold view of what the crop stage did; holdout PQs exist only with a holdout."""
    rows = []
    for result in sorted(results, key=lambda item: item["fold"]):
        selected = result["selected_parameters"]
        row = dict(
            fold=result["fold"], mode=selected["mode"], confidence=selected["confidence"],
            refiner_threshold=selected["refiner_threshold"], gate_pixels=selected["gate_pixels"],
            tuning_pq=round(result["inner_selected_scores"]["pq"], 4),
            tuning_pq_by_mode={k: round(v, 4) for k, v in result["tuning_pq_by_mode"].items()},
            training_photos=result["training_photos"],
            voting=result.get("tta", {}), voting_fallbacks=result.get("tta_fallbacks", 0),
        )
        if result["outer_mean"] is not None:
            row.update(
                outer_pq_selected=round(result["outer_mean"]["pq"], 4),
                outer_pq_yolo_only=round(result["outer_yolo_only_mean"]["pq"], 4),
                outer_instances_per_image=round(result["outer_mean"]["instances"], 2))
        rows.append(row)
    return rows


def export_single(ns, plan, shared_plan):
    """Write the one trained model's cached test predictions as a validated CSV."""
    if ns["SMOKE_TEST"] or shared_plan["smoke_test"]:
        raise ValueError("Smoke experiments cannot export competition submissions.")
    directory = ns["CV_ROOT"] / f"fold_{plan['fold']}"
    result = json.loads((directory / "fold_result.json").read_text())
    validate_result(result, plan, cv_digest(shared_plan))
    identity = fold_model_identity(ns, plan, shared_plan)
    if result["checkpoint_hashes"] != checkpoint_hashes(identity):
        raise ValueError("Export checkpoints differ from the evaluated models.")
    calibration = json.loads((directory / "calibration.json").read_text())
    if (calibration["identity"] != identity
            or calibration["selected"]["parameters"] != result["selected_parameters"]):
        raise ValueError("Export calibration differs from evaluated settings.")
    root, shape, paths = ns["CV_ROOT"], tuple(ns["FULL_SHAPE"]), ns["test_paths"]
    names = {path.stem for path in paths}
    if result.get("tta", {}).get("enabled"):
        predictions = load_cached_predictions(
            ns, paths, tta_parameters(result["selected_parameters"], result["tta"]),
            directory / "test_predictions_tta.json", identity)
    else:
        predictions = load_cached_predictions(
            ns, paths, result["selected_parameters"], directory / "test_predictions.json", identity)
    rows = []
    for path in paths:
        instances = decode_masks(predictions[path.stem])
        validate_disjoint(instances, shape)
        rows.extend(dict(filament_id=f"{path.stem}_{i}", segmentation_rle=rle["counts"].decode("ascii"))
                    for i, rle in enumerate(instances, 1))
    validate_submission_rows(rows, names, shape)
    version = ns.get("VERSION", "V8")
    destination = root / f"submission_{version}_{ns['PREPROCESSING']}.csv"
    temporary = destination.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUBMISSION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with temporary.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUBMISSION_FIELDS:
            raise ValueError("Written CSV columns differ.")
        readback = list(reader)
    if readback != rows:
        raise ValueError("Written CSV differs from generated predictions.")
    audit = validate_submission_rows(readback, names, shape)
    audit["masks_per_image_with_predictions"] = round(
        audit["rows"] / audit["images_with_predictions"], 2)
    digest = file_hash(temporary)
    temporary.replace(destination)
    top_level = root.parent / "submission.csv"
    shutil.copyfile(destination, top_level)
    atomic_json(root / "submission_audit.json", dict(
        run_name=root.name, preprocessing=ns["PREPROCESSING"],
        architecture="single YOLO11s-seg + ResNet34 U-Net 256 px crop refiner",
        plan_sha256=cv_digest(shared_plan), hashes=result["checkpoint_hashes"],
        parameters=result["selected_parameters"], voting=result.get("tta"),
        voting_fallbacks=result.get("tta_fallbacks", 0),
        test_manifest_sha256=cv_digest(image_manifest(paths)),
        validation=audit, submission_filename=destination.name, submission_sha256=digest,
        copied_to=str(top_level),
    ))
    if file_hash(destination) != digest or file_hash(top_level) != digest:
        raise ValueError("Final submission checksum mismatch.")
    print(f"FINAL {version} CSV VERIFIED:", destination, "\nCopied to:", top_level, "\nSHA256:", digest,
          "\nVALIDATION:", audit, flush=True)
    return destination


def verify_ready_to_submit(ns, path, min_masks=2.0, max_masks=10.5, force=False):
    """Re-check the exported CSV against its audit before anything is uploaded."""
    path = Path(path)
    audit = json.loads((ns["CV_ROOT"] / "submission_audit.json").read_text())
    validation = audit["validation"]
    if ns["SMOKE_TEST"] or not path.is_file() or path.name != audit["submission_filename"]:
        raise ValueError("Submission file is missing or is not the audited CSV.")
    if file_hash(path) != audit["submission_sha256"]:
        raise ValueError("Submission checksum differs from its audit.")
    if validation["overlapping_pixels"] != 0:
        raise ValueError("Submission masks overlap.")
    if validation["images_with_predictions"] < 0.9 * len(ns["test_paths"]):
        raise ValueError("Fewer than 90% of test images have predictions.")
    density = validation["masks_per_image_with_predictions"]
    if not force and not min_masks <= density <= max_masks:
        raise ValueError(
            f"{density} masks per image is outside {min_masks}-{max_masks} (labels average about 7.1; "
            "V7 scored 0.26 PQ at about 11.6). Set FORCE_SUBMIT = True to upload anyway.")
    return audit
