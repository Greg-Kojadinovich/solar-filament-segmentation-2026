"""CPU crop caching and batched refiner inference for the standalone V6 notebook."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from torch.utils.data import Dataset


def crop_bounds(mask, context=1.8, minimum=96):
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Cannot crop an empty instance.")
    height, width = mask.shape
    side = min(height, width, max(minimum, int(np.ceil(
        max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))))
    x = int(np.clip(round((xs.max() + xs.min() + 1 - side) / 2), 0, width - side))
    y = int(np.clip(round((ys.max() + ys.min() + 1 - side) / 2), 0, height - side))
    return x, y, x + side, y + side


def normalized_crop(image, bounds, size):
    array = np.asarray(image.crop(bounds).resize(
        (size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return np.repeat(((array - 0.5) / 0.5)[None], 3, axis=0).copy()


class CachedInstanceCrops(Dataset):
    """Cache compact pre-augmentation uint8 image/mask pairs, shared across folds."""

    def __init__(self, entries, cache_dir, size=256, context=1.8, minimum=96, augment=False):
        self.size, self.context, self.minimum = size, context, minimum
        self.augment = augment
        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.entries = []
        for path, rle in entries:
            path = Path(path)
            stat = path.stat()
            counts = rle["counts"]
            if not isinstance(counts, bytes):
                raise TypeError("Cached crop entries require compressed byte-count RLEs.")
            identity = json.dumps([
                str(path.resolve()), stat.st_size, stat.st_mtime_ns, list(rle["size"]),
                size, context, minimum, "uint8-bilinear-nearest-v1",
            ]).encode() + counts
            key = hashlib.sha256(identity).hexdigest()
            self.entries.append((path, rle, directory / f"{key}.npy"))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, rle, cached = self.entries[index]
        if cached.exists():
            pair = np.load(cached, allow_pickle=False)
            if pair.shape != (2, self.size, self.size) or pair.dtype != np.uint8:
                raise ValueError(f"Invalid cached crop: {cached}")
        else:
            mask = mask_utils.decode(rle)
            bounds = crop_bounds(mask, self.context, self.minimum)
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"Unexpected grayscale crop image: {path}")
                pixels = np.asarray(image.crop(bounds).resize(
                    (self.size, self.size), Image.Resampling.BILINEAR), dtype=np.uint8)
            x0, y0, x1, y1 = bounds
            target = np.asarray(Image.fromarray(mask[y0:y1, x0:x1]).resize(
                (self.size, self.size), Image.Resampling.NEAREST), dtype=np.uint8)
            pair = np.stack([pixels, target])
            temporary = cached.with_name(f"{cached.stem}.{os.getpid()}.tmp.npy")
            try:
                np.save(temporary, pair, allow_pickle=False)
                temporary.replace(cached)
            finally:
                if temporary.exists():
                    temporary.unlink()
        if not np.isin(pair[1], [0, 1]).all():
            raise ValueError(f"Nonbinary cached crop target: {cached}")
        x = np.repeat(((pair[0].astype(np.float32) / 255 - 0.5) / 0.5)[None], 3, axis=0)
        y = pair[1].astype(np.float32)[None]
        if self.augment:
            turns = int(torch.randint(4, ()).item())
            x, y = np.rot90(x, turns, axes=(1, 2)), np.rot90(y, turns, axes=(1, 2))
            if torch.rand(()).item() < 0.5:
                x, y = np.flip(x, axis=2), np.flip(y, axis=2)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def loader_options(workers, cuda):
    if type(workers) is not int or workers < 0:
        raise ValueError("DataLoader worker count must be a nonnegative integer.")
    options = dict(num_workers=workers, pin_memory=cuda)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2,
                       multiprocessing_context="spawn")
    return options


def parallel_refiner(model, device_ids):
    if not device_ids:
        return model
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("Refiner device IDs must be unique.")
    if len(device_ids) > 1:
        return torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    return model


def base_refiner(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


@torch.inference_mode()
def refine_proposals(path, masks, refiner, device, shape, batch_size=16,
                     crop_size=256, context=1.8, minimum=96):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Crop inference batch size must be a positive integer.")
    if masks.ndim != 3 or masks.shape[1:] != tuple(shape) or not np.isfinite(masks).all():
        raise ValueError("YOLO masks are not finite full-resolution masks.")
    proposals = []
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (shape[1], shape[0]):
            raise ValueError(f"Unexpected grayscale inference image: {path}")
        for start in range(0, len(masks), batch_size):
            batch, metadata = [], []
            for mask in masks[start:start + batch_size]:
                coarse = (mask >= .5).astype(np.uint8)
                if not coarse.any():
                    print("WARNING: YOLO returned an empty mask; omitted:", path.name)
                    continue
                bounds = crop_bounds(coarse, context, minimum)
                batch.append(normalized_crop(image, bounds, crop_size))
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
