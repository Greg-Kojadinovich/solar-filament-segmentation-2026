"""V13: Mask R-CNN detector + ResNet18 crop U-Net refiner with candidate voting.

The detector and refiner follow the public "Mask R-CNN + U-Net Refiner" notebook by Nomannic19 and the
"LB 0.70 Topology-Safe Solar Filaments" notebook by Fususu (flip test-time views, refiner threshold, positive-overlap merge).
The candidate voting (predict up to 10 per view, vote down to a cap) is new here.
"""

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import Dataset

IMAGE_SIZE = 1024
FLIPS = [(), (1,), (0,), (0, 1)]  # identity, horizontal, vertical, both (axes of a 2-D frame)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_mask(mask):
    return mask_utils.encode(np.asfortranarray(np.asarray(mask).astype(np.uint8)))


def largest_component(mask):
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count <= 1:
        return mask
    return labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)


def merge_overlapping_rles(rles):
    """Union every group of instances that overlap by at least one pixel (removes duplicate detections)."""
    if len(rles) < 2:
        return list(rles)
    adjacency = mask_utils.iou(rles, rles, [0] * len(rles)) > 0
    groups, seen = [], set()
    for start in range(len(rles)):
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            group.append(index)
            stack.extend(np.flatnonzero(adjacency[index]).tolist())
        groups.append(group)
    return [mask_utils.merge([rles[i] for i in group]) if len(group) > 1 else rles[group[0]] for group in groups]


def square_bounds(mask, context=1.8, minimum=96):
    height, width = mask.shape
    ys, xs = np.where(mask)
    center_x, center_y = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return int(x0), int(y0), int(x1), int(y1)


# ----------------------------------------------------------------------------------------------- manifest

def build_manifest(annotation_path, seed, val_fraction=0.2):
    """One annotator set per physical observation (the one with the most filaments), stratified 80/20."""
    coco = COCO(str(annotation_path))
    best = {}
    for image_id, info in coco.imgs.items():
        annotations = coco.imgToAnns[image_id]
        key = (len(annotations), sum(float(a["area"]) for a in annotations), str(image_id))
        stem = Path(info["file_name"]).stem
        if stem not in best or key > best[stem][0]:
            best[stem] = (key, dict(id=str(image_id), file_name=info["file_name"], observation_id=stem,
                                    site=stem[-2:], n_filaments=len(annotations)))
    rows = [value[1] for value in best.values()]
    rows.sort(key=lambda row: row["file_name"])
    counts = np.array([row["n_filaments"] for row in rows])
    edges = np.unique(np.quantile(counts, [0.25, 0.5, 0.75]))
    strata = {}
    for row in rows:
        row["count_bin"] = int(np.searchsorted(edges, row["n_filaments"], side="right"))
        strata.setdefault((row["site"], row["count_bin"]), []).append(row)
    rng = random.Random(seed)
    for members in strata.values():
        rng.shuffle(members)
        take = int(round(len(members) * val_fraction)) if len(members) > 1 else 0
        for index, row in enumerate(members):
            row["split"] = "val" if index < take else "train"
    return rows


def all_annotator_ids(annotation_path):
    """observation_id -> every annotator record id (used to score against all human sets)."""
    coco = COCO(str(annotation_path))
    groups = {}
    for image_id, info in coco.imgs.items():
        groups.setdefault(Path(info["file_name"]).stem, []).append(str(image_id))
    return groups


# ----------------------------------------------------------------------------------------------- detector

def masks_to_boxes(masks):
    boxes = []
    for mask in masks:
        ys, xs = np.where(mask)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class FilamentDataset(Dataset):
    def __init__(self, records, annotation_path, image_dir, augment=False):
        self.records, self.augment = list(records), augment
        self.image_dir = Path(image_dir)
        self.coco = COCO(str(annotation_path))

    def __len__(self):
        return len(self.records)

    @staticmethod
    def photometric(image):
        valid = image > 0.01
        if np.random.rand() < 0.70:
            image[valid] = image[valid] ** np.random.uniform(0.80, 1.25)
        if np.random.rand() < 0.70:
            mean = image[valid].mean()
            contrast, gain = np.random.uniform(0.85, 1.15), np.random.uniform(0.90, 1.10)
            image[valid] = ((image[valid] - mean) * contrast + mean) * gain
        if np.random.rand() < 0.15:
            blurred = cv2.GaussianBlur(image, (0, 0), np.random.uniform(0.30, 0.80))
            image[valid] = blurred[valid]
        if np.random.rand() < 0.25:
            image[valid] += np.random.normal(0, np.random.uniform(0.003, 0.015), valid.sum())
        return np.clip(image, 0, 1)

    def __getitem__(self, index):
        row = self.records[index]
        image = cv2.imread(str(self.image_dir / row["file_name"]), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        masks = []
        for annotation in self.coco.imgToAnns[row["id"]]:
            mask = cv2.resize(self.coco.annToMask(annotation), (IMAGE_SIZE, IMAGE_SIZE),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            if mask.any():
                masks.append(mask)
        masks = np.stack(masks) if masks else np.zeros((0, IMAGE_SIZE, IMAGE_SIZE), bool)
        if self.augment:
            rotations = np.random.randint(4)
            image = np.rot90(image, rotations)
            masks = np.rot90(masks, rotations, axes=(1, 2))
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1)
                masks = np.flip(masks, axis=2)
            image = self.photometric(np.ascontiguousarray(image))
        image, masks = np.ascontiguousarray(image), np.ascontiguousarray(masks)
        target = dict(
            boxes=torch.as_tensor(masks_to_boxes(masks), dtype=torch.float32),
            labels=torch.ones(len(masks), dtype=torch.int64),
            masks=torch.as_tensor(masks, dtype=torch.uint8),
            image_id=torch.tensor([index]),
            area=torch.as_tensor(masks.sum((1, 2)), dtype=torch.float32),
            iscrowd=torch.zeros(len(masks), dtype=torch.int64))
        return torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1), target, row["observation_id"]


def collate_fn(batch):
    return tuple(zip(*batch))


def build_detector(pretrained=True):
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, hidden, 2)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2)
    model.transform.min_size, model.transform.max_size = (IMAGE_SIZE,), IMAGE_SIZE
    return model


# ----------------------------------------------------------------------------------------------- scoring

def instance_scores(predicted, truth):
    """Soft instance Dice (matched Dice / max(p, t), as in the public scorers) and PQ (IoU > 0.5)."""
    p, t = len(predicted), len(truth)
    if not p or not t:
        both_empty = float(p == 0 and t == 0)
        return dict(soft_dice=both_empty, pq=both_empty, tp=0, predictions=p)
    iou = mask_utils.iou(predicted, truth, [0] * t)
    dice = 2 * iou / (1 + iou + 1e-7)
    rows, columns = linear_sum_assignment(-dice)
    soft = float(dice[rows, columns].sum() / max(p, t))
    valid = iou > 0.5
    rewards = valid * (1 + iou / (min(p, t) + 1))
    rows, columns = linear_sum_assignment(-rewards)
    matched = [float(iou[r, c]) for r, c in zip(rows, columns) if valid[r, c]]
    tp = len(matched)
    denominator = tp + 0.5 * (p - tp) + 0.5 * (t - tp)
    return dict(soft_dice=soft, pq=sum(matched) / denominator if denominator else 1.0, tp=tp, predictions=p)


def mean_scores(predictions, references):
    """predictions: obs -> [rle]; references: obs -> [[rle] per annotator set]. Averages sets, then images."""
    rows = []
    for observation, sets in references.items():
        per_set = [instance_scores(predictions.get(observation, []), truth) for truth in sets]
        rows.append({key: float(np.mean([s[key] for s in per_set])) for key in per_set[0]})
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def annotation_rles(coco, record_id, size):
    rles = []
    for annotation in coco.imgToAnns[str(record_id)]:
        mask = coco.annToMask(annotation)
        if mask.shape != (size, size):
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        if mask.any():
            rles.append(encode_mask(mask))
    return rles


# ----------------------------------------------------------------------------------------------- detector loop

def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total, count = 0.0, 0
    for images, targets, _ in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            loss = sum(model(images, targets).values())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.detach())
        count += 1
    return total / max(count, 1)


@torch.inference_mode()
def evaluate_detector(model, loader, device, score_threshold=0.6, mask_threshold=0.5):
    model.eval()
    rows = []
    for images, targets, _ in loader:
        output = model([images[0].to(device)])[0]
        keep = output["scores"] >= score_threshold
        predicted = [encode_mask(m) for m in (output["masks"][keep, 0] >= mask_threshold).cpu().numpy()]
        predicted = merge_overlapping_rles([r for r in predicted if mask_utils.area(r) > 0])
        truth = [encode_mask(m) for m in targets[0]["masks"].numpy()]
        rows.append(instance_scores(predicted, truth))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


# ----------------------------------------------------------------------------------------------- refiner

class CropDataset(Dataset):
    def __init__(self, image_path, mask_path, augment=False):
        self.images = np.load(image_path, mmap_mode="r")
        self.masks = np.load(mask_path, mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(np.asarray(self.images[index]).copy()).float() / 255
        mask = torch.from_numpy(np.asarray(self.masks[index]).copy()).float().unsqueeze(0)
        if self.augment:
            rotations = random.randrange(4)
            image, mask = torch.rot90(image, rotations, (0, 1)), torch.rot90(mask, rotations, (1, 2))
            if random.random() < 0.5:
                image, mask = torch.flip(image, (1,)), torch.flip(mask, (2,))
            if random.random() < 0.5:
                image = image ** random.uniform(0.85, 1.20)
            if random.random() < 0.5:
                image = torch.clamp(image * random.uniform(0.90, 1.10), 0, 1)
        return image.unsqueeze(0).repeat(3, 1, 1), mask


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))


class CropUNet(nn.Module):
    """ResNet18 encoder U-Net on 256 x 256 instance crops."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool, self.layer1, self.layer2 = encoder.maxpool, encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], 1))
        x = self.dec3(torch.cat([self.up3(x), x2], 1))
        x = self.dec2(torch.cat([self.up2(x), x1], 1))
        x = self.dec1(torch.cat([self.up1(x), x0], 1))
        return self.head(self.up0(x))


def soft_dice_loss(logits, targets):
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def build_crop_cache(coco, rows, image_dir, directory, prefix, size=256):
    directory = Path(directory)
    total = sum(len(coco.imgToAnns[row["id"]]) for row in rows)
    image_path, mask_path = directory / f"{prefix}_crop_images.npy", directory / f"{prefix}_crop_masks.npy"
    images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.uint8, shape=(total, size, size))
    masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(total, size, size))
    position = 0
    for row in rows:
        image = cv2.imread(str(Path(image_dir) / row["file_name"]), cv2.IMREAD_GRAYSCALE)
        for annotation in coco.imgToAnns[row["id"]]:
            mask = coco.annToMask(annotation).astype(np.uint8)
            x0, y0, x1, y1 = square_bounds(mask)
            images[position] = cv2.resize(image[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_LINEAR)
            masks[position] = cv2.resize(mask[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST)
            position += 1
    images.flush()
    masks.flush()
    return image_path, mask_path, total


def train_refiner(model, train_loader, val_loader, epochs, output_path, device, log=print):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total = samples = 0
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = 0.5 * nn.functional.binary_cross_entropy_with_logits(logits, masks) \
                    + 0.5 * soft_dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(images)
            samples += len(images)
        model.eval()
        dice_total = count = 0
        with torch.inference_mode():
            for images, masks in val_loader:
                masks = masks.to(device).bool()
                predictions = model(images.to(device)).sigmoid() >= 0.65
                intersection = (predictions & masks).sum((1, 2, 3)).float()
                denominator = predictions.sum((1, 2, 3)) + masks.sum((1, 2, 3))
                dice_total += ((2 * intersection + 1e-7) / (denominator + 1e-7)).sum().item()
                count += len(images)
        val_dice = dice_total / max(count, 1)
        if val_dice > best:
            best = val_dice
            torch.save({"epoch": epoch, "model": model.state_dict()}, output_path)
        scheduler.step()
        history.append(dict(epoch=epoch, loss=total / samples, val_dice=val_dice,
                            minutes=(time.perf_counter() - started) / 60))
        log(f"REFINER {epoch:02d}/{epochs} | loss {history[-1]['loss']:.4f} | val Dice {val_dice:.4f} "
            f"| {history[-1]['minutes']:.2f} min")
    return history


# ----------------------------------------------------------------------------------------------- candidates + voting

def flip_array(array, axes, offset=0):
    return np.ascontiguousarray(np.flip(array, axis=tuple(a + offset for a in axes))) if axes else array


@torch.inference_mode()
def detect_views(detector, image, device, score_threshold, max_candidates, mask_threshold=0.5, flips=FLIPS):
    """Run the detector on flipped copies; each view keeps its top `max_candidates` detections."""
    detector.roi_heads.score_thresh = score_threshold
    detector.roi_heads.detections_per_img = max_candidates
    views = []
    for axes in flips:
        tensor = torch.from_numpy(flip_array(image, axes)).unsqueeze(0).repeat(3, 1, 1).to(device)
        output = detector([tensor])[0]
        masks = (output["masks"][:, 0] >= mask_threshold).cpu().numpy()
        scores = output["scores"].cpu().numpy()
        rles, kept = [], []
        for mask, score in zip(masks, scores):
            mask = flip_array(mask, axes)
            if mask.any():
                rles.append(encode_mask(mask))
                kept.append(float(score))
        views.append((rles, kept))
    return views


def cluster_views(views, match_iou=0.5):
    """Group instances across views: one-to-one IoU matching, at most one member per view."""
    clusters = []
    for view_index, (rles, scores) in enumerate(views):
        if not rles:
            continue
        matched = set()
        if clusters:
            overlaps = mask_utils.iou([c["rep"] for c in clusters], rles, [0] * len(rles))
            eligible = overlaps >= match_iou
            weights = eligible * (min(overlaps.shape) + 1 + overlaps)
            for row, column in zip(*linear_sum_assignment(-weights)):
                if eligible[row, column]:
                    clusters[row]["members"].append(rles[column])
                    clusters[row]["scores"].append(scores[column])
                    matched.add(column)
        clusters.extend(dict(rep=rle, members=[rle], scores=[scores[column]], origin=view_index)
                        for column, rle in enumerate(rles) if column not in matched)
    return clusters


def fuse_cluster(cluster, full_size):
    """Majority-vote mask of one cluster, resized to the full frame."""
    members = cluster["members"]
    votes = np.zeros(mask_utils.decode(members[0]).shape, dtype=np.uint8)
    for rle in members:
        votes += mask_utils.decode(rle)
    fused = votes >= (len(members) + 1) // 2
    if not fused.any():
        fused = votes > 0
    return cv2.resize(fused.astype(np.uint8), (full_size[1], full_size[0]), interpolation=cv2.INTER_NEAREST)


def square_crop(image, mask, size=256):
    x0, y0, x1, y1 = square_bounds(mask)
    return (x0, y0, x1, y1), cv2.resize(image[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_LINEAR)


@torch.inference_mode()
def refiner_probabilities(refiner, crops, device, batch=32, flips=FLIPS):
    """Average the refiner's probability over the flipped copies of each 256 x 256 crop."""
    out = []
    for start in range(0, len(crops), batch):
        tensor = torch.from_numpy(np.stack(crops[start:start + batch])).float().unsqueeze(1).repeat(1, 3, 1, 1)
        tensor = tensor.to(device) / 255
        total = 0
        for axes in flips:
            dims = tuple(a + 2 for a in axes)
            view = torch.flip(tensor, dims) if dims else tensor
            probability = refiner(view).sigmoid()
            total = total + (torch.flip(probability, dims) if dims else probability)
        out.append((total / len(flips)).cpu().numpy()[:, 0].astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 256, 256), np.float16)


def image_candidates(detector, refiner, image_path, device, settings):
    """All voted candidates of one photo: fused mask, refiner probability map and vote support."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    height, width = gray.shape
    small = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    views = detect_views(detector, small, device, settings["candidate_threshold"], settings["max_candidates"])
    clusters = cluster_views(views)
    fused = [fuse_cluster(c, (height, width)) for c in clusters]
    crops, bounds = [], []
    for mask in fused:
        box, crop = square_crop(gray, mask)
        bounds.append(box)
        crops.append(crop)
    probabilities = refiner_probabilities(refiner, crops, device) if refiner is not None and crops else []
    candidates = []
    for index, cluster in enumerate(clusters):
        candidates.append(dict(
            support=len(cluster["members"]), score=float(np.mean(cluster["scores"])),
            origin=cluster["origin"], first_score=float(cluster["scores"][0]),
            fallback=encode_mask(fused[index]), bounds=bounds[index], shape=(height, width),
            probability=probabilities[index] if len(probabilities) else None, refined={}))
    first = views[0]
    return dict(candidates=candidates, view0=first, shape=(height, width))


def refined_rle(candidate, threshold, single_piece):
    if candidate["probability"] is None:
        return candidate["fallback"]
    key = (threshold, single_piece)
    if key not in candidate["refined"]:
        x0, y0, x1, y1 = candidate["bounds"]
        probability = cv2.resize(candidate["probability"].astype(np.float32), (x1 - x0, y1 - y0),
                                 interpolation=cv2.INTER_LINEAR)
        keep = probability >= threshold
        if single_piece and keep.any():
            keep = largest_component(keep)
        if keep.any():
            full = np.zeros(candidate["shape"], dtype=np.uint8)
            full[y0:y1, x0:x1] = keep
            candidate["refined"][key] = encode_mask(full)
        else:
            candidate["refined"][key] = candidate["fallback"]
    return candidate["refined"][key]


def select_instances(item, cap, min_support, threshold, single_piece=True, min_area=64,
                     min_score=0.0, view0_only=False):
    """Vote down: drop clusters seen in too few views or with a low mean score, keep the best `cap`,
    refine, merge duplicates. `view0_only` mimics the public recipe (identity view, score floor, no voting)."""
    candidates = sorted(item["candidates"], key=lambda c: (c["support"], c["score"]), reverse=True)
    if view0_only:
        chosen = [c for c in candidates if c["origin"] == 0 and c["first_score"] >= min_score]
    else:
        chosen = [c for c in candidates if c["support"] >= min_support and c["score"] >= min_score]
    if not chosen and candidates:
        chosen = candidates[:1]
    chosen = chosen[:cap] if cap else chosen
    rles = [refined_rle(c, threshold, single_piece) for c in chosen]
    rles = [r for r in rles if mask_utils.area(r) >= min_area] or rles[:1]
    merged = merge_overlapping_rles(rles)
    merged.sort(key=lambda r: (float(mask_utils.toBbox(r)[1]), float(mask_utils.toBbox(r)[0])))
    return merged


def single_view_rles(item, score_threshold=0.6):
    """Reference: one view, score >= threshold, detector masks only, overlaps merged (the public recipe)."""
    rles, scores = item["view0"]
    height, width = item["shape"]
    out = []
    for rle, score in zip(rles, scores):
        if score >= score_threshold:
            mask = cv2.resize(mask_utils.decode(rle), (width, height), interpolation=cv2.INTER_NEAREST)
            out.append(encode_mask(mask))
    return merge_overlapping_rles(out)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2))
