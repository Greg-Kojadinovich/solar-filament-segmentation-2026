"""Solar-disk detection and per-image normalization for H-alpha frames."""

import cv2
import numpy as np
from scipy import ndimage as ndi

BIN_PIXELS = 1
OUTPUT_CENTER = 128
OUTPUT_SCALE = 32.0
RIM_ZONE = 30
SECTORS = 72
MIN_BIN_SAMPLES = 20
SMOOTH_SIGMA = 4.0
TOPHAT_KERNEL = 31
TOPHAT_SIGMA = 1.5
RIDGE_GAIN = 1.4


def _fit_circle(points):
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(design, x ** 2 + y ** 2, rcond=None)
    cx, cy = solution[0], solution[1]
    return cx, cy, float(np.sqrt(max(solution[2] + cx ** 2 + cy ** 2, 0.0)))


def _limb_points(gray, cx, cy, radii, angles):
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    samples = ndi.map_coordinates(
        gray, [cy + radii[None, :] * sin, cx + radii[None, :] * cos], order=1, mode="nearest")
    samples = ndi.gaussian_filter1d(samples, 1.5, axis=1)
    # The limb ring is the brightest feature; take the steepest drop just outside it.
    position = np.arange(samples.shape[1])[None, :]
    peak = np.argmax(samples, axis=1)[:, None]
    gradient = np.where((position >= peak) & (position <= peak + 20),
                        np.gradient(samples, axis=1), np.inf)
    edge = radii[np.argmin(gradient, axis=1)]
    return np.column_stack([cx + edge * np.cos(angles), cy + edge * np.sin(angles)])


def fit_disk(gray, passes=2):
    """Return (cx, cy, radius, ok): the circle at the steepest outward brightness drop.

    The limb is a thin bright ring. A threshold or the global steepest drop can lock
    onto a bright halo's outer edge in some frames, so the rim is taken from the
    ring itself.
    """
    gray = np.asarray(gray, dtype=np.float32)
    height, width = gray.shape
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    radii = np.arange(0.38 * width, 0.52 * width, 1.0)
    cx, cy = width / 2, height / 2
    radius, rms, inlier_fraction = 0.0, np.inf, 0.0
    for _ in range(passes):
        points = _limb_points(gray, cx, cy, radii, angles)
        # Start from the median radius so prominences cannot drag the first fit.
        first = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        seed = np.abs(first - np.median(first)) <= 6.0
        cx, cy, radius = _fit_circle(points[seed] if seed.sum() >= 100 else points)
        for _ in range(3):
            distance = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
            inliers = np.abs(distance - radius) <= 4.0
            if inliers.sum() < 100:
                break
            cx, cy, radius = _fit_circle(points[inliers])
        distance = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        inliers = np.abs(distance - radius) <= 4.0
        inlier_fraction = float(inliers.mean())
        rms = float(np.sqrt(np.mean((distance[inliers] - radius) ** 2))) if inliers.any() else np.inf
        radii = np.arange(max(radius - 60, 0.3 * width), radius + 60, 1.0)
    ok = (0.38 * width <= radius <= 0.52 * width and abs(cx - width / 2) <= 0.15 * width
          and abs(cy - height / 2) <= 0.15 * height and inlier_fraction >= 0.6 and rms <= 2.5)
    return float(cx), float(cy), float(radius), bool(ok)


def feature_channels(normalized, outside):
    """Three views of one normalized frame: the frame, a denoised copy, a dark-ridge map.

    The ridge map is a black top-hat (dark structures narrower than the kernel), lightly
    blurred; it is computed at half size and scaled back up because it is smooth.
    """
    height, width = normalized.shape
    smooth = cv2.GaussianBlur(normalized, (0, 0), SMOOTH_SIGMA)
    half = cv2.resize(normalized, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    ridge = cv2.morphologyEx(half, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    ridge = cv2.GaussianBlur(ridge, (0, 0), TOPHAT_SIGMA)
    ridge = cv2.resize(ridge, (width, height), interpolation=cv2.INTER_LINEAR)
    ridge = np.clip(ridge * RIDGE_GAIN, 0, 255).astype(np.uint8)
    smooth[outside] = OUTPUT_CENTER
    ridge[outside] = 0
    return np.stack([normalized, smooth, ridge], axis=-1)


def normalize_disk(gray, margin=10, features=False):
    """Radially flat-field the disk, z-score it, and replace everything beyond the rim.

    Pixels farther than radius + margin from the fitted center become the neutral
    value 128, as does the whole frame's fallback when no disk can be fitted.
    Returns the uint8 image and a dict of the fit. With features=True the image has three
    channels (see feature_channels).
    """
    gray = np.asarray(gray, dtype=np.float32)
    height, width = gray.shape
    cx, cy, radius, ok = fit_disk(gray)
    info = dict(cx=cx, cy=cy, radius=radius, margin=margin, disk_found=ok)
    yy, xx = np.mgrid[:height, :width]
    if not ok:
        center, spread = float(np.median(gray)), max(1.4826 * float(np.median(np.abs(gray - np.median(gray)))), 1.0)
        z = (gray - center) / spread
        flat = np.clip(OUTPUT_CENTER + OUTPUT_SCALE * z, 0, 255).astype(np.uint8)
        if features:
            flat = feature_channels(flat, np.zeros(flat.shape, dtype=bool))
        return flat, info
    r = np.hypot(xx - cx, yy - cy).astype(np.float32)
    # The profile spans the safety margin too, so the rim step is flattened, not clipped.
    inside = r <= radius + margin
    bins = int(np.ceil((radius + margin) / BIN_PIXELS)) + 1
    index = np.minimum((r / BIN_PIXELS).astype(np.int32), bins - 1)
    flat_index, flat_value = index[inside], gray[inside]
    order = np.argsort(flat_index, kind="stable")
    sorted_index, sorted_value = flat_index[order], flat_value[order]
    edges = np.searchsorted(sorted_index, np.arange(bins + 1))
    profile = np.full(bins, np.nan, dtype=np.float32)
    for b in range(bins):
        if edges[b + 1] - edges[b] >= 200:
            profile[b] = np.median(sorted_value[edges[b]:edges[b + 1]])
    known = ~np.isnan(profile)
    profile = np.interp(np.arange(bins), np.nonzero(known)[0], profile[known]).astype(np.float32)
    residual = gray - profile[index]
    # The limb ring is not perfectly circular, so near the rim use a per-sector profile.
    zone = inside & (r > radius - RIM_ZONE)
    first = int((radius - RIM_ZONE) / BIN_PIXELS)
    depth = bins - first
    angle = np.arctan2(yy - cy, xx - cx)
    sector = np.minimum(((angle + np.pi) / (2 * np.pi) * SECTORS).astype(np.int32), SECTORS - 1)
    key = (sector * depth + np.clip(index - first, 0, depth - 1))[zone]
    order = np.argsort(key, kind="stable")
    sorted_key, sorted_value = key[order], gray[zone][order]
    cuts = np.searchsorted(sorted_key, np.arange(SECTORS * depth + 1))
    local = np.full(SECTORS * depth, np.nan, dtype=np.float32)
    for k in range(SECTORS * depth):
        if cuts[k + 1] - cuts[k] >= MIN_BIN_SAMPLES:
            local[k] = np.median(sorted_value[cuts[k]:cuts[k + 1]])
    local_value = local[sector * depth + np.clip(index - first, 0, depth - 1)]
    use = zone & ~np.isnan(local_value)
    residual[use] = (gray - local_value)[use]
    core = r <= 0.95 * radius
    center = float(np.median(residual[core]))
    spread = max(1.4826 * float(np.median(np.abs(residual[core] - center))), 1.0)
    z = (residual - center) / spread
    output = np.clip(OUTPUT_CENTER + OUTPUT_SCALE * z, 0, 255)
    output[r > radius + margin] = OUTPUT_CENTER
    info.update(spread=spread)
    output = output.astype(np.uint8)
    if features:
        output = feature_channels(output, r > radius + margin)
    return output, info


def normalize_file(source, destination, margin=10, features=False):
    """Normalize one frame to a lossless-enough grayscale JPEG; return its disk fit."""
    from pathlib import Path
    from PIL import Image

    source, destination = Path(source), Path(destination)
    with Image.open(source) as image:
        if image.mode != "L":
            raise ValueError(f"Expected a grayscale frame: {source}")
        gray = np.asarray(image)
    output, info = normalize_disk(gray, margin, features)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    Image.fromarray(output).save(temporary, format="JPEG", quality=98, subsampling=0)
    temporary.replace(destination)
    info.update(name=source.name, shape=list(gray.shape))
    return info
