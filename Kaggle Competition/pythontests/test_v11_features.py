import tempfile
import unittest
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image

from solar_disk import normalize_disk, normalize_file
from v8_runtime import (
    CachedFeatureCrops, VIEWS, _yolo_source, apply_view, cluster_views, cluster_votes,
    fuse_clusters, invert_mask, refine_image,
)

SIZE = 512


def synthetic_disk(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    r = np.hypot(xx - 256, yy - 256)
    frame = np.where(r <= 225, 120 - 25 * (r / 225) ** 2 + rng.normal(0, 4, (SIZE, SIZE)),
                     8 + rng.normal(0, 2, (SIZE, SIZE)))
    frame += 120 * np.exp(-(((r - 225) / 1.5) ** 2))
    frame[250:256, 200:330] -= 55  # a dark filament
    return np.clip(frame, 0, 255).astype(np.uint8), r


def rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


class FeatureChannelTests(unittest.TestCase):
    def test_three_channels_with_a_neutral_outside_and_a_bright_ridge_on_the_filament(self):
        frame, r = synthetic_disk()
        output, info = normalize_disk(frame, 10, features=True)
        self.assertEqual((output.shape, output.dtype), ((SIZE, SIZE, 3), np.uint8))
        outside = r > info["radius"] + 11
        self.assertTrue(np.all(output[..., 0][outside] == 128))
        self.assertTrue(np.all(output[..., 1][outside] == 128))
        self.assertTrue(np.all(output[..., 2][outside] == 0))
        ridge = output[..., 2].astype(float)
        self.assertGreater(ridge[250:256, 220:310].mean(), ridge[200:240, 220:310].mean() + 25)

    def test_the_second_channel_is_a_denoised_copy_of_the_first(self):
        frame, r = synthetic_disk()
        output, _ = normalize_disk(frame, 10, features=True)
        inside = (r > 60) & (r < 150)
        self.assertLess(output[..., 1][inside].std(), 0.6 * output[..., 0][inside].std())

    def test_single_channel_output_is_unchanged_when_features_are_off(self):
        frame, _ = synthetic_disk()
        flat, _ = normalize_disk(frame, 10)
        rgb, _ = normalize_disk(frame, 10, features=True)
        self.assertEqual(flat.ndim, 2)
        self.assertTrue(np.array_equal(flat, rgb[..., 0]))

    def test_files_are_written_as_rgb_jpeg(self):
        directory = Path(tempfile.mkdtemp())
        frame, _ = synthetic_disk()
        Image.fromarray(frame).save(directory / "a.jpeg")
        normalize_file(directory / "a.jpeg", directory / "out" / "a.jpeg", 10, True)
        with Image.open(directory / "out" / "a.jpeg") as image:
            self.assertEqual((image.mode, image.size), ("RGB", (SIZE, SIZE)))


class FeatureCropTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(1)
        self.path = self.directory / "f.jpeg"
        Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)).save(
            self.path, quality=98, subsampling=0)
        mask = np.zeros((SIZE, SIZE), np.uint8)
        mask[100:140, 200:215] = 1
        self.rle = rle(mask)
        self.mask = mask

    def test_crops_are_three_channel_cached_and_prewarmed_once(self):
        crops = CachedFeatureCrops([(self.path, self.rle)] * 2, self.directory / "cache", size=32,
                                   minimum=48)
        crops.prewarm(2)
        files = list((self.directory / "cache").glob("*.npy"))
        self.assertEqual(len(files), 1)
        self.assertEqual(np.load(files[0]).shape, (4, 32, 32))
        x, y = crops[0]
        self.assertEqual((tuple(x.shape), tuple(y.shape)), ((3, 32, 32), (1, 32, 32)))
        self.assertGreater(y.sum().item(), 0)
        self.assertTrue(-1.0 <= x.min().item() and x.max().item() <= 1.0)

    def test_a_cache_miss_builds_the_same_crop_as_prewarming(self):
        first = CachedFeatureCrops([(self.path, self.rle)], self.directory / "a", size=32, minimum=48)
        second = CachedFeatureCrops([(self.path, self.rle)], self.directory / "b", size=32, minimum=48)
        second.prewarm(1)
        self.assertTrue(torch.equal(first[0][0], second[0][0]))

    def test_refine_image_accepts_gray_and_three_channel_frames(self):
        refiner = torch.nn.Conv2d(3, 1, 1).eval()
        masks = np.stack([self.mask, np.zeros_like(self.mask)])
        masks[1, 300:330, 300:340] = 1
        for mode in ("L", "RGB"):
            image = Image.new(mode, (SIZE, SIZE), 100)
            proposals = refine_image(image, masks, refiner, "cpu", (SIZE, SIZE), batch_size=2,
                                     crop_size=32, minimum=48)
            self.assertEqual(len(proposals), 2)
            self.assertEqual(proposals[0]["probability"].shape, (32, 32))

    def test_yolo_source_uses_opencv_channel_order(self):
        rgb = np.zeros((4, 4, 3), np.uint8)
        rgb[..., 0] = 10
        rgb[..., 2] = 30
        bgr = _yolo_source(Image.fromarray(rgb))
        self.assertEqual((bgr[0, 0, 0], bgr[0, 0, 2]), (30, 10))
        gray = _yolo_source(Image.fromarray(np.full((4, 4), 7, np.uint8)))
        self.assertEqual(gray.shape, (4, 4, 3))


class VotingTests(unittest.TestCase):
    def test_every_view_maps_back_exactly(self):
        array = np.random.default_rng(2).integers(0, 255, (64, 64), dtype=np.uint8)
        for k, flip in VIEWS:
            self.assertTrue(np.array_equal(invert_mask(apply_view(array, k, flip), k, flip), array))
        self.assertEqual(len(VIEWS), 8)

    def make_lists(self):
        shape = (96, 96)
        core = np.zeros(shape, bool)
        core[30:60, 40:50] = True
        lists = []
        for view in range(8):
            mask = core.copy()
            if view < 2:
                mask[30:60, 50:53] = True  # a fringe only two views include
            spurious = np.zeros(shape, bool)
            if view < 2:
                spurious[5:15, 5:15] = True  # an object only two views find
            items = [rle(mask)] + ([rle(spurious)] if view < 2 else [])
            lists.append(items)
        return shape, core, lists

    def test_instances_and_pixels_need_enough_votes(self):
        shape, core, lists = self.make_lists()
        clusters = cluster_views(lists)
        self.assertEqual(sorted(len(c["members"]) for c in clusters), [2, 8])
        votes = cluster_votes(clusters, shape, 3)
        self.assertEqual(len(votes), 1)
        (instance,) = fuse_clusters(clusters, votes, 3, 5, 20, shape)
        result = mask_utils.decode(instance).astype(bool)
        self.assertTrue(np.array_equal(result, core))

    def test_a_low_pixel_vote_keeps_the_fringe_and_a_high_instance_vote_drops_everything(self):
        shape, core, lists = self.make_lists()
        clusters = cluster_views(lists)
        votes = cluster_votes(clusters, shape, 1)
        (loose,) = fuse_clusters(clusters, votes, 3, 2, 20, shape)
        self.assertGreater(mask_utils.area(loose), core.sum())
        self.assertEqual(fuse_clusters(clusters, votes, 9, 1, 20, shape), [])


if __name__ == "__main__":
    unittest.main()
