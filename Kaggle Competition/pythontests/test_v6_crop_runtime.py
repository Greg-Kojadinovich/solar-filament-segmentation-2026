import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from torch.utils.data import DataLoader

from v6_crop_runtime import (
    CachedInstanceCrops, base_refiner, crop_bounds, loader_options, normalized_crop,
    parallel_refiner, refine_proposals,
)


class CropRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "image.png"
        Image.fromarray(np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)).save(self.image)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[6:18, 8:22] = 1
        self.rle = mask_utils.encode(np.asfortranarray(mask))
        self.mask = mask

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, **kwargs):
        return CachedInstanceCrops([(self.image, self.rle)] * 3, self.root / "cache",
                                   size=16, minimum=8, **kwargs)

    def test_cache_matches_original_preprocessing_and_skips_redecode(self):
        dataset = self.dataset()
        x, y = dataset[0]
        bounds = crop_bounds(self.mask, minimum=8)
        with Image.open(self.image) as image:
            expected = normalized_crop(image, bounds, 16)
        np.testing.assert_array_equal(x.numpy(), expected)
        x0, y0, x1, y1 = bounds
        expected_y = np.asarray(Image.fromarray(self.mask[y0:y1, x0:x1]).resize(
            (16, 16), Image.Resampling.NEAREST))
        np.testing.assert_array_equal(y.numpy()[0], expected_y)
        self.assertEqual(len(list((self.root / "cache").glob("*.npy"))), 1)
        with patch("v6_crop_runtime.mask_utils.decode", side_effect=AssertionError("cache miss")):
            cached_x, cached_y = dataset[1]
        torch.testing.assert_close(x, cached_x)
        torch.testing.assert_close(y, cached_y)
        augmented = self.dataset(augment=True)
        self.assertEqual(augmented[0][0].shape, (3, 16, 16))
        self.assertTrue(set(augmented[0][1].unique().tolist()) <= {0, 1})

    def test_workers_spawn_real_loader_and_match_single_process(self):
        dataset = self.dataset()
        expected = list(DataLoader(dataset, batch_size=2, **loader_options(0, False)))
        loader = DataLoader(dataset, batch_size=2, **loader_options(2, False))
        actual = list(loader)
        for (x, y), (a, b) in zip(expected, actual):
            torch.testing.assert_close(x, a)
            torch.testing.assert_close(y, b)
        del loader

    def test_cache_corruption_is_explicit(self):
        dataset = self.dataset()
        dataset[0]
        path = next((self.root / "cache").glob("*.npy"))
        np.save(path, np.zeros((1, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Invalid cached"):
            dataset[0]

    def test_batched_predictions_preserve_order_and_equal_serial(self):
        class FakeRefiner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, x):
                self.batch_sizes.append(len(x))
                return x[:, :1]

        model = FakeRefiner()
        masks = np.stack([self.mask, np.roll(self.mask, 1, 0), self.mask,
                          np.zeros_like(self.mask), self.mask])
        batched = refine_proposals(self.image, masks, model, "cpu", (32, 32),
                                   batch_size=3, crop_size=16, minimum=8)
        self.assertEqual(model.batch_sizes, [3, 1])
        serial = refine_proposals(self.image, masks, FakeRefiner(), "cpu", (32, 32),
                                  batch_size=1, crop_size=16, minimum=8)
        for a, b in zip(batched, serial):
            self.assertEqual(a["bounds"], b["bounds"])
            self.assertEqual(a["coarse"], b["coarse"])
            np.testing.assert_array_equal(a["probability"], b["probability"])
        self.assertEqual(len(batched), 4)

    def test_device_wrapping_and_loader_guards(self):
        model = torch.nn.Linear(2, 1)
        self.assertIs(parallel_refiner(model, []), model)
        self.assertIs(parallel_refiner(model, [0]), model)
        with patch("v6_crop_runtime.torch.nn.DataParallel", return_value="wrapped") as parallel:
            self.assertEqual(parallel_refiner(model, [0, 1]), "wrapped")
            parallel.assert_called_once_with(model, device_ids=[0, 1], output_device=0)
        self.assertIs(base_refiner(model), model)
        self.assertTrue(loader_options(2, True)["pin_memory"])
        with self.assertRaises(ValueError):
            loader_options(-1, False)
        with self.assertRaises(ValueError):
            parallel_refiner(model, [0, 0])


if __name__ == "__main__":
    unittest.main()
