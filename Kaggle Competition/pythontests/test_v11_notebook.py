import ast
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from v8_runtime import YOLO_AUGMENT_DEFAULTS, rotate_batch, yolo_augmentation

ROOT = Path(__file__).resolve().parents[1]


class V11NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(
            (ROOT / "Notebooks/solar_filament_CV_V11_crop_refined.ipynb").read_text())
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]}

    def test_code_is_valid_without_saved_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell["source"])
                if not source.startswith("%"):
                    ast.parse(source)

    def test_thirty_epochs_with_rotation_augmentation(self):
        config = ast.parse(self.cells["v8-config"])
        assignment = next(node for node in config.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "V8" for t in node.targets))
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in assignment.value.keywords}
        self.assertEqual(settings["weights"], "yolo11s-seg.pt")
        self.assertEqual((settings["image_size"], settings["batch_size"], settings["nbs"]),
                         (1280, 12, 32))
        self.assertEqual((settings["epochs"], settings["patience"]), (30, 20))
        self.assertEqual((settings["refiner_epochs"], settings["refiner_patience"]), (30, 20))
        self.assertEqual(settings["yolo_augment"], {
            "flipud": 0.5, "degrees": 180.0, "hsv_v": 0.2, "mosaic": 0.15, "close_mosaic": 10})
        self.assertIs(settings["refiner_rotate"], True)
        self.assertIs(settings["feature_channels"], True)
        self.assertIs(settings["single_piece"], True)
        self.assertEqual(settings["min_areas"], [64, 250])
        self.assertEqual(settings["confidences"], [0.15, 0.25, 0.35, 0.5])
        self.assertEqual(settings["gate_pixels"], [8])
        self.assertIs(settings["refiner_mask_channel"], True)
        self.assertIs(settings["refiner_jitter"], True)
        self.assertNotIn("refiner_zoom", settings)  # the zoom pass is switched off
        self.assertEqual((settings["tta_views"], settings["tta_instance_votes"],
                          settings["tta_pixel_votes"]), (8, [3, 4, 5], [3, 4, 5]))
        yolo_augmentation(settings)

    def test_naming_embedding_and_no_upload(self):
        source = self.cells["v8-config"]
        self.assertIn('VERSION = "V11"', source)
        self.assertIn("v11_single_", source)
        self.assertIn("## Rotation augmentation (V11)", self.cells["v8-intro"])
        for name in ["grouped_yolo_cv.py", "submission_masks.py", "solar_disk.py",
                     "v6_crop_runtime.py", "v7_yolo_runtime.py", "v8_runtime.py"]:
            self.assertIn(repr((ROOT / "Python Files" / name).read_text()), self.cells["v8-runtime"])
        joined = "".join(self.cells.values())
        for forbidden in ["kaggle_secrets", "get_secret", '"competitions", "submit"']:
            self.assertNotIn(forbidden, joined)
        self.assertEqual(self.notebook["cells"][-1]["id"], "v8-final-check")


class AugmentationTests(unittest.TestCase):
    def test_older_notebooks_keep_the_original_detector_augmentation(self):
        self.assertEqual(yolo_augmentation({}), YOLO_AUGMENT_DEFAULTS)
        self.assertEqual(YOLO_AUGMENT_DEFAULTS["degrees"], 10.0)
        self.assertEqual(YOLO_AUGMENT_DEFAULTS["flipud"], 0.0)
        self.assertEqual((YOLO_AUGMENT_DEFAULTS["mosaic"], YOLO_AUGMENT_DEFAULTS["close_mosaic"]),
                         (0.3, 5))

    def test_overrides_merge_and_unknown_keys_fail(self):
        merged = yolo_augmentation({"yolo_augment": {"flipud": 0.5, "degrees": 180.0}})
        self.assertEqual((merged["flipud"], merged["degrees"], merged["fliplr"]), (0.5, 180.0, 0.5))
        with self.assertRaises(ValueError):
            yolo_augmentation({"yolo_augment": {"bogus": 1}})

    def make_batch(self, size=64):
        y = torch.zeros(2, 1, size, size)
        y[:, :, 10:30, 20:26] = 1
        y[:, :, 28:34, 20:50] = 1
        x = y * 2 - 1
        return x.repeat(1, 3, 1, 1), y

    def test_zero_angle_is_the_identity(self):
        x, y = self.make_batch()
        rx, ry = rotate_batch(x, y, angles=torch.zeros(2))
        self.assertTrue(torch.allclose(rx, x, atol=1e-5))
        self.assertTrue(torch.equal(ry, y))

    def test_quarter_turn_matches_a_numpy_rotation_for_image_and_mask(self):
        x, y = self.make_batch()
        rx, ry = rotate_batch(x, y, angles=torch.full((2,), np.pi / 2))
        expected = [np.rot90(y[0, 0].numpy(), k) for k in (1, -1)]
        self.assertTrue(any(np.array_equal(ry[0, 0].numpy(), e) for e in expected))
        self.assertTrue(torch.equal((rx[:, 0:1] > 0).float(), ry))

    def test_arbitrary_angles_keep_mask_area_and_image_consistency(self):
        x, y = self.make_batch()
        angles = torch.tensor([0.7, 2.4])
        rx, ry = rotate_batch(x, y, angles=angles)
        for index in range(2):
            self.assertAlmostEqual(ry[index].sum().item() / y[index].sum().item(), 1.0, delta=0.08)
        agreement = ((rx[:, 0:1] > 0).float() == ry).float().mean().item()
        self.assertGreater(agreement, 0.97)

    def test_random_angles_are_drawn_per_sample(self):
        x, y = self.make_batch()
        torch.manual_seed(0)
        rx, ry = rotate_batch(x.repeat(8, 1, 1, 1), y.repeat(8, 1, 1, 1))
        self.assertGreater(len({tuple(ry[i].flatten().nonzero()[:3].flatten().tolist())
                                for i in range(16)}), 1)


if __name__ == "__main__":
    unittest.main()
