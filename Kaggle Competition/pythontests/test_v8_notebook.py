import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from v6_crop_runtime import crop_bounds, normalized_crop
from v8_runtime import (
    MODES, build_instances, candidate_parameters, merge_holdout_into_training, prepare_images,
    validate_result, verify_ready_to_submit,
)

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (256, 256)


def rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


class V8NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(
            (ROOT / "Notebooks/solar_filament_CV_V8_crop_refined.ipynb").read_text())
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]}

    def test_notebook_code_is_valid_and_has_no_saved_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                self.assertEqual(cell["outputs"], [])
                if not source.startswith("%"):
                    ast.parse(source)

    def test_settings_are_one_split_twenty_epochs_with_disk_normalization(self):
        config = ast.parse(self.cells["v8-config"])
        assignment = next(node for node in config.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "V8" for t in node.targets))
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in assignment.value.keywords}
        self.assertEqual((settings["epochs"], settings["patience"]), (20, 20))
        self.assertEqual(settings["refiner_epochs"], 20)
        self.assertEqual(settings["batch_size"], 16)
        self.assertEqual((settings["nbs"], settings["warmup_epochs"]), (32, 1))
        self.assertEqual(settings["refiner_gpus"], [0, 1])
        self.assertEqual(settings["refiner_batch"], 32)
        self.assertEqual(settings["disk_margin"], 10)
        self.assertEqual((settings["crop_size"], settings["crop_context"], settings["minimum_crop"]),
                         (256, 1.8, 96))
        self.assertEqual(settings["refiner_encoder"], "resnet34")
        self.assertGreaterEqual(min(settings["confidences"]), 0.1)
        source = self.cells["v8-config"]
        self.assertIn("DISK_NORMALIZE = True", source)
        self.assertIn("TRAIN_ON_HOLDOUT = True", source)
        self.assertIn("FOLDS_TO_RUN = [SPLIT_FOLD]", source)
        self.assertIn("YOLO_TRAIN_DEVICE = [0, 1]", source)
        self.assertIn("requires two visible GPUs", source)
        self.assertNotIn("ensemble", self.cells["v8-export"].lower())

    def test_scratch_files_stay_out_of_the_output_folder(self):
        self.assertIn('runtime_dir = SCRATCH_DIR / "_runtime"', self.cells["v8-runtime"])
        self.assertIn('CROP_CACHE_DIR = SCRATCH_DIR / "_crop_cache"', self.cells["v8-config"])
        self.assertIn("/kaggle/temp", self.cells["v8-config"])

    def test_preprocessing_precedes_planning_and_training(self):
        order = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(order.index("v8-data"), order.index("v8-preprocess"))
        self.assertLess(order.index("v8-preprocess"), order.index("v8-plan"))
        self.assertLess(order.index("v8-plan"), order.index("v8-run"))
        self.assertIn("prepare_images(globals())", self.cells["v8-preprocess"])

    def test_embedded_runtime_files_match_sources(self):
        for name in ["grouped_yolo_cv.py", "submission_masks.py", "solar_disk.py",
                     "v6_crop_runtime.py", "v7_yolo_runtime.py", "v8_runtime.py"]:
            source = (ROOT / "Python Files" / name).read_text()
            self.assertIn(repr(source), self.cells["v8-runtime"], name)

    def test_preflight_only_reports_and_runs_before_the_data_cell(self):
        import tempfile

        import torch

        order = [cell["id"] for cell in self.notebook["cells"]]
        self.assertLess(order.index("v8-config"), order.index("v8-preflight"))
        self.assertLess(order.index("v8-preflight"), order.index("v8-data"))
        self.assertNotIn("raise", self.cells["v8-preflight"])
        exec(self.cells["v8-preflight"], dict(torch=torch, OUTPUT_DIR=Path(tempfile.mkdtemp())))

    def test_no_automatic_upload_and_final_check_never_raises(self):
        joined = "".join(self.cells.values())
        for forbidden in ["kaggle_secrets", "get_secret", '"competitions", "submit"']:
            self.assertNotIn(forbidden, joined)
        order = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(order[-1], "v8-final-check")
        self.assertLess(order.index("v8-export"), order.index("v8-final-check"))
        from unittest import mock

        namespace = dict(
            submission_path=Path("/tmp/s.csv"), COMPETITION="comp", PREPROCESSING="raw",
            CV_ROOT=Path("/tmp/x/run"),
            verify_ready_to_submit=mock.Mock(side_effect=ValueError("blocked")))
        exec(self.cells["v8-final-check"], namespace)
        namespace["verify_ready_to_submit"].assert_called_once()


class SubmitGuardTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        import tempfile

        self.directory = Path(tempfile.mkdtemp())
        self.csv = self.directory / "submission_V8_disk_normalized.csv"
        self.csv.write_text("filament_id,segmentation_rle\n")
        self.digest = hashlib.sha256(self.csv.read_bytes()).hexdigest()
        self.ns = dict(CV_ROOT=self.directory, SMOKE_TEST=False, test_paths=list(range(180)))

    def audit(self, **validation):
        values = dict(overlapping_pixels=0, images_with_predictions=178,
                      masks_per_image_with_predictions=6.5)
        values.update(validation)
        (self.directory / "submission_audit.json").write_text(json.dumps(dict(
            validation=values, submission_filename=self.csv.name,
            submission_sha256=self.digest)))

    def test_valid_submission_passes(self):
        self.audit()
        self.assertEqual(verify_ready_to_submit(self.ns, self.csv)["submission_sha256"], self.digest)

    def test_over_prediction_is_blocked_unless_forced(self):
        self.audit(masks_per_image_with_predictions=11.6)
        with self.assertRaises(ValueError):
            verify_ready_to_submit(self.ns, self.csv)
        verify_ready_to_submit(self.ns, self.csv, force=True)

    def test_tampered_file_overlaps_and_low_coverage_are_blocked(self):
        self.audit()
        self.csv.write_text("changed")
        with self.assertRaises(ValueError):
            verify_ready_to_submit(self.ns, self.csv, force=True)
        self.csv.write_text("filament_id,segmentation_rle\n")
        self.audit(overlapping_pixels=3)
        with self.assertRaises(ValueError):
            verify_ready_to_submit(self.ns, self.csv)
        self.audit(images_with_predictions=100)
        with self.assertRaises(ValueError):
            verify_ready_to_submit(self.ns, self.csv)

    def test_smoke_runs_are_never_submitted(self):
        self.audit()
        with self.assertRaises(ValueError):
            verify_ready_to_submit(dict(self.ns, SMOKE_TEST=True), self.csv)


class AllDataSplitTests(unittest.TestCase):
    plan = dict(fold=0, splits=dict(train=["a", "b"], tune=["c"], holdout=["d", "e"]))

    def test_holdout_joins_training_and_tuning_stays_out(self):
        merged = merge_holdout_into_training(self.plan)
        self.assertEqual(merged["splits"], dict(train=["a", "b", "d", "e"], tune=["c"], holdout=[]))
        self.assertEqual(self.plan["splits"]["holdout"], ["d", "e"])

    def test_overlapping_days_are_rejected(self):
        bad = dict(fold=0, splits=dict(train=["a"], tune=["c"], holdout=["a"]))
        with self.assertRaises(ValueError):
            merge_holdout_into_training(bad)

    def test_result_without_a_holdout_validates_and_stray_outer_rows_do_not(self):
        merged = merge_holdout_into_training(self.plan)
        result = dict(plan_sha256="h", fold=0, status="complete", outer_rows=[],
                      checkpoint_hashes={"yolo_best.pt": "x"}, selected_parameters={"mode": "yolo"})
        validate_result(result, merged, "h")
        with self.assertRaises(ValueError):
            validate_result(dict(result, outer_rows=[{}]), merged, "h")
        with self.assertRaises(ValueError):
            validate_result(result, merged, "other")


class PrepareImagesTests(unittest.TestCase):
    def test_threads_write_normalized_copies_and_remap_lookups(self):
        import tempfile

        from test_solar_disk import synthetic_frame

        root = Path(tempfile.mkdtemp())
        frame, _ = synthetic_frame()
        train, test = {}, []
        for index in range(3):
            path = root / f"train{index}.jpeg"
            Image.fromarray(frame).save(path, quality=95)
            train[path.name] = path
        test_path = root / "test0.jpeg"
        Image.fromarray(frame).save(test_path, quality=95)
        test.append(test_path)
        ns = dict(V8=dict(disk_margin=10, preprocess_workers=2), CV_ROOT=root / "run",
                  SCRATCH_DIR=root / "scratch",
                  train_lookup=train, test_paths=test)
        (root / "run").mkdir()
        new_train, new_test, report = prepare_images(ns)
        self.assertEqual(sorted(new_train), sorted(train))
        self.assertTrue(all(p.is_file() and "_preprocessed" in str(p) for p in new_train.values()))
        self.assertEqual([p.name for p in new_test], ["test0.jpeg"])
        self.assertEqual((report["frames"], report["disk_not_found"]), (4, []))
        self.assertTrue((root / "run" / "disk_fits.json").is_file())


class CropGeometryTests(unittest.TestCase):
    def test_window_is_square_clipped_and_contains_the_mask(self):
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        mask[100:160, 1990:2040] = 1
        x0, y0, x1, y1 = crop_bounds(mask)
        side = x1 - x0
        self.assertEqual(side, y1 - y0)
        self.assertEqual(side, int(np.ceil(60 * 1.8)))
        self.assertTrue(0 <= x0 and x1 <= 2048 and 0 <= y0 and y1 <= 2048)
        self.assertTrue(x0 <= 1990 and x1 >= 2040 and y0 <= 100 and y1 >= 160)

    def test_small_objects_get_the_minimum_window(self):
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        mask[500:510, 500:520] = 1
        x0, _, x1, _ = crop_bounds(mask)
        self.assertEqual(x1 - x0, 96)

    def test_normalized_crop_is_three_channel_and_scaled(self):
        image = Image.fromarray(np.full((64, 64), 255, dtype=np.uint8))
        crop = normalized_crop(image, (0, 0, 32, 32), 16)
        self.assertEqual(crop.shape, (3, 16, 16))
        self.assertTrue(np.allclose(crop, 1.0))


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.zeros(SHAPE, dtype=bool)
        self.truth[100:140, 100:112] = True
        self.neighbor = np.zeros(SHAPE, dtype=bool)
        self.neighbor[100:140, 150:162] = True
        self.coarse = np.zeros(SHAPE, dtype=bool)
        self.coarse[96:144, 96:116] = True
        self.bounds = (64, 64, 192, 192)
        window = (self.truth | self.neighbor)[64:192, 64:192].astype(np.float32)
        probability = np.asarray(Image.fromarray(window).resize((256, 256), Image.Resampling.BILINEAR))
        self.proposal = dict(coarse=rle(self.coarse), bounds=self.bounds, probability=probability)

    def parameters(self, mode, **overrides):
        values = dict(mode=mode, confidence=0.25, refiner_threshold=0.5,
                      gate_pixels=6 if mode == "gated" else None, min_area=20)
        values.update(overrides)
        return values

    def iou(self, instance):
        mask = mask_utils.decode(instance).astype(bool)
        return (mask & self.truth).sum() / (mask | self.truth).sum()

    def test_yolo_mode_returns_the_coarse_mask(self):
        (instance,), fallbacks = build_instances([self.proposal], self.parameters("yolo"), SHAPE)
        self.assertEqual(fallbacks, 0)
        self.assertTrue(np.array_equal(mask_utils.decode(instance).astype(bool), self.coarse))

    def test_refined_mode_improves_overlap_but_can_absorb_a_neighbor(self):
        (refined,), _ = build_instances([self.proposal], self.parameters("refined"), SHAPE)
        (coarse,), _ = build_instances([self.proposal], self.parameters("yolo"), SHAPE)
        self.assertTrue(mask_utils.decode(refined).astype(bool)[:, 150:162].any())
        self.assertLess(self.iou(refined), 0.6)
        self.assertGreater(self.iou(coarse), 0.0)

    def test_gated_mode_drops_the_neighbor_and_tightens_the_boundary(self):
        (gated,), _ = build_instances([self.proposal], self.parameters("gated"), SHAPE)
        (coarse,), _ = build_instances([self.proposal], self.parameters("yolo"), SHAPE)
        self.assertFalse(mask_utils.decode(gated).astype(bool)[:, 150:162].any())
        self.assertGreater(self.iou(gated), self.iou(coarse))
        self.assertGreater(self.iou(gated), 0.9)

    def test_empty_refinement_falls_back_to_the_detector_mask(self):
        proposal = dict(self.proposal, probability=np.zeros((256, 256), dtype=np.float32))
        (instance,), fallbacks = build_instances([proposal], self.parameters("refined"), SHAPE)
        self.assertEqual(fallbacks, 1)
        self.assertTrue(np.array_equal(mask_utils.decode(instance).astype(bool), self.coarse))

    def test_instances_are_disjoint_and_unknown_modes_fail(self):
        instances, _ = build_instances([self.proposal, self.proposal], self.parameters("refined"), SHAPE)
        self.assertEqual(len(instances), 1)
        with self.assertRaises(ValueError):
            build_instances([self.proposal], self.parameters("bogus"), SHAPE)

    def test_candidate_grid_covers_every_mode_once_per_threshold(self):
        settings = dict(min_areas=[64], refiner_thresholds=[0.5, 0.65], gate_pixels=[12])
        candidates = list(candidate_parameters(settings, 0.25))
        self.assertEqual({c["mode"] for c in candidates}, set(MODES) - {"zoom"})
        self.assertEqual(len(candidates), 5)


if __name__ == "__main__":
    unittest.main()
