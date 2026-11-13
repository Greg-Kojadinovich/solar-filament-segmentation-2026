import ast
import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pycocotools.mask as mask_utils

from submission_masks import (
    MASK_POLICY, SUBMISSION_FIELDS, disjoint_instances, repair_submission,
    validate_disjoint, validate_submission_rows,
)


SHAPE = (8, 8)


def encode(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


class SubmissionMaskTests(unittest.TestCase):
    def setUp(self):
        self.first = np.zeros(SHAPE, dtype=bool)
        self.first[1:5, 1:5] = True
        self.second = np.zeros(SHAPE, dtype=bool)
        self.second[3:7, 3:7] = True

    def test_overlap_reproduces_rejection(self):
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_disjoint([encode(self.first), encode(self.second)], SHAPE)

    def test_first_wins_preserves_union_and_unique_pixels(self):
        masks, audit = disjoint_instances([encode(self.first), encode(self.second)], SHAPE)
        validate_disjoint(masks, SHAPE)
        a, b = [mask_utils.decode(r).astype(bool) for r in masks]
        np.testing.assert_array_equal(a, self.first)
        np.testing.assert_array_equal(b, self.second & ~self.first)
        np.testing.assert_array_equal(a | b, self.first | self.second)
        self.assertEqual(audit["overlap_pixels_removed"], 4)
        again, _ = disjoint_instances(masks, SHAPE)
        self.assertEqual(masks, again)

    def test_duplicate_and_small_remainder_removed(self):
        masks, audit = disjoint_instances(
            [encode(self.first), encode(self.first), encode(self.second)], SHAPE, min_area=13,
        )
        self.assertEqual(len(masks), 1)
        self.assertEqual(audit["dropped_indices"], [1, 2])

    def test_dropped_small_mask_does_not_claim_pixels(self):
        tiny = np.zeros(SHAPE, dtype=bool)
        tiny[1, 1] = True
        masks, audit = disjoint_instances([encode(tiny), encode(self.first)], SHAPE, min_area=2)
        np.testing.assert_array_equal(mask_utils.decode(masks[0]), self.first)
        self.assertEqual(audit["kept_indices"], [1])

    def test_empty_invalid_shape_and_threshold_rejected(self):
        for rle in [encode(np.zeros(SHAPE)), encode(np.ones((4, 4)))]:
            with self.assertRaises(ValueError):
                disjoint_instances([rle], SHAPE)
        with self.assertRaises(ValueError):
            disjoint_instances([], SHAPE, min_area=0)
        self.assertEqual(disjoint_instances([], SHAPE)[0], [])

    def test_csv_repair_interleaved_images_and_preserved_source(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "original.csv", Path(directory) / "fixed.csv"
            rows = [
                dict(filament_id="test_a_1", segmentation_rle=encode(self.first)["counts"].decode()),
                dict(filament_id="test_b_1", segmentation_rle=encode(self.first)["counts"].decode()),
                dict(filament_id="test_a_2", segmentation_rle=encode(self.second)["counts"].decode()),
                dict(filament_id="test_a_3", segmentation_rle=encode(self.first)["counts"].decode()),
            ]
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=SUBMISSION_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            original_bytes = source.read_bytes()
            report = repair_submission(source, output, {"test_a", "test_b", "test_c"}, 1, SHAPE)
            with output.open(newline="") as stream:
                repaired = list(csv.DictReader(stream))
            self.assertEqual([r["filament_id"] for r in repaired], ["test_a_1", "test_b_1", "test_a_2"])
            self.assertEqual(report["images_with_overlap"], 1)
            self.assertEqual(report["dropped_rows"], 1)
            self.assertEqual(report["validation"]["zero_detection_images"], ["test_c"])
            self.assertEqual(source.read_bytes(), original_bytes)
            validate_submission_rows(repaired, {"test_a", "test_b", "test_c"}, SHAPE)
            with self.assertRaises(FileExistsError):
                repair_submission(source, output, {"test_a", "test_b"}, 1, SHAPE)
            with self.assertRaises(ValueError):
                repair_submission(source, source, {"test_a", "test_b"}, 1, SHAPE)

    def test_invalid_or_duplicate_ids_rejected(self):
        row = dict(filament_id="test_a_1", segmentation_rle=encode(self.first)["counts"].decode())
        for rows, names in [([row, row], {"test_a"}), ([row], {"other"}), ([], {"test_a"})]:
            with self.assertRaises(ValueError):
                validate_submission_rows(rows, names, SHAPE)

    def test_notebook_embeds_shared_policy_and_applies_to_both_modes(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "solar_filament_CV_V3.ipynb").read_text())
        cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
        shared = (root / "submission_masks.py").read_text().split('\nif __name__ == "__main__":')[0]
        self.assertEqual(cells["v3-disjoint"], shared)
        workflow = ast.parse(cells["v3-workflow"])
        needed = {"v3_instances", "v3_calibrate", "v3_export", "v3_scores"}
        functions = [node for node in workflow.body
                     if isinstance(node, ast.FunctionDef) and node.name in needed]
        ns = {}
        exec(compile(shared, "v3-disjoint", "exec"), ns)
        ns.update(FULL_SHAPE=SHAPE, Image=__import__("PIL.Image", fromlist=["Image"]))
        exec(compile(ast.Module(body=functions, type_ignores=[]), "workflow", "exec"), ns)
        proposals = [
            dict(coarse=encode(self.first), bounds=(0, 0, 8, 8), probability=self.first.astype(float)),
            dict(coarse=encode(self.second), bounds=(0, 0, 8, 8), probability=self.second.astype(float)),
        ]
        for mode in ["yolo", "refined"]:
            masks, fallbacks = ns["v3_instances"](
                proposals, dict(mode=mode, min_area=1, refiner_threshold=0.5),
            )
            validate_disjoint(masks, SHAPE)
            self.assertEqual(fallbacks, 0)
            self.assertEqual(sum(int(mask_utils.area(r)) for r in masks), 28)
        def checked_score(predicted, ground_truth):
            validate_disjoint(predicted, SHAPE)
            return {"pq": 1.0}

        ns.update(
            records_by_name={"test_a": [{"id": 1}]}, gt_rles=lambda identity: [],
            local_scores=checked_score,
        )
        scores, _ = ns["v3_scores"](
            ["test_a"], {"test_a": proposals},
            dict(mode="refined", min_area=1, refiner_threshold=0.5),
        )
        self.assertEqual(scores["pq"], 1.0)
        ns.update(
            OUTPUT_DIR=Path("/not-used"), SMOKE_TEST=False,
            V3={}, annotation_sha256="test",
        )
        with self.assertRaisesRegex(ValueError, "Re-run calibration"):
            ns["v3_export"](None, None, {"postprocessing": "old-policy"})
        self.assertEqual(ns["MASK_POLICY"], MASK_POLICY)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ["yolo_best.pt", "refiner_best.pt"]:
                (output / name).write_bytes(b"test checkpoint, not a trained model")
            hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                      for name in ["yolo_best.pt", "refiner_best.pt"]}
            parameters = dict(mode="refined", confidence=0.25, min_area=1, refiner_threshold=0.5)
            calibration = dict(
                hashes=hashes, settings={}, annotation_sha256="test",
                smoke_test=False, postprocessing=MASK_POLICY,
                selected=dict(parameters=parameters), best_by_mode={},
            )
            ns.update(
                OUTPUT_DIR=output, test_paths=[Path("test_a.jpeg"), Path("test_b.jpeg")],
                v3_proposals=lambda *args: proposals,
                tqdm=lambda values, **kwargs: values,
                write_json=lambda path, value: path.write_text(json.dumps(value)),
                RUN_V3_CALIBRATION=False,
            )
            (output / "v3_calibration.json").write_text(json.dumps(calibration))
            self.assertEqual(ns["v3_calibrate"](None, None)["postprocessing"], MASK_POLICY)
            old = dict(calibration)
            old.pop("postprocessing")
            (output / "v3_calibration.json").write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "calibration does not match"):
                ns["v3_calibrate"](None, None)
            exported = ns["v3_export"](None, None, calibration)
            with exported.open(newline="") as stream:
                exported_rows = list(csv.DictReader(stream))
            result = validate_submission_rows(exported_rows, {"test_a", "test_b"}, SHAPE)
            self.assertEqual(result["rows"], 4)
            manifest = json.loads((output / "submission_manifest.json").read_text())
            self.assertEqual(manifest["validation"]["overlapping_pixels"], 0)
            self.assertEqual(manifest["postprocessing"], MASK_POLICY)
            ns["v3_instances"] = lambda *args: ([encode(self.first), encode(self.second)], 0)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                ns["v3_export"](None, None, calibration)
            calibration["hashes"] = {}
            with self.assertRaisesRegex(ValueError, "Export calibration"):
                ns["v3_export"](None, None, calibration)


if __name__ == "__main__":
    unittest.main()
