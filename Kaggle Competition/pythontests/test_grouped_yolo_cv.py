import gc
import ast
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pycocotools.mask as mask_utils
import submission_masks

from grouped_yolo_cv import (
    collect_fold_results, cv_digest, evaluate_outer, export_cv_submission, export_cv_ensemble,
    fold_records, fuse_fold_instances, make_day_folds,
    run_cv_fold, save_cv_plan, summarize_folds,
)


class GroupedCVTests(unittest.TestCase):
    def setUp(self):
        self.names = [f"202601{day:02d}{station}.jpeg" for day in range(1, 21)
                      for station in ["a", "b"]]
        self.plans = make_day_folds(self.names)
        self.hash = "test-plan"

    def result(self, fold):
        names = self.plans[fold]["splits"]["holdout"]
        return dict(
            fold=fold, status="complete", plan_sha256=self.hash,
            checkpoint_hashes={"detector": "hash"}, selected_parameters={"mode": "refined"},
            outer_mean={"pq": .5},
            outer_rows=[dict(filename=name, annotator_sets=2, pq=.5, dice=.6, iou=.4)
                        for name in names],
        )

    def test_partition_coverage_day_grouping_and_reproducibility(self):
        self.assertEqual(self.plans, make_day_folds(reversed(self.names)))
        outer = []
        for plan in self.plans:
            splits = plan["splits"]
            days = {key: {name[:8] for name in names} for key, names in splits.items()}
            for a, b in [("train", "tune"), ("train", "holdout"), ("tune", "holdout")]:
                self.assertFalse(days[a] & days[b])
            self.assertEqual(set.union(*map(set, splits.values())), set(self.names))
            outer.extend(splits["holdout"])
        self.assertEqual(sorted(outer), sorted(self.names))

    def test_invalid_inputs_fail(self):
        for names in [self.names + [self.names[0]], ["invalid.jpeg"] * 20,
                      [f"20260101{i}.jpeg" for i in range(20)]]:
            with self.assertRaises(ValueError):
                make_day_folds(names)

    def test_annotator_sets_follow_physical_image(self):
        records = {name: [{"id": name + "_a"}, {"id": name + "_b"}] for name in self.names}
        selected = fold_records(self.plans[0], records)
        for phase in selected:
            self.assertEqual(len(selected[phase]), 2 * len(self.plans[0]["splits"][phase]))

    def test_partial_never_reports_full_metrics_and_complete_aggregates(self):
        summary = summarize_folds([self.result(0)], self.plans, self.hash)
        self.assertFalse(summary["complete"])
        self.assertNotIn("fold_mean_pq", summary)
        summary = summarize_folds([self.result(i) for i in range(5)], self.plans, self.hash)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["physical_images"], 40)
        self.assertEqual(summary["pooled_oof_mean_pq"], .5)
        self.assertEqual(summary["fold_sample_sd_pq"], 0)
        broken = self.result(0)
        broken["outer_rows"][0]["filename"] = self.plans[0]["splits"]["train"][0]
        with self.assertRaises(ValueError):
            summarize_folds([broken], self.plans, self.hash)

    def test_plan_and_collection_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cv_plan.json"
            save_cv_plan(path, {"a": 1})
            save_cv_plan(path, {"a": 1})
            with self.assertRaises(ValueError):
                save_cv_plan(path, {"a": 2})
            for folder in ["one", "copy"]:
                p = root / folder
                p.mkdir()
                (p / "fold_result.json").write_text(json.dumps(self.result(0)))
            self.assertEqual(len(collect_fold_results([root], self.plans, self.hash)), 1)
            bad = self.result(0)
            bad["outer_mean"]["pq"] = .9
            (root / "copy/fold_result.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                collect_fold_results([root], self.plans, self.hash)

    def test_outer_averages_annotators_before_images(self):
        ns = dict(
            tqdm=lambda names, **kwargs: names,
            v4_proposals=lambda *args: [], v4_instances=lambda *args: ([], 0),
            validate_disjoint=lambda *args: None, FULL_SHAPE=(8, 8),
            train_lookup={"a": "a", "b": "b"},
            records_by_name={"a": [{"id": 0}, {"id": 1}], "b": [{"id": 1}]},
            gt_rles=lambda identity: identity,
            local_scores=lambda predicted, truth: {"pq": float(truth), "dice": 0., "iou": 0.},
        )
        mean, rows = evaluate_outer(["a", "b"], {"confidence": .25}, ns, None, None)
        self.assertEqual(mean["pq"], .75)
        self.assertEqual(rows[0]["annotator_sets"], 2)

    def test_orchestrator_isolation_reuse_and_incomplete_evidence(self):
        shared = {"smoke_test": True}
        records = {name: [{"id": name}] for name in self.names}
        plan = self.plans[0]
        calls = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()

        ns = dict(
            CFG={"seed": 2026}, records_by_name=records, annotation_sha256="hash",
            gc=gc, torch=FakeTorch(), tqdm=lambda names, **kwargs: names,
            FULL_SHAPE=(8, 8), train_lookup={name: name for name in self.names},
            write_json=lambda path, value: path.write_text(json.dumps(value)),
            gt_rles=lambda identity: [], local_scores=lambda *args: {"pq": .5, "dice": .5, "iou": .5},
            v4_instances=lambda *args: ([], 0), validate_disjoint=lambda *args: None,
        )

        def detector():
            calls.append("train")
            self.assertEqual(ns["split_names"], plan["splits"])
            self.assertTrue(ns["RUN_YOLO_TRAINING"])
            self.assertTrue(ns["RUN_REFINER_TRAINING"])
            self.assertTrue(ns["RUN_V4_CALIBRATION"])
            return object()

        def calibration(*args):
            calls.append("calibrate")
            return dict(
                hashes={"yolo": "hash"}, selected=dict(
                    parameters={"confidence": .25}, scores={"pq": .6},
                ),
            )

        def proposal(name, *args):
            self.assertIn(name, plan["splits"]["holdout"])
            self.assertEqual(calls, ["train", "calibrate"])
            return []

        ns.update(v4_train_yolo=detector, v4_train_refiner=lambda: object(),
                  v4_calibrate=calibration, v4_proposals=proposal)
        with TemporaryDirectory() as directory:
            result = run_cv_fold(ns, plan, shared, directory)
            self.assertEqual(result["plan_sha256"], cv_digest(shared))
            self.assertEqual(result, run_cv_fold(ns, plan, shared, directory))
            self.assertEqual(calls, ["train", "calibrate"])
        with TemporaryDirectory() as directory:
            root = Path(directory) / "fold_0"
            root.mkdir()
            (root / "refiner_best.pt").touch()
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
                run_cv_fold(ns, plan, shared, directory)

    def test_predeclared_submission_reload_never_trains_or_selects_outer_score(self):
        shared = {"smoke_test": False}
        plan = self.plans[0]
        result = self.result(0)
        result["plan_sha256"] = cv_digest(shared)

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()

        ns = dict(records_by_name={name: [{"id": name}] for name in self.names},
                  gc=gc, torch=FakeTorch())

        def detector():
            self.assertFalse(ns["RUN_YOLO_TRAINING"])
            self.assertFalse(ns["RUN_REFINER_TRAINING"])
            self.assertFalse(ns["RUN_V4_CALIBRATION"])
            return object()

        ns.update(v4_train_yolo=detector, v4_train_refiner=lambda: object(),
                  v4_calibrate=lambda *args: {"hashes": result["checkpoint_hashes"]},
                  v4_export=lambda *args: "validated.csv")
        with TemporaryDirectory() as directory:
            root = Path(directory) / "fold_0"
            root.mkdir()
            (root / "fold_result.json").write_text(json.dumps(result))
            self.assertEqual(export_cv_submission(ns, plan, shared, directory), "validated.csv")
            ns["v4_calibrate"] = lambda *args: {"hashes": {}}
            with self.assertRaisesRegex(ValueError, "Submission checkpoints"):
                export_cv_submission(ns, plan, shared, directory)

    def test_notebook_embeds_shared_code_and_submission_export(self):
        root = Path(__file__).resolve().parents[1]
        n = json.loads((root / "Notebooks/solar_filament_CV_V5_grouped.ipynb").read_text())
        cells = {c["id"]: "".join(c["source"]) for c in n["cells"]}
        self.assertTrue(cells["cv-helpers"].startswith((root / "Python Files/grouped_yolo_cv.py").read_text()))
        self.assertEqual(cells["cv-budget"], (root / "Python Files/v5_budget.py").read_text())
        self.assertIn("FOLDS_TO_RUN = [0, 1, 2, 3, 4]", cells["cv-config"])
        self.assertNotIn("SUBMISSION_FOLD", cells["cv-config"])
        self.assertNotIn("v4-export", cells)
        self.assertNotIn("v4-yolo-train", cells)
        self.assertNotIn("split_names =", cells["cv-data"])
        self.assertIn('if not cv_summary["complete"]', cells["cv-export"])
        self.assertIn("export_cv_ensemble(", cells["cv-export"])
        self.assertIn("ensemble_policy=ENSEMBLE_POLICY", cells["cv-plan"])
        self.assertIn("def cv_detector_hours():", cells["cv-helpers"])
        self.assertIn("yolo_epochs=5", cells["v4-workflow"])
        self.assertIn("refiner_epochs=2", cells["v4-workflow"])
        self.assertIn('refiner_encoder="resnet34"', cells["v4-workflow"])
        self.assertIn('encoder_name=V4["refiner_encoder"]', cells["v4-workflow"])
        self.assertIn("time=None", cells["v4-workflow"])
        self.assertIn('detector.add_callback("on_train_batch_end", stop_at_budget)', cells["v4-workflow"])
        self.assertIn("deadline = CV_FOLD_DEADLINE", cells["v4-workflow"])
        workflow = ast.parse(cells["v4-workflow"])
        callback = next(node for node in ast.walk(workflow)
                        if isinstance(node, ast.FunctionDef) and node.name == "stop_at_budget")
        class FakeTime:
            now = 0

            @staticmethod
            def monotonic():
                return FakeTime.now

        class Trainer:
            stop = False

        state = {"stopped": False}
        ns = dict(time=FakeTime, detector_deadline=10, budget_status=state)
        exec(compile(ast.Module(body=[callback], type_ignores=[]), "callback", "exec"), ns)
        trainer = Trainer()
        ns["stop_at_budget"](trainer)
        self.assertFalse(trainer.stop)
        FakeTime.now = 10
        ns["stop_at_budget"](trainer)
        self.assertTrue(trainer.stop)
        self.assertTrue(state["stopped"])

    @staticmethod
    def rle(y0, y1, x0, x1):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        return mask_utils.encode(np.asfortranarray(mask))

    def test_instance_matching_majority_and_no_duplicate_fold_votes(self):
        a = self.rle(0, 6, 0, 6)
        b = self.rle(9, 15, 9, 15)
        folds = [[a, b], [b, a], [a, b], [], []]
        fused = fuse_fold_instances(folds, (16, 16))
        self.assertEqual(len(fused), 2)
        submission_masks.validate_disjoint(fused, (16, 16))
        self.assertEqual(fuse_fold_instances([[a, a], [a], [], [], []], (16, 16)), [])
        self.assertEqual(fuse_fold_instances([[a], [a], [], [], []], (16, 16)), [])
        self.assertEqual(len(fuse_fold_instances([[a], [a], [], [], [a]], (16, 16))), 1)
        with self.assertRaises(ValueError):
            fuse_fold_instances([[a]] * 4, (16, 16))

    def test_pixel_majority_is_not_union_or_fold_zero(self):
        a = self.rle(0, 8, 0, 8)
        shifted = self.rle(0, 8, 2, 10)
        fused = fuse_fold_instances([[a], [a], [shifted], [shifted], [shifted]], (16, 16))
        self.assertEqual(len(fused), 1)
        np.testing.assert_array_equal(mask_utils.decode(fused[0]), mask_utils.decode(shifted))

    def test_all_five_models_export_validated_csv_and_fail_on_provenance(self):
        from types import SimpleNamespace
        import csv

        calls = []
        a = self.rle(0, 6, 0, 6)
        b = self.rle(9, 15, 9, 15)
        shared = {"smoke_test": False}
        ns = dict(
            SMOKE_TEST=False, FULL_SHAPE=(16, 16), gc=gc,
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            records_by_name={name: [{"id": name}] for name in self.names},
            test_paths=[Path("test_image.jpg"), Path("zero.jpg")],
            tqdm=lambda items, **kwargs: items,
            write_json=lambda path, value: path.write_text(json.dumps(value)),
            validate_disjoint=submission_masks.validate_disjoint,
            disjoint_instances=submission_masks.disjoint_instances,
            validate_submission_rows=submission_masks.validate_submission_rows,
        )

        def detector():
            self.assertFalse(ns["RUN_YOLO_TRAINING"])
            self.assertFalse(ns["RUN_REFINER_TRAINING"])
            self.assertFalse(ns["RUN_V4_CALIBRATION"])
            fold = int(ns["OUTPUT_DIR"].name[-1])
            calls.append(fold)
            return fold

        def calibrate(*args):
            result = json.loads((ns["OUTPUT_DIR"] / "fold_result.json").read_text())
            return dict(hashes=result["checkpoint_hashes"],
                        selected=dict(parameters=result["selected_parameters"]))

        ns.update(v4_train_yolo=detector, v4_train_refiner=lambda: None,
                  v4_calibrate=calibrate,
                  v4_proposals=lambda path, fold, *args: (
                      [] if path.stem == "zero" else ([a] if fold < 2 else [b])),
                  v4_instances=lambda proposals, parameters: (proposals, 0))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for fold in range(5):
                result = self.result(fold)
                result.update(plan_sha256=cv_digest(shared), smoke_test=False,
                              selected_parameters=dict(confidence=.25))
                folder = root / f"fold_{fold}"
                folder.mkdir()
                (folder / "fold_result.json").write_text(json.dumps(result))
            destination = export_cv_ensemble(ns, self.plans, shared, root)
            self.assertEqual(calls, list(range(5)))
            with destination.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["filament_id"], "test_image_1")
            self.assertEqual(rows[0]["segmentation_rle"], b["counts"].decode("ascii"))
            audit = json.loads((root / "ensemble_audit.json").read_text())
            self.assertEqual(len(audit["folds"]), 5)
            self.assertEqual(audit["validation"]["overlapping_pixels"], 0)
            self.assertEqual(audit["validation"]["zero_detection_images"], ["zero"])
            ns["v4_calibrate"] = lambda *args: dict(
                hashes={}, selected=dict(parameters=dict(confidence=.25)))
            with self.assertRaisesRegex(ValueError, "checkpoints/calibration"):
                export_cv_ensemble(ns, self.plans, shared, root)
            (root / "fold_4/fold_result.json").unlink()
            with self.assertRaises(FileNotFoundError):
                export_cv_ensemble(ns, self.plans, shared, root)
            with self.assertRaisesRegex(ValueError, "Smoke"):
                export_cv_ensemble(ns, self.plans, {"smoke_test": True}, root)


if __name__ == "__main__":
    unittest.main()
