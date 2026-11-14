import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class V6NotebookTests(unittest.TestCase):
    def test_no_time_limits_preserved_early_stopping_and_ensemble(self):
        notebook = json.loads((ROOT / "solar_filament_CV_V6_grouped.ipynb").read_text())
        cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                if not "".join(cell["source"]).startswith("%"):
                    ast.parse("".join(cell["source"]))
        workflow = ast.parse(cells["v4-workflow"])
        assignment = next(node for node in workflow.body if isinstance(node, ast.Assign)
                          and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id == "V4")
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in assignment.value.keywords}
        self.assertEqual(settings["yolo_epochs"], 25)
        self.assertEqual(settings["refiner_epochs"], 20)
        self.assertEqual(settings["refiner_encoder"], "resnet34")
        self.assertEqual(settings["yolo_patience"], 15)
        self.assertEqual(settings["refiner_patience"], 8)
        self.assertEqual(settings["refiner_min_delta"], 1e-4)
        self.assertNotIn("cv-budget", cells)
        source = "\n".join("".join(c["source"]) for c in notebook["cells"]
                           if c["cell_type"] == "code")
        for removed in ["CV_FOLD_DEADLINE", "CV_TOTAL_SECONDS", "CV_EXPORT_RESERVE",
                        "remaining_training_seconds", "stop_at_budget", "cv_detector_hours",
                        "budget_truncated"]:
            self.assertNotIn(removed, source)
        detector = next(node for node in ast.walk(workflow) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "detector" and node.func.attr == "train")
        arguments = {kw.arg: ast.unparse(kw.value) for kw in detector.keywords}
        self.assertEqual(arguments["time"], "None")
        self.assertEqual(arguments["patience"], "V4['yolo_patience']")
        refiner = next(node for node in workflow.body if isinstance(node, ast.FunctionDef)
                       and node.name == "v4_train_refiner")
        self.assertEqual(sum(isinstance(node, ast.Break) for node in ast.walk(refiner)), 1)
        self.assertIn('if stale >= V4["refiner_patience"]:', cells["v4-workflow"])
        self.assertIn('if totals["tune"] < significant_best - V4["refiner_min_delta"]:',
                      cells["v4-workflow"])
        self.assertIn("export_cv_ensemble(globals(), cv_plans", cells["cv-export"])
        self.assertEqual(cells["cv-helpers"], (ROOT / "grouped_yolo_cv.py").read_text())
        self.assertIn(hashlib.sha256(cells["v4-workflow"].encode()).hexdigest(), cells["cv-plan"])
        self.assertIn("v6_fivefold_resnet34_no_time_limit_seed2026", cells["cv-config"])
        self.assertIn(repr((ROOT / "v6_crop_runtime.py").read_text()), cells["v6-performance"])
        self.assertIn('parallel_refiner(v4_refiner(REFINER_PRETRAINED), REFINER_GPU_IDS)',
                      cells["v4-workflow"])
        self.assertIn('base_refiner(refiner).state_dict().items()', cells["v4-workflow"])
        self.assertIn("return refine_proposals(", cells["v4-workflow"])
        self.assertIn('workers=LOADER_WORKERS', cells["v4-workflow"])
        self.assertIn('**loader_options(LOADER_WORKERS, DEVICE.type == "cuda")',
                      cells["v4-workflow"])

    def test_ultralytics_validation_stagnation_stops_at_patience(self):
        from ultralytics.utils.torch_utils import EarlyStopping

        stopper = EarlyStopping(patience=15)
        self.assertFalse(stopper(1, fitness=.5))
        for epoch in range(2, 16):
            self.assertFalse(stopper(epoch, fitness=.4))
        self.assertTrue(stopper(16, fitness=.4))

    def test_refiner_tuning_loss_stagnation_retains_early_stop(self):
        notebook = json.loads((ROOT / "solar_filament_CV_V6_grouped.ipynb").read_text())
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["id"] == "v4-workflow")
        tree = ast.parse(source)
        refiner = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "v4_train_refiner")
        progress = next(node for node in ast.walk(refiner) if isinstance(node, ast.If)
                        and ast.unparse(node.test).startswith("totals['tune'] < significant_best"))
        stopping = next(node for node in ast.walk(refiner) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "stale >= V4['refiner_patience']")
        loop = ast.parse("for epoch in range(20):\n    completed.append(epoch + 1)\n").body[0]
        loop.body.extend([progress, stopping])
        code = compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])),
                       "refiner-stopping", "exec")
        ns = dict(totals={"tune": .5}, significant_best=float("inf"), stale=0, completed=[],
                  V4=dict(refiner_patience=8, refiner_min_delta=1e-4))
        exec(code, ns)
        self.assertEqual(len(ns["completed"]), 9)
        self.assertEqual(ns["status"], "early_stopped")


if __name__ == "__main__":
    unittest.main()
