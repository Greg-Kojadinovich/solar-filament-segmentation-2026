import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class V4NotebookTests(unittest.TestCase):
    def test_longer_training_defaults_and_no_deadlines(self):
        notebook = json.loads((ROOT / "solar_filament_CV_V4.ipynb").read_text())
        cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and not source.startswith("%"):
                ast.parse(source)
                self.assertEqual(cell["outputs"], [])
        workflow = ast.parse(cells["v4-workflow"])
        config = next(node for node in workflow.body
                      if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "V4")
        settings = {keyword.arg: ast.literal_eval(keyword.value) for keyword in config.value.keywords}
        self.assertEqual(settings["yolo_epochs"], 120)
        self.assertEqual(settings["refiner_epochs"], 40)
        self.assertIsNone(settings["yolo_hours"])
        self.assertIsNone(settings["refiner_hours"])
        self.assertEqual(settings["yolo_optimizer"], "AdamW")
        self.assertEqual(settings["yolo_lr"], .002)
        self.assertEqual(settings["yolo_warmup_bias_lr"], 0)
        self.assertIn("deadline is not None and time.monotonic() >= deadline", cells["v4-workflow"])
        self.assertIn("else started + V4", cells["v4-workflow"])
        self.assertIn("RUN_NAME = 'v4_", cells["v4-config"])
        self.assertIn("SMOKE_TEST = False", cells["v4-config"])
        self.assertIn("RUN_YOLO_TRAINING = True", cells["v4-workflow"])
        self.assertIn("RUN_REFINER_TRAINING = True", cells["v4-workflow"])
        self.assertIn("'ultralytics==8.4.211'", cells["v4-install"])
        shared = (ROOT / "submission_masks.py").read_text().split('\nif __name__ == "__main__":')[0]
        self.assertEqual(cells["v4-disjoint"], shared)
        calls = [node for node in ast.walk(workflow) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "train"
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "detector"]
        self.assertEqual(len(calls), 1)
        arguments = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
        self.assertEqual(arguments["time"], "V4['yolo_hours']")
        self.assertEqual(arguments["optimizer"], "V4['yolo_optimizer']")
        metadata = json.loads((ROOT / "kaggle-v4/kernel-metadata.json").read_text())
        self.assertEqual(metadata["code_file"], "solar_filament_CV_V4.ipynb")
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["competition_sources"], ["filament-segmentation-2026"])


if __name__ == "__main__":
    unittest.main()
