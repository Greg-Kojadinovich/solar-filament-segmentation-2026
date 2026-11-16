import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class V10NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(
            (ROOT / "Notebooks/solar_filament_CV_V10_crop_refined.ipynb").read_text())
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]}

    def test_code_is_valid_without_saved_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell["source"])
                if not source.startswith("%"):
                    ast.parse(source)

    def test_medium_model_100_and_50_epochs_with_patience_20(self):
        config = ast.parse(self.cells["v8-config"])
        assignment = next(node for node in config.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "V8" for t in node.targets))
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in assignment.value.keywords}
        self.assertEqual(settings["weights"], "yolo11m-seg.pt")
        self.assertEqual((settings["image_size"], settings["batch_size"], settings["nbs"]),
                         (1280, 8, 32))
        self.assertEqual((settings["epochs"], settings["patience"]), (100, 20))
        self.assertEqual((settings["refiner_epochs"], settings["refiner_patience"]), (50, 20))
        self.assertEqual(settings["gate_pixels"], [6, 12])
        self.assertEqual(settings["refiner_gpus"], [0, 1])

    def test_run_name_changes_with_any_setting(self):
        source = self.cells["v8-config"]
        self.assertIn('VERSION = "V10"', source)
        self.assertIn("SETTINGS_TAG = hashlib.sha1(json.dumps(V8, sort_keys=True)", source)
        self.assertIn("v10_single_", source)
        self.assertIn("submission_V10_disk_normalized.csv", self.cells["v8-final-guide"])

    def test_runtime_embeds_sources_keeps_disk_small_and_never_uploads(self):
        runtime = (ROOT / "Python Files/v8_runtime.py").read_text()
        self.assertIn("save_period=-1", runtime)
        self.assertNotIn("save_period=1,", runtime)
        for name in ["grouped_yolo_cv.py", "submission_masks.py", "solar_disk.py",
                     "v6_crop_runtime.py", "v7_yolo_runtime.py", "v8_runtime.py"]:
            self.assertIn(repr((ROOT / "Python Files" / name).read_text()), self.cells["v8-runtime"])
        joined = "".join(self.cells.values())
        for forbidden in ["kaggle_secrets", "get_secret", '"competitions", "submit"']:
            self.assertNotIn(forbidden, joined)
        self.assertEqual(self.notebook["cells"][-1]["id"], "v8-final-check")
        self.assertIn("under 15 GB free", self.cells["v8-preflight"])


if __name__ == "__main__":
    unittest.main()
