import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class V9NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(
            (ROOT / "Notebooks/solar_filament_CV_V9_crop_refined.ipynb").read_text())
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]}

    def test_code_is_valid_without_saved_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell["source"])
                if not source.startswith("%"):
                    ast.parse(source)

    def test_forty_epochs_with_early_stopping_for_both_models(self):
        config = ast.parse(self.cells["v8-config"])
        assignment = next(node for node in config.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "V8" for t in node.targets))
        settings = {kw.arg: ast.literal_eval(kw.value) for kw in assignment.value.keywords}
        self.assertEqual((settings["epochs"], settings["patience"]), (40, 20))
        self.assertEqual((settings["refiner_epochs"], settings["refiner_patience"]), (40, 20))
        # 1536 px with 8 images per GPU ran out of memory on T4s; 1280 px with 6 per GPU fits.
        self.assertEqual((settings["image_size"], settings["batch_size"], settings["nbs"],
                          settings["refiner_batch"]), (1280, 12, 32, 32))

    def test_version_and_run_name_do_not_collide_with_v8(self):
        source = self.cells["v8-config"]
        self.assertIn('VERSION = "V9"', source)
        self.assertIn('v9_single_', source)
        self.assertIn("_40ep_img{V8['image_size']}", source)
        self.assertNotIn("_20ep_", source)
        self.assertIn("submission_V9_disk_normalized.csv", self.cells["v8-final-guide"])

    def test_embedded_runtime_matches_sources_and_nothing_uploads(self):
        for name in ["grouped_yolo_cv.py", "submission_masks.py", "solar_disk.py",
                     "v6_crop_runtime.py", "v7_yolo_runtime.py", "v8_runtime.py"]:
            self.assertIn(repr((ROOT / "Python Files" / name).read_text()), self.cells["v8-runtime"])
        joined = "".join(self.cells.values())
        for forbidden in ["kaggle_secrets", "get_secret", '"competitions", "submit"']:
            self.assertNotIn(forbidden, joined)
        self.assertEqual(self.notebook["cells"][-1]["id"], "v8-final-check")


if __name__ == "__main__":
    unittest.main()
