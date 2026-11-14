import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class V7NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads((ROOT / "Notebooks/solar_filament_CV_V7_yolo_only.ipynb").read_text())
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in self.notebook["cells"]}

    def test_notebook_code_is_valid_and_has_no_saved_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                self.assertEqual(cell["outputs"], [])
                if not source.startswith("%"):
                    ast.parse(source)

    def test_five_fold_yolo_only_epoch_bounded_training(self):
        config = ast.parse(self.cells["v7-config"])
        assignment = next(node for node in config.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "V7"
                                  for target in node.targets))
        settings = {keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in assignment.value.keywords}
        self.assertEqual(settings["epochs"], 10)
        self.assertEqual(settings["patience"], 10)
        self.assertEqual(settings["batch_size"], 10)
        self.assertEqual(settings["inference_batch"], 8)
        config_source = self.cells["v7-config"]
        self.assertIn("YOLO_TRAIN_DEVICE = [0, 1]", config_source)
        self.assertIn("YOLO_INFERENCE_DEVICE = 0", config_source)
        self.assertIn("requires two visible GPUs", config_source)
        folds = next(node.value for node in config.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "FOLDS_TO_RUN"
                             for target in node.targets))
        self.assertEqual(ast.literal_eval(folds), [0, 1, 2, 3, 4])
        runtime = (ROOT / "Python Files/v7_yolo_runtime.py").read_text()
        tree = ast.parse(runtime)
        train = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "train"
                     and any(keyword.arg == "epochs" for keyword in node.keywords))
        arguments = {kw.arg: ast.unparse(kw.value) for kw in train.keywords}
        self.assertEqual(arguments["epochs"], "settings['epochs']")
        self.assertEqual(arguments["time"], "None")
        for removed_component in ["CachedInstanceCrops", "refine_proposals",
                                  "v4_train_refiner", "refiner_best.pt"]:
            self.assertNotIn(removed_component, runtime)
        self.assertIn(repr(runtime), self.cells["v7-runtime"])

    def test_verified_csv_is_submitted_through_kaggle_cli(self):
        submit = ast.parse(self.cells["v7-submit"])
        calls = [node for node in ast.walk(submit) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "run"]
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(isinstance(node, ast.Name) and node.id == "submission_path"
                            for node in ast.walk(submit)))
        self.assertTrue(any(isinstance(node, ast.Name) and node.id == "cv_summary"
                            for node in ast.walk(submit)))
        self.assertIn('"competitions", "submit"', self.cells["v7-submit"])
        self.assertIn('get_secret("KAGGLE_USERNAME")', self.cells["v7-submit"])
        self.assertIn('get_secret("KAGGLE_KEY")', self.cells["v7-submit"])
        self.assertIn("check=True", self.cells["v7-submit"])


if __name__ == "__main__":
    unittest.main()
