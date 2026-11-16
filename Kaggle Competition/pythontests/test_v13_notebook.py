import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Notebooks" / "solar_filament_CV_V14_maskrcnn_refiner.ipynb"


class V13NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cells = json.loads(NOTEBOOK.read_text())["cells"]
        cls.cells = {c["id"]: "".join(c["source"]) for c in cells}
        cls.code = {c["id"]: "".join(c["source"]) for c in cells if c["cell_type"] == "code"}

    def test_code_cells_parse(self):
        for name, source in self.code.items():
            if not source.lstrip().startswith("%"):
                ast.parse(source, name)

    def test_epochs_caps_and_recipe_fallback(self):
        config = self.cells["v13-config"]
        self.assertIn("detector_epochs=15", config)
        self.assertIn("refiner_epochs=10", config)
        self.assertIn("max_candidates=10", config)
        self.assertIn("caps=[7, 8, 10, None]", config)
        self.assertIn("min_scores=[0.2, 0.5, 0.7]", config)
        self.assertIn("recipe_margin=0.01", config)
        self.assertIn('select_metric="blend"', config)

    def test_embedded_library_matches_the_files(self):
        data = self.cells["v13-data"]
        namespace = {}
        exec(data.split("\nmanifest = ")[0].split("\n(WORK_DIR")[0].split("\ncandidates = ")[0], namespace)
        self.assertEqual(namespace["LIB_SOURCE"], (ROOT / "Python Files" / "v13_lib.py").read_text())
        self.assertEqual(namespace["TRAIN_SOURCE"], (ROOT / "Python Files" / "v13_train_detector.py").read_text())

    def test_notebook_never_uploads(self):
        for source in self.cells.values():
            self.assertNotIn("competitions submit -c", source.replace('print("Submit with: kaggle competitions submit -c"', ""))
            self.assertNotIn("subprocess.run(['kaggle", source)


if __name__ == "__main__":
    unittest.main()
