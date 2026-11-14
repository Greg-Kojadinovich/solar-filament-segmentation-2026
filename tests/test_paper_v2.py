import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "AcademicPaper"
spec = importlib.util.spec_from_file_location("plot_paper_v2", PAPER / "plot_paper_v2.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


class PaperV2Tests(unittest.TestCase):
    def test_scored_provenance_and_pending_result(self):
        data = plots.load_results(PAPER / "paper_v2_results.json")
        self.assertEqual([row["public_pq"] for row in data["runs"]], [.29, .32, .35, .36])
        self.assertEqual([row["submission_id"] for row in data["runs"]],
                         [58301531, 58304132, 58316663, 58327117])
        csv_path = ROOT / "V5_SubmissionFile" / "submission.csv"
        self.assertEqual(hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                         data["runs"][3]["csv_sha256"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            for value in [0, .4, float("nan")]:
                broken = copy.deepcopy(data)
                broken["runs"][3]["status"] = "pending"
                broken["runs"][3]["submission_id"] = None
                broken["runs"][3]["public_pq"] = value
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    plots.load_results(path)
                for index, field, value in [(3, "submission_id", None),
                                            (0, "public_pq", -.1),
                                            (0, "public_pq", 1.1),
                                            (0, "submission_id", None),
                                            (0, "status", "estimated")]:
                    with self.subTest(index=index, field=field, value=value):
                        broken = copy.deepcopy(data)
                        broken["runs"][index][field] = value
                        path.write_text(json.dumps(broken))
                        with self.assertRaises(ValueError):
                            plots.load_results(path)
            broken = copy.deepcopy(data)
            broken["runs"][0]["public_pq"] = float("nan")
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                plots.load_results(path)

    def test_score_plot_does_not_draw_pending_as_zero(self):
        plots.setup_style()
        data = plots.load_results(PAPER / "paper_v2_results.json")
        for pending in [False, True]:
            with self.subTest(pending=pending):
                scenario = copy.deepcopy(data)
                if pending:
                    scenario["runs"][3].update(status="pending", public_pq=None,
                                               submission_id=None)
                with patch.object(plots, "save_figure") as save:
                    plots.score_chart(scenario, PAPER)
                fig = save.call_args.args[0]
                ax = fig.axes[0]
                self.assertEqual(
                    [float(container.patches[0].get_height()) for container in ax.containers],
                    [.29, .32, .35] if pending else [.29, .32, .35, .36])
                self.assertEqual(any("PENDING" in text.get_text() for text in ax.texts),
                                 pending)
                plots.plt.close(fig)

    def test_saved_vectors_have_editable_text_and_complete_evidence(self):
        output = PAPER / "figures-v2"
        manifest = json.loads((output / "figure_manifest.json").read_text())
        self.assertEqual(len(manifest["figures"]), 5)
        self.assertEqual(manifest["measurements"],
                         json.loads((PAPER / "paper_v2_results.json").read_text()))
        for relative, expected in manifest["sources"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        for name in manifest["figures"]:
            for extension in ["png", "svg", "pdf"]:
                self.assertGreater((output / f"{name}.{extension}").stat().st_size, 1000)
            svg = ElementTree.parse(output / f"{name}.svg")
            self.assertTrue(svg.findall(".//{http://www.w3.org/2000/svg}text"))
            self.assertTrue((output / f"{name}.pdf").read_bytes().startswith(b"%PDF-"))
            self.assertEqual((output / f"{name}.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        source = (PAPER / "SolarFilament_V2.tex").read_text()
        self.assertIn("esairposter2026", source)
        self.assertIn("The ensemble itself has no learned parameters", source)
        self.assertIn("No random-initialization", source)

    def test_baseline_review_image_is_from_first_draft(self):
        source = (PAPER / "SolarFilament_V2.tex").read_text()
        original = (PAPER / "SolarFilament.tex").read_text()
        image = "../baselinevideo/thumbnail.jpg"
        self.assertIn(image, original)
        self.assertIn(image, source)
        self.assertIn("not an\nindependent performance estimate", source)
        self.assertTrue((PAPER / image).is_file())


if __name__ == "__main__":
    unittest.main()
