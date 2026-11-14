import csv
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
from sklearn.metrics import f1_score, jaccard_score


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Research" / "Academic Paper"
spec = importlib.util.spec_from_file_location("plot_results_review", PAPER / "plot_results_review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ResultsReviewTests(unittest.TestCase):
    def test_weighted_scores_match_explicit_pixel_metrics(self):
        first = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        second = np.array([0, 1, 1, 1, 0, 0, 0, 0])
        result = review.foreground_agreement(int(first.sum()), int(second.sum()),
                                             int((first & second).sum()), len(first))
        self.assertAlmostEqual(result["f1"], f1_score(first, second))
        self.assertAlmostEqual(result["jaccard"], jaccard_score(first, second))
        self.assertIsNone(review.foreground_agreement(0, 0, 0, 8))
        self.assertEqual(review.foreground_agreement(1, 0, 0, 8),
                         {"f1": 0.0, "jaccard": 0.0})
        with self.assertRaises(ValueError):
            review.foreground_agreement(1, 2, 2, 8)

    def test_rle_intersection_matches_binary_pixels(self):
        first = np.zeros(review.SHAPE, dtype=np.uint8, order="F")
        second = np.zeros(review.SHAPE, dtype=np.uint8, order="F")
        first[1:4, 2:5] = 1
        second[2:5, 3:6] = 1
        encoded = [mask_utils.encode(first), mask_utils.encode(second)]
        intersection = int(mask_utils.area(mask_utils.merge(encoded, intersect=True)))
        self.assertEqual(intersection, int(np.count_nonzero(first & second)))

    def test_generated_metrics_agree_with_prior_v1_v3_eda(self):
        report = json.loads((PAPER / "results-review/measurements.json").read_text())
        self.assertEqual(report["test_image_count"], 180)
        self.assertEqual(len(report["pairs"]), 6)
        prior = {}
        with (ROOT / "Kaggle Competition/reports/v1-v3-assets/agreement.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["both_empty"] == "False":
                    prior.setdefault(row["pair"], []).append(float(row["union_iou"]))
        for pair in report["pairs"]:
            self.assertEqual((pair["images"], pair["both_empty_excluded"]), (177, 3))
            key = pair["left"] + "-" + pair["right"]
            if key in prior:
                self.assertAlmostEqual(pair["jaccard"], np.mean(prior[key]), places=10)
        self.assertEqual(report["v1_labeled_holdout"]["image_count"], 105)


if __name__ == "__main__":
    unittest.main()
