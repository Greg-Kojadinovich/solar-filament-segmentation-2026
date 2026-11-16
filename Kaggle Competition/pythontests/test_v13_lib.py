import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Python Files"))
import v13_lib as lib  # noqa: E402


def rect(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, np.uint8)
    mask[y0:y1, x0:x1] = 1
    return lib.encode_mask(mask)


class StubDetector:
    """Segments every bright blob of the input, so flipped inputs give flipped masks."""

    def __init__(self):
        self.roi_heads = SimpleNamespace(score_thresh=0.0, detections_per_img=100)

    def __call__(self, images):
        gray = images[0][0].numpy()
        count, labels = cv2.connectedComponents((gray > 0.5).astype(np.uint8))
        masks = [labels == i for i in range(1, count)][: self.roi_heads.detections_per_img]
        scores = [0.9 - 0.1 * i for i in range(len(masks))]
        masks = torch.from_numpy(np.stack(masks)).float().unsqueeze(1) if masks else torch.zeros((0, 1, *gray.shape))
        return [dict(masks=masks, scores=torch.tensor(scores, dtype=torch.float32))]


class BrightRefiner(torch.nn.Module):
    def forward(self, x):
        return (x[:, :1] - 0.5) * 20


class V13LibraryTests(unittest.TestCase):
    def setUp(self):
        self.old_size = lib.IMAGE_SIZE
        lib.IMAGE_SIZE = 256

    def tearDown(self):
        lib.IMAGE_SIZE = self.old_size

    def test_overlapping_instances_merge_and_separate_ones_stay(self):
        shape = (100, 100)
        merged = lib.merge_overlapping_rles([rect(shape, 0, 20, 0, 20), rect(shape, 10, 30, 10, 30),
                                             rect(shape, 60, 80, 60, 80)])
        self.assertEqual(sorted(int(mask_utils.area(r)) for r in merged), [400, 700])

    def test_scores_for_perfect_empty_and_shifted_predictions(self):
        shape = (200, 200)
        truth = [rect(shape, 10, 30, 10, 60), rect(shape, 100, 120, 100, 160)]
        self.assertEqual(lib.instance_scores(truth, truth)["pq"], 1.0)
        self.assertEqual(lib.instance_scores([], truth)["soft_dice"], 0.0)
        self.assertEqual(lib.instance_scores([], [])["soft_dice"], 1.0)
        extra = truth + [rect(shape, 150, 160, 10, 30)]
        scores = lib.instance_scores(extra, truth)
        self.assertAlmostEqual(scores["soft_dice"], 2 / 3, places=6)
        self.assertAlmostEqual(scores["pq"], 0.8, places=6)

    def test_flip_views_agree_and_cluster_with_full_support(self):
        image = np.zeros((256, 256), np.float32)
        image[30:60, 20:120] = 1.0
        image[180:200, 150:250] = 1.0
        views = lib.detect_views(StubDetector(), image, torch.device("cpu"), 0.0, 10)
        self.assertEqual([len(v[0]) for v in views], [2, 2, 2, 2])
        clusters = lib.cluster_views(views)
        self.assertEqual(sorted(len(c["members"]) for c in clusters), [4, 4])
        fused = lib.fuse_cluster(clusters[0], (512, 512))
        self.assertEqual(fused.shape, (512, 512))
        self.assertGreater(int(fused.sum()), 2 * 30 * 100 * 4 * 0.9 / 2)

    def test_candidates_cap_and_support_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((1024, 1024), np.uint8)
            for index in range(4):  # far enough apart that no 1.8x crop reaches a neighbour
                frame[40 + 240 * index:80 + 240 * index, 100:200] = 255
            path = Path(directory) / "frame.jpeg"
            cv2.imwrite(str(path), frame)
            settings = dict(candidate_threshold=0.0, max_candidates=10)
            item = lib.image_candidates(StubDetector(), BrightRefiner(), path, torch.device("cpu"), settings)
        self.assertEqual(len(item["candidates"]), 4)
        self.assertTrue(all(c["support"] == 4 for c in item["candidates"]))
        self.assertEqual(len(lib.select_instances(item, 3, 1, 0.5)), 3)
        self.assertEqual(len(lib.select_instances(item, None, 1, 0.5)), 4)
        self.assertEqual(len(lib.select_instances(item, 8, 4, 0.5)), 4)
        self.assertGreaterEqual(len(lib.select_instances(item, 8, 5, 0.5)), 1)  # never leaves a photo empty
        self.assertEqual(len(lib.single_view_rles(item, 0.0)), 4)

    def test_candidate_cap_keeps_the_best_supported_first(self):
        shape = (64, 64)
        base = dict(fallback=rect(shape, 0, 10, 0, 10), bounds=(0, 0, 64, 64), shape=shape,
                    probability=None, refined={})
        item = dict(candidates=[
            dict(base, support=1, score=0.99, fallback=rect(shape, 0, 10, 0, 10)),
            dict(base, support=4, score=0.40, fallback=rect(shape, 20, 30, 20, 30)),
            dict(base, support=3, score=0.90, fallback=rect(shape, 40, 50, 40, 50)),
        ], view0=([], []), shape=shape)
        kept = lib.select_instances(item, 2, 1, 0.5, min_area=1)
        self.assertEqual(sorted(int(mask_utils.toBbox(r)[0]) for r in kept), [20, 40])

    def test_score_floor_and_public_recipe_selection(self):
        shape = (64, 64)

        def candidate(origin, first, mean, y):
            return dict(support=4, score=mean, origin=origin, first_score=first, shape=shape, probability=None,
                        refined={}, bounds=(0, 0, 64, 64), fallback=rect(shape, y, y + 10, 5, 25))

        item = dict(candidates=[candidate(0, 0.9, 0.9, 0), candidate(0, 0.5, 0.5, 20),
                                candidate(1, 0.8, 0.8, 40)], view0=([], []), shape=shape)
        self.assertEqual(len(lib.select_instances(item, None, 1, 0.5, min_area=1)), 3)
        self.assertEqual(len(lib.select_instances(item, None, 1, 0.5, min_area=1, min_score=0.7)), 2)
        recipe = lib.select_instances(item, None, 1, 0.5, min_area=1, min_score=0.7, view0_only=True)
        self.assertEqual(len(recipe), 1)  # only the identity-view candidate at or above 0.7

    def test_cluster_origin_is_the_first_view_that_saw_it(self):
        shape = (64, 64)
        views = [([rect(shape, 0, 10, 0, 10)], [0.9]), ([rect(shape, 0, 10, 0, 10), rect(shape, 30, 40, 30, 40)], [0.8, 0.7])]
        clusters = lib.cluster_views(views)
        self.assertEqual(sorted((c["origin"], len(c["members"])) for c in clusters), [(0, 2), (1, 1)])

    def test_largest_component_and_manifest_split(self):
        mask = np.zeros((40, 40), bool)
        mask[5:25, 5:12] = True
        mask[30:33, 30:33] = True
        self.assertEqual(int(lib.largest_component(mask).sum()), 140)


if __name__ == "__main__":
    unittest.main()
