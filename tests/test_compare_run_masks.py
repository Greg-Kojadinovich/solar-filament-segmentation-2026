import unittest

import numpy as np
import pycocotools.mask as masks

from compare_run_masks import agreement, describe_instances


class CompareRunMaskTests(unittest.TestCase):
    def test_geometry_and_connectivity(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        mask[6, 6] = 1
        rle = masks.encode(np.asfortranarray(mask))
        union, row, details = describe_instances([rle], (8, 8))
        self.assertEqual(row["foreground_pixels"], 5)
        self.assertEqual(row["disconnected_instances"], 1)
        self.assertEqual(details[0]["components"], 2)
        self.assertEqual(details[0]["bbox_fill"], 5 / 36)
        result = agreement([rle], [rle], union, union)
        self.assertEqual(result["union_iou"], 1)
        self.assertEqual(result["instance_agreement"], 1)

    def test_empty_not_counted_as_perfect_agreement(self):
        empty = np.zeros((8, 8), dtype=bool)
        result = agreement([], [], empty, empty)
        self.assertIsNone(result["union_iou"])
        self.assertIsNone(result["instance_agreement"])
        self.assertTrue(result["both_empty"])

    def test_split_union_same_but_instance_agreement_differs(self):
        full = np.ones((8, 8), dtype=np.uint8)
        left, right = full.copy(), full.copy()
        left[:, 4:] = 0
        right[:, :4] = 0
        first = [masks.encode(np.asfortranarray(full))]
        second = [masks.encode(np.asfortranarray(m)) for m in [left, right]]
        result = agreement(first, second, full.astype(bool), full.astype(bool))
        self.assertEqual(result["union_iou"], 1)
        self.assertEqual(result["instance_agreement"], 0)


if __name__ == "__main__":
    unittest.main()
