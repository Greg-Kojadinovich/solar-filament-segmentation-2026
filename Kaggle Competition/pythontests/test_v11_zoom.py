import unittest

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image

from v8_runtime import (
    MODES, build_instances, build_refiner, candidate_parameters, jitter_batch, pass_mask,
    refine_image, refiner_channels, rotate_batch, simulate_coarse, zoom_pass,
)

SIZE = 256


class Echo(torch.nn.Module):
    """A stand-in refiner whose output is its fourth channel (the mask it was given)."""

    def forward(self, x):
        return (x[:, 3:4] * 2 - 1) * 12


def blob_batch(count=6, size=64):
    y = torch.zeros(count, 1, size, size)
    for index in range(count):
        y[index, :, 20:44, 28 + index:36 + index] = 1
        y[index, :, 36:44, 20:50] = 1
    return y


class MaskChannelTests(unittest.TestCase):
    def test_channels_follow_the_setting(self):
        self.assertEqual(refiner_channels({}), 3)
        self.assertEqual(refiner_channels({"refiner_mask_channel": True}), 4)
        model = build_refiner(dict(refiner_encoder="resnet18", refiner_mask_channel=True), False)
        self.assertEqual(tuple(model(torch.zeros(1, 4, 64, 64)).shape), (1, 1, 64, 64))

    def test_simulated_detector_masks_are_binary_nonempty_and_overlap_the_truth(self):
        y = blob_batch(12)
        torch.manual_seed(0)
        coarse = simulate_coarse(y)
        self.assertEqual(coarse.shape, y.shape)
        self.assertTrue(torch.all((coarse == 0) | (coarse == 1)))
        self.assertTrue(torch.all(coarse.sum(dim=(1, 2, 3)) >= 4))
        inter = (coarse * y).sum(dim=(1, 2, 3))
        union = ((coarse + y) > 0).float().sum(dim=(1, 2, 3))
        iou = inter / union
        self.assertGreater(iou.mean().item(), 0.3)
        self.assertLess((iou == 1).float().mean().item(), 0.5)  # they are genuinely imperfect

    def test_a_fixed_generator_repeats_exactly(self):
        y = blob_batch(5)
        first = simulate_coarse(y, torch.Generator().manual_seed(7))
        second = simulate_coarse(y, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, simulate_coarse(y, torch.Generator().manual_seed(8))))

    def test_zoom_and_shift_keep_image_and_mask_registered(self):
        y = blob_batch(4)
        x = (y * 2 - 1).repeat(1, 3, 1, 1)
        rx, ry = rotate_batch(x, y, angles=torch.zeros(4), scales=torch.full((4,), 0.8),
                              shifts=torch.zeros(4, 2))
        self.assertGreater(ry.sum().item(), y.sum().item())  # zooming in enlarges the object
        agreement = ((rx[:, 0:1] > 0).float() == ry).float().mean().item()
        self.assertGreater(agreement, 0.97)
        jx, jy = jitter_batch(x, y)
        self.assertEqual(jx.shape, x.shape)
        self.assertGreater(((jx[:, 0:1] > 0).float() == jy).float().mean().item(), 0.95)


class ZoomPassTests(unittest.TestCase):
    def setUp(self):
        self.shape = (SIZE, SIZE)
        self.mask = np.zeros(self.shape, np.uint8)
        self.mask[100:140, 120:134] = 1
        self.image = Image.new("RGB", self.shape, 90)
        self.settings = dict(crop_size=32, minimum_crop=48, refiner_thresholds=[0.5, 0.65],
                             refiner_inference_batch=4, refiner_zoom_context=1.4)

    def proposals(self):
        return refine_image(self.image, self.mask[None], Echo(), "cpu", self.shape, batch_size=2,
                            crop_size=32, minimum=48, mask_channel=True)

    def test_first_pass_returns_the_given_mask_and_the_zoom_window_is_tighter(self):
        proposals = zoom_pass(self.image, self.proposals(), Echo(), "cpu", self.shape, self.settings)
        first = pass_mask(proposals[0], 0.5, self.shape)
        iou = (first.astype(bool) & self.mask.astype(bool)).sum() / (first.astype(bool) | self.mask.astype(bool)).sum()
        self.assertGreater(iou, 0.9)
        wide = proposals[0]["bounds"]
        tight = proposals[0]["zoom"][0.5]["bounds"]
        self.assertLess(tight[2] - tight[0], wide[2] - wide[0])
        self.assertTrue(set(proposals[0]["zoom"]) == {0.5, 0.65})

    def test_zoom_mode_reapplies_the_mask_in_full_frame_coordinates(self):
        proposals = zoom_pass(self.image, self.proposals(), Echo(), "cpu", self.shape, self.settings)
        parameters = dict(mode="zoom", confidence=0.25, refiner_threshold=0.5, gate_pixels=None,
                          min_area=20)
        (instance,), fallbacks = build_instances(proposals, parameters, self.shape)
        result = mask_utils.decode(instance).astype(bool)
        truth = self.mask.astype(bool)
        self.assertGreater((result & truth).sum() / (result | truth).sum(), 0.85)
        self.assertEqual(fallbacks, 0)

    def test_zoom_candidates_exist_only_when_enabled(self):
        base = dict(min_areas=[64], refiner_thresholds=[0.5, 0.65], gate_pixels=[12])
        self.assertNotIn("zoom", {c["mode"] for c in candidate_parameters(base, 0.25)})
        enabled = {c["mode"] for c in candidate_parameters(dict(base, refiner_zoom=True), 0.25)}
        self.assertEqual(enabled, set(MODES))


if __name__ == "__main__":
    unittest.main()


class SinglePieceTests(unittest.TestCase):
    def test_largest_component_drops_stray_pieces(self):
        from v8_runtime import largest_component

        mask = np.zeros((40, 40), bool)
        mask[5:25, 5:12] = True
        mask[30:33, 30:33] = True
        result = largest_component(mask)
        self.assertEqual(int(result.sum()), 20 * 7)
        self.assertFalse(result[30:33, 30:33].any())

    def test_build_instances_keeps_one_piece_only_when_asked(self):
        shape = (96, 96)
        coarse = np.zeros(shape, np.uint8)
        coarse[20:60, 30:40] = 1
        probability = np.zeros((32, 32), np.float32)
        probability[2:26, 10:16] = 0.9
        probability[26:30, 26:30] = 0.9  # a stray blob far from the filament
        proposal = dict(coarse=mask_utils.encode(np.asfortranarray(coarse)), bounds=(0, 0, 96, 96),
                        probability=probability)
        base = dict(mode="refined", confidence=0.25, refiner_threshold=0.5, gate_pixels=None, min_area=20)
        (loose,), _ = build_instances([proposal], base, shape)
        (single,), _ = build_instances([proposal], dict(base, single_piece=True), shape)
        self.assertGreater(mask_utils.area(loose), mask_utils.area(single))
        pieces = np.bincount(__import__("scipy.ndimage", fromlist=["label"]).label(
            mask_utils.decode(single))[0].ravel())[1:]
        self.assertEqual(len(pieces), 1)

    def test_candidates_carry_the_flag_only_when_enabled(self):
        from v8_runtime import candidate_parameters as grid

        base = dict(min_areas=[64], refiner_thresholds=[0.5], gate_pixels=[8])
        self.assertTrue(all("single_piece" not in c for c in grid(base, 0.25)))
        self.assertTrue(all(c["single_piece"] for c in grid(dict(base, single_piece=True), 0.25)))
