import unittest

import numpy as np

from solar_disk import fit_disk, normalize_disk

SIZE, CENTER, RADIUS = 512, (258.0, 254.0), 225.0


def synthetic_frame(halo=True, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    r = np.hypot(xx - CENTER[0], yy - CENTER[1])
    disk = 120 - 25 * (r / RADIUS) ** 2 + rng.normal(0, 4, (SIZE, SIZE))
    ring = 120 * np.exp(-(((r - RADIUS) / 1.5) ** 2))
    sky = (55 * np.exp(-(r - RADIUS) / 30) + 8) if halo else np.full_like(r, 8)
    frame = np.where(r <= RADIUS, disk + ring, sky + rng.normal(0, 2, r.shape))
    frame[225:240, 270:330] -= 45  # a dark filament on the disk
    return np.clip(frame, 0, 255).astype(np.uint8), r


class SolarDiskTests(unittest.TestCase):
    def test_fit_recovers_center_and_radius_with_and_without_a_halo(self):
        for halo in (True, False):
            frame, _ = synthetic_frame(halo)
            cx, cy, radius, ok = fit_disk(frame)
            self.assertTrue(ok)
            self.assertAlmostEqual(cx, CENTER[0], delta=2.0)
            self.assertAlmostEqual(cy, CENTER[1], delta=2.0)
            self.assertAlmostEqual(radius, RADIUS, delta=3.0)

    def test_outside_the_rim_is_neutral_and_the_interior_is_normalized(self):
        frame, r = synthetic_frame()
        output, info = normalize_disk(frame, margin=10)
        self.assertEqual(output.shape, frame.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(info["disk_found"])
        self.assertTrue(np.all(output[r > info["radius"] + 11] == 128))
        interior = output[r < 0.9 * RADIUS].astype(float)
        self.assertAlmostEqual(np.median(interior), 128, delta=3)
        self.assertGreater(interior.std(), 20)

    def test_limb_ring_and_halo_step_are_flattened(self):
        frame, r = synthetic_frame()
        output, _ = normalize_disk(frame, margin=10)
        interior = np.abs(output[(r > 0.5 * RADIUS) & (r < 0.9 * RADIUS)].astype(float) - 128).mean()
        for low, high in [(-6, -1), (-1, 3), (3, 9)]:
            band = (r > RADIUS + low) & (r <= RADIUS + high)
            self.assertLess(np.abs(output[band].astype(float) - 128).mean(), 2.5 * interior)

    def test_filament_stays_darker_than_its_surroundings(self):
        frame, _ = synthetic_frame()
        output, _ = normalize_disk(frame)
        filament = output[225:240, 270:330].astype(float).mean()
        around = output[200:224, 270:330].astype(float).mean()
        self.assertLess(filament, around - 30)

    def test_frame_without_a_disk_falls_back_without_failing(self):
        noise = np.random.default_rng(1).integers(0, 40, (SIZE, SIZE)).astype(np.uint8)
        output, info = normalize_disk(noise)
        self.assertFalse(info["disk_found"])
        self.assertEqual(output.shape, noise.shape)


if __name__ == "__main__":
    unittest.main()
