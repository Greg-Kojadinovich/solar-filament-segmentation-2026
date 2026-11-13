import unittest
from unittest.mock import patch

from v5_budget import remaining_training_seconds


class V5BudgetTests(unittest.TestCase):
    def test_reserve_and_elapsed_time(self):
        self.assertEqual(remaining_training_seconds(100, now=100), 6000)
        self.assertEqual(remaining_training_seconds(100, now=1100), 5000)
        self.assertEqual(remaining_training_seconds(100, now=10000), 0)
        with self.assertRaises(ValueError):
            remaining_training_seconds(0, total_seconds=10, reserve_seconds=10)

    def test_real_clock_and_exhaustion(self):
        with patch("v5_budget.time.monotonic", return_value=0):
            self.assertEqual(remaining_training_seconds(0), 6000)
        with patch("v5_budget.time.monotonic", return_value=5900):
            self.assertEqual(remaining_training_seconds(0), 100)
        with patch("v5_budget.time.monotonic", return_value=6000):
            self.assertEqual(remaining_training_seconds(0), 0)


if __name__ == "__main__":
    unittest.main()
