import unittest
import sys
import os
import math
from typing import List, Tuple

# ==========================================
# MOCK NUMPY
# ==========================================
# Since the environment lacks numpy, we mock it to test logic that depends on it.

class MockArray(list):
    def __init__(self, data):
        super().__init__(data)
        # Handle shape
        if isinstance(data, list):
            self.shape = (len(data),)
        else:
            self.shape = ()

    def fill(self, val):
        for i in range(len(self)):
            self[i] = val

    def __getitem__(self, item):
        if isinstance(item, slice):
            return MockArray(super().__getitem__(item))
        return super().__getitem__(item)

    def __setitem__(self, key, value):
         super().__setitem__(key, value)

    def __gt__(self, other):
        # element-wise greater than, returns MockArray of bools
        return MockArray([x > other for x in self])

    def __lt__(self, other):
        return MockArray([x < other for x in self])

    def __le__(self, other):
        return MockArray([x <= other for x in self])

    def __ge__(self, other):
        return MockArray([x >= other for x in self])

    # Add other operators as needed by LidarEngine or SwarmCore imports

class MockNumpy:
    class ndarray:
        pass # for type checking

    float64 = float

    def zeros(self, shape, dtype=None):
        if isinstance(shape, int):
            return MockArray([0.0] * shape)
        elif isinstance(shape, tuple):
             # minimal support for multidimensional zeros if needed
             size = 1
             for dim in shape: size *= dim
             return MockArray([0.0] * size) # flattens it for now
        return MockArray([0.0] * shape[0])

    def sum(self, arr, axis=None):
        # sum handles booleans correctly (True=1)
        return sum(arr)

    def clip(self, a, a_min, a_max):
        if isinstance(a, list):
             return MockArray([min(max(x, a_min), a_max) for x in a])
        return min(max(a, a_min), a_max)

    def array(self, data):
        return MockArray(data)

    class linalg:
        @staticmethod
        def norm(x):
             # Simplified norm
             if isinstance(x, (int, float)): return abs(x)
             return sum([v**2 for v in x])**0.5

    class random:
         @staticmethod
         def normal(loc=0.0, scale=1.0, size=None):
             return 0.0 # dummy
         @staticmethod
         def rand(d0=1):
             return 0.0 # dummy

    def min(self, a): return min(a) if isinstance(a, list) else a
    def max(self, a): return max(a) if isinstance(a, list) else a
    def abs(self, a):
        if isinstance(a, list): return MockArray([abs(x) for x in a])
        return abs(a)
    def mean(self, a): return sum(a)/len(a) if len(a) > 0 else 0.0
    def dot(self, a, b): return 0.0 # dummy
    def exp(self, a): return math.exp(a)
    def tanh(self, a): return math.tanh(a)
    def radians(self, x): return math.radians(x)
    def cos(self, x): return math.cos(x)
    def sin(self, x): return math.sin(x)
    def savez(self, *args, **kwargs): pass
    def load(self, *args, **kwargs): return {}

# Inject Mock
sys.modules['numpy'] = MockNumpy()

# ==========================================
# IMPORTS
# ==========================================

# Add project root to path to allow importing Core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from Core.swarm_core_v5_5 import LidarEngine, SwarmConfig
except ImportError as e:
    print(f"Failed to import LidarEngine: {e}")
    sys.exit(1)

# ==========================================
# TESTS
# ==========================================

class TestLidarProcessing(unittest.TestCase):
    def setUp(self):
        self.config = SwarmConfig()
        # Default MAX_RANGE is 3.0
        self.lidar = LidarEngine(self.config)

    def test_initialization(self):
        """Test initial state of LidarEngine."""
        # Check explicit zeros check (MockArray should equal list of zeros)
        self.assertEqual(self.lidar.sectors_16, [0.0]*16)
        self.assertEqual(self.lidar.min_dist, self.config.LIDAR_MAX_RANGE)

    def test_process_empty(self):
        """Test processing with no points."""
        points = []
        result = self.lidar.process(points)
        self.assertEqual(result, [0.0]*16)
        self.assertEqual(self.lidar.min_dist, self.config.LIDAR_MAX_RANGE)

    def test_process_basic_sectors(self):
        """Test points in specific sectors."""
        # Angle 0 -> Sector 0. Dist 1.5 (half range) -> Value should be 1.0 - (1.5/3.0) = 0.5
        # Angle 22.5 -> Sector 1. Dist 0.3 (close) -> Value 1.0 - (0.3/3.0) = 0.9
        points = [
            (0, 1.5),
            (23, 0.3) # 23 degrees falls in sector 1 (22.5 to 45)
        ]
        result = self.lidar.process(points)

        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 0.9)
        self.assertEqual(result[2], 0.0) # Should be empty

    def test_process_out_of_range(self):
        """Test points outside valid range are ignored."""
        points = [
            (0, 3.5),  # > MAX_RANGE
            (10, -1.0), # Negative
            (20, 0.0)   # Zero
        ]
        result = self.lidar.process(points)
        self.assertEqual(result, [0.0]*16)
        self.assertEqual(self.lidar.min_dist, self.config.LIDAR_MAX_RANGE)

    def test_process_min_distance(self):
        """Test that minimum distance in sector is used."""
        # Sector 0: Two points, one at 1.0, one at 2.0. Min is 1.0.
        # Result should be based on 1.0 -> 1.0 - (1.0/3.0) = 0.666...
        points = [
            (5, 2.0),
            (10, 1.0)
        ]
        result = self.lidar.process(points)
        self.assertAlmostEqual(result[0], 1.0 - 1.0/3.0)
        self.assertEqual(self.lidar.min_dist, 1.0)

    def test_process_normalization(self):
        """Test normalization logic."""
        # Dist 3.0 (Max) -> Value 0.0
        # Dist 0.001 (Very close) -> Value ~1.0
        points = [
            (0, 3.0),
            (25, 0.001)
        ]
        result = self.lidar.process(points)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 1.0 - (0.001/3.0))

    def test_check_front_sectors_blocked(self):
        """Test blockage detection."""
        # Mock sectors directly
        # Front sectors are usually 0, 1, 2, ...?
        # Wait, check_front_sectors_blocked uses self.sectors_16[:num_sectors]
        # Usually front is centered around 0?
        # If Lidar sector 0 is front, then indices 0,1,2,3 are front-left?
        # The implementation uses sectors_16[:num_sectors].

        # Case 1: All clear
        self.lidar.sectors_16 = MockArray([0.0]*16)
        self.assertFalse(self.lidar.check_front_sectors_blocked(0.5, 4))

        # Case 2: Blocked (values > threshold)
        # Set first 3 sectors to 0.8 (blocked)
        data = [0.0]*16
        data[0] = 0.8
        data[1] = 0.8
        data[2] = 0.8
        self.lidar.sectors_16 = MockArray(data)

        # Threshold 0.5. 3 sectors > 0.5. 3 >= 4//2 (2). Should be True.
        self.assertTrue(self.lidar.check_front_sectors_blocked(0.5, 4))

        # Case 3: Partial blocked but below count
        data = [0.0]*16
        data[0] = 0.8
        self.lidar.sectors_16 = MockArray(data)
        # 1 blocked. 1 < 2. Should be False.
        self.assertFalse(self.lidar.check_front_sectors_blocked(0.5, 4))

if __name__ == '__main__':
    unittest.main()
