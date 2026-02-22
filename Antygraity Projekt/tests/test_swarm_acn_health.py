import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# 1. Mock numpy BEFORE importing the module
mock_np = MagicMock()
mock_np.random.rand.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
mock_np.zeros.return_value = [0.0] * 5
mock_np.median = lambda x: sorted(x)[len(x)//2]
# Mock where to return a tuple containing a list (acting as array)
mock_np.where.return_value = ([0],)

sys.modules['numpy'] = mock_np

# 2. Import the module under test
import importlib.util

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Path to the file
file_path = "Antygraity Projekt/src/swarm_acn.py"
if not os.path.exists(file_path):
    # Fallback for different CWD
    file_path = "../src/swarm_acn.py"

swarm_acn = load_module_from_path("swarm_acn", file_path)

class TestSwarmACN(unittest.TestCase):
    def test_acn_tick_runs(self):
        """
        Verify that acn_tick runs without errors.
        This serves as a regression test for the refactoring.
        """
        # Create mock agents
        sender = MagicMock()
        receiver = MagicMock()

        sender.id = 1
        receiver.id = 2

        # Mock BioAgent properties
        sender.stm = MagicMock()
        sender.stm.copy.return_value = "stimulus_copy"
        sender.stm.__len__.return_value = 5

        sender.l_state = (0.1, 0.2, 0.3)

        sender.activity = MagicMock()
        # Mock comparison > 0.4 for the line: np.where(sender.activity > 0.4)
        sender.activity.__gt__.return_value = MagicMock()

        agents = [sender, receiver]

        # We need to patch random.sample because we want deterministic sender/receiver
        with patch('random.sample', return_value=(sender, receiver)):
            sid, rid, energy = swarm_acn.acn_tick(agents)

        # Assertions
        self.assertEqual(sid, 1)
        self.assertEqual(rid, 2)
        # Ensure receive_stimulus was called with the copy of STM
        receiver.receive_stimulus.assert_called_with("stimulus_copy", energy)

if __name__ == '__main__':
    unittest.main()
