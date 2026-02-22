import sys

with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

# Insert StuckDetector before SwarmCoreV55
# Find class SwarmCoreV55
insert_idx = -1
for i, line in enumerate(lines):
    if 'class SwarmCoreV55' in line:
        insert_idx = i
        break

if insert_idx != -1:
    stuck_code = [
        'class StuckDetector:\n',
        '    """\n',
        '    Wykrywa czy robot jest zablokowany.\n',
        '    Koła się kręcą (enkodery > 0), ale odległość z przodu nie maleje.\n',
        '    """\n',
        '    def __init__(self, config: SwarmConfig):\n',
        '        self.config = config\n',
        '        self.front_dist_history = deque(maxlen=10)\n',
        '        self.stuck_count = 0\n',
        '        self.last_stuck_cycle = 0\n',
        '        self.stuck_cooldown = 50\n',
        '        \n',
        '    def update(self, front_dist: float, enc_l: float, enc_r: float, current_cycle: int = 0) -> bool:\n',
        '        if current_cycle - self.last_stuck_cycle < self.stuck_cooldown:\n',
        '            return False\n',
        '            \n',
        '        self.front_dist_history.append(front_dist)\n',
        '        \n',
        '        wheels_moving = abs(enc_l) > 0.02 or abs(enc_r) > 0.02\n',
        '        \n',
        '        if not wheels_moving:\n',
        '            self.stuck_count = 0\n',
        '            return False\n',
        '            \n',
        '        if len(self.front_dist_history) == 10:\n',
        '            dist_std = np.std(self.front_dist_history)\n',
        '            if dist_std < 0.01:\n',
        '                self.stuck_count += 1\n',
        '                if self.stuck_count > 8:\n',
        '                    self.last_stuck_cycle = current_cycle\n',
        '                    return True\n',
        '            else:\n',
        '                self.stuck_count = max(0, self.stuck_count - 1)\n',
        '                \n',
        '        return False\n',
        '\n'
    ]

    new_lines = lines[:insert_idx] + stuck_code + lines[insert_idx:]
    with open('Core/swarm_core_v5_5.py', 'w') as f:
        f.writelines(new_lines)
    print("Restored StuckDetector")
else:
    print("SwarmCoreV55 not found")
