import sys

with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

# Find class BumperSystem:
start_idx = -1
for i, line in enumerate(lines):
    if line.strip() == 'class BumperSystem:':
        start_idx = i
        break

if start_idx == -1:
    print("Could not find BumperSystem")
    sys.exit(1)

# Check if followed by class EncoderMonitor:
if start_idx + 1 < len(lines) and 'class EncoderMonitor:' in lines[start_idx+1]:
    # Insert BumperSystem body
    bumper_code = [
        '    """\n',
        '    Obsługa fizycznych zderzaków (damperów).\n',
        '    Absolutny priorytet.\n',
        '    """\n',
        '    def __init__(self, config: SwarmConfig):\n',
        '        self.config = config\n',
        '        self.escape_sequence = 0\n',
        '        self.escape_action = None\n',
        '        \n',
        '    def check_collision(self, rear_bumper: int) -> Optional[Action]:\n',
        '        # Jeśli trwa ucieczka\n',
        '        if self.escape_sequence > 0:\n',
        '            self.escape_sequence -= 1\n',
        '            return self.escape_action\n',
        '            \n',
        '        # Wykrycie kolizji\n',
        '        if rear_bumper == 1:\n',
        '            # Kolizja z tyłu -> uciekaj do przodu\n',
        '            logger.warning("BUMPER: Kolizja z tyłu!")\n',
        '            self.escape_sequence = 5 # 5 cykli\n',
        '            self.escape_action = Action.FORWARD\n',
        '            return Action.FORWARD\n',
        '            \n',
        '        # (Tu można dodać obsługę przednich/bocznych bumperów jeśli są w inputach)\n',
        '        return None\n',
        '\n'
    ]

    new_lines = lines[:start_idx+1] + bumper_code + lines[start_idx+1:]

    with open('Core/swarm_core_v5_5.py', 'w') as f:
        f.writelines(new_lines)
    print("Fixed BumperSystem")
else:
    print("Structure not as expected")
