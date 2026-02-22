import sys

with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

# Insert _decide_from_fusion BEFORE loop if missing
# It should be after validate_safety_constraints

safety_end = -1
for i, line in enumerate(lines):
    if 'def validate_safety_constraints' in line:
        safety_end = i
    if 'def loop(self,' in line:
        loop_idx = i
        break

if safety_end != -1 and loop_idx != -1:
    # Check if decide is between them
    has_decide = False
    for j in range(safety_end, loop_idx):
        if 'def _decide_from_fusion' in lines[j]:
            has_decide = True
            break

    if not has_decide:
        decide_method = [
            '    def _decide_from_fusion(self, enc_l, enc_r) -> Tuple[Optional[Action], str]:\n',
            '        # POZIOM 2: Blokada\n',
            '        if self.stuck_detector.update(self.fusion.front_dist, enc_l, enc_r):\n',
            '            logger.warning("STUCK DETECTED! Uwalniam robota...")\n',
            '            if self.fusion.rear_dist > 0.3:\n',
            '                return Action.REVERSE, "STUCK_REVERSE"\n',
            '            else:\n',
            '                if self.fusion.left_dist >= self.fusion.right_dist:\n',
            '                    return Action.SPIN_LEFT, "STUCK_SPIN_LEFT"\n',
            '                else:\n',
            '                    return Action.SPIN_RIGHT, "STUCK_SPIN_RIGHT"\n',
            '\n',
            '        # POZIOM 3: Bezpieczenstwo\n',
            '        if self.fusion.front_dist < self.config.LIDAR_HARD_SAFETY_MIN:\n',
            '             if self.fusion.rear_dist > 0.3:\n',
            '                  return Action.REVERSE, "SAFETY_REVERSE"\n',
            '             else:\n',
            '                  if self.fusion.left_dist >= self.fusion.right_dist:\n',
            '                       return Action.SPIN_LEFT, "SAFETY_SPIN"\n',
            '                  else:\n',
            '                       return Action.SPIN_RIGHT, "SAFETY_SPIN"\n',
            '        return None, "NORMAL"\n',
            '\n'
        ]

        # Insert before loop
        new_lines = lines[:loop_idx] + decide_method + lines[loop_idx:]
        with open('Core/swarm_core_v5_5.py', 'w') as f:
            f.writelines(new_lines)
        print("Restored _decide_from_fusion")
    else:
        print("_decide_from_fusion exists")
