import sys

with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

# It seems I pasted validate_safety_constraints multiple times or didn't remove the old one correctly.
# And indentation/structure is messed up.
# The error says "SyntaxError: invalid syntax" at def validate_safety_constraints
# This usually happens if previous line is incomplete or indentation is wrong.
# Previous lines in output:
#        return None
#        lidar_safety = self.config.LIDAR_SAFETY_RADIUS + (scale * v)
#        lidar_safety = max(self.config.SAFETY_LIDAR_MIN,
#    def validate_safety_constraints...

# The previous lines look like garbage/leftovers.
# "lidar_safety = ..." seems to be from _compute_dynamic_safety?
# But it's outside a function? Or indented?
# It seems I messed up when inserting/replacing code blocks.

# Strategy:
# 1. Identify valid classes/methods.
# 2. Clean up garbage lines between methods.

cleaned_lines = []
skip = False
for i, line in enumerate(lines):
    # Heuristic: if we see "lidar_safety =" indented at level 8, but it's not inside a function we expect...
    # Actually, let's just look at the specific error location.

    # Error at 2823.
    # Lines 2820-2822 seem to be orphans from a bad paste/replace.
    # "return None" might be from previous function.
    # Then "lidar_safety = ..." which is incomplete logic.

    if i >= 2820 and i < 2823:
        # Check if these are the specific garbage lines
        if 'lidar_safety =' in line or 'lidar_safety = max' in line:
             continue

    cleaned_lines.append(line)

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.writelines(cleaned_lines)
