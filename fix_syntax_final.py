with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

# It seems I left a dangling logger.info( because I removed lines 2589-2595 but maybe the indentation or previous line was truncated?
# The error says line 2553.
# Output:
#                          )
#                          self._last_concept = best_concept.name
#                          logger.info(
#        if final_action == self._last_action_type:

# This confirms a dangling logger.info( call.
# I need to remove it and the preceding line if it's part of the duplicate block.

new_lines = []
skip = False
for i, line in enumerate(lines):
    # Detect the bad block
    if 'self._last_concept = best_concept.name' in line and 'logger.info(' in lines[i+1]:
        # Check if it's the duplicate (i.e., not inside the if check properly or redundant)
        # Previous valid block ends with )
        # If we see ) then assignment then logger.info(, it's likely the duplicate.

        # In lines 2550-2560 output:
        # )
        # self._last_concept = ...
        # logger.info(
        # if final_action ...

        # The logger.info( is NOT closed before the next valid code (if final_action).
        # So we must remove  and .
        skip = True

    if skip:
        if 'if final_action ==' in line:
            skip = False
        else:
            continue

    new_lines.append(line)

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.writelines(new_lines)
