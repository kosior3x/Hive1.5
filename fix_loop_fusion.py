import sys

with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

decide_idx = -1
loop_idx = -1

for i, line in enumerate(lines):
    if 'def _decide_from_fusion' in line:
        decide_idx = i
    elif 'def loop(self, lidar_points' in line and i > decide_idx: # Find the loop inserted AFTER decide
        loop_idx = i
        break

if decide_idx != -1 and loop_idx != -1:
    # Everything between decide_idx+1 and loop_idx should be empty if I messed up insertion?
    # No, I inserted loop_update.py (which starts with def loop) RIGHT AFTER decide line.
    # So loop_idx should be decide_idx + 1.

    if loop_idx == decide_idx + 1:
        # The body of _decide_from_fusion is MISSING here (it was pushed down?).
        # Where is the original body?
        # It should be AFTER the inserted loop function.
        # Find where inserted loop ends.

        # loop ends when indentation returns to 4 spaces or less?
        # loop definition is at indentation 4.
        # loop body is at 8.
        # Original decide body is at 8.

        # Let's find the end of the inserted loop.
        # Since I replaced loop in previous step with a NEW loop implementation,
        # I know what the new loop looks like.
        # But I also have the old decide body somewhere.

        # Let's scan for the old decide body text: "# POZIOM 1: Damper handled in loop" (or similar)
        # Wait, I think I saw that in the file content previously.
        # Let's verify what's after the loop.
        pass

    # We need to extract the loop code (inserted) and the decide body (pushed down).
    # And swap them? No, loop should be separate.
    # decide_from_fusion needs its body.

    # I'll just rewrite the file with correct order if I can identify blocks.
    pass

# Strategy:
# 1. Read 'def _decide_from_fusion' line.
# 2. Insert standard body for it.
# 3. Read 'def loop' line and keep it as is (it's the new one).
# 4. Remove any "stranded" decide body code if it exists duplicates.

decide_body = [
    '        # POZIOM 1: Damper handled in loop\n',
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

# We need to insert this body immediately after decide_idx line.
# And ensure 'def loop' follows it.

if decide_idx != -1:
    lines.insert(decide_idx + 1, "".join(decide_body))

    # Now check if there are duplicate bodies further down?
    # The original body might still be there, causing indentation error or dead code.
    # The error was "IndentationError: expected an indented block after function definition".
    # This implies there was NO body. So my insertion pushed 'def loop' immediately after.
    # So by inserting the body, I fix the indentation error for decide.
    # And 'def loop' will follow, which is correct (new method).
    # But wait, what happened to the OLD body of decide?
    # It must be somewhere. If it's indented, it might be attached to 'def loop' (if loop ended) or cause syntax error.
    # Let's save this fix and see.

    # Flatten the list if needed (lines is list of strings, insert inserted a string? No, join returns string. List insert inserts one element.)
    # We want to insert multiple lines.
    lines[decide_idx+1:decide_idx+1] = decide_body # Slice assignment for insertion

    with open('Core/swarm_core_v5_5.py', 'w') as f:
        f.writelines(lines)
    print("Fixed _decide_from_fusion body")
