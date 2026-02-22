import re

file_path = 'Core/swarm_core_v5_5.py'
with open(file_path, 'r') as f:
    content = f.read()

# 1. Remove duplicate concept logging
# Look for the duplicated block around line 2589
# The first block is correct (conditional). The second one might be unconditional or duplicate.
# In previous steps I added one.
# Let's search for the logger.info call for concept.

concept_log_pattern = r'if best_concept and \(not hasattr\(self, "_last_concept"\) or self._last_concept != best_concept\.name\):\s+self\._last_concept = best_concept\.name\s+logger\.info\(\s+f"\[CONCEPT\] Uzyto konceptu.*?\'None\'"\s+\)'

# Find all occurrences
matches = list(re.finditer(concept_log_pattern, content, re.DOTALL))
print(f"Found {len(matches)} concept log blocks")

# If more than 1, we might want to keep only one.
# But wait, the user said "Remove duplicate concept logging (linia ~2650)".
# And "Duplikat logowania konceptów - wystarczy jedno!".
# Let's inspect the file content around there.

# 2. Remove dead code at the end
# "POZIOM 1: Damper handled in loop..."
# This is a comment block, maybe followed by code?
dead_code_marker = "# POZIOM 1: Damper handled in loop"
if dead_code_marker in content:
    print("Found dead code marker")
    # Remove from marker to end of file? Or just the marker block?
    # The user said "linia ~2950... To jest duplikat metody _decide_from_fusion - do usunięcia!"
    # It seems to be a leftover function body without definition?
    # Let's find where it starts and cut it.
    idx = content.find(dead_code_marker)
    # Check if it is inside a function or dangling.
    # If it is at the very end of class SwarmCoreV55, we can cut it.
    # But be careful not to cut valid code.
    pass

# 3. StuckDetector update with cooldown
# We need to add cooldown logic.
# Update __init__ and update()

# 4. SensorFusion update
# Ensure it uses min(us_min, lidar_front_dist)
