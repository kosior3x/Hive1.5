import re

with open('Core/swarm_core_v5_5.py', 'r') as f:
    content = f.read()

pattern = r"def _compute_dynamic_safety\(self, avg_speed: float\) -> Tuple\[float, float\]:.*?return us_safety, lidar_safety"
replacement = r'''def _compute_dynamic_safety(self, avg_speed: float) -> Tuple[float, float]:
        """
        Oblicza dynamiczne progi bezpieczeństwa.
        Teraz znacznie odważniejsze - reaguje tylko naprawdę blisko!
        """
        scale = self.config.SAFETY_DIST_SPEED_SCALE
        v = abs(avg_speed)

        # US safety - od 8 cm do 15 cm zależnie od prędkości
        us_safety = self.config.US_SAFETY_DIST + (scale * v)
        us_safety = max(self.config.SAFETY_US_MIN,
                        min(self.config.SAFETY_US_MAX, us_safety))

        # LIDAR safety - analogicznie
        lidar_safety = self.config.LIDAR_SAFETY_RADIUS + (scale * v)
        lidar_safety = max(self.config.SAFETY_LIDAR_MIN,
                           min(self.config.SAFETY_LIDAR_MAX, lidar_safety))

        return us_safety, lidar_safety'''

# Note: The pattern in file might be slightly different due to previous messed up edits.
# It might end with "return us, lr" if I reverted or "return us_safety, lidar_safety" if I updated.
# Let's try both or use a regex that catches the end of the block.
# Previous valid code had "return us, lr". My update changed it to "return us_safety, lidar_safety".
# But the update was done via 'sed' which might have failed or left garbage.
# Let's read the file to see what's actually there.

print("Fixing...")
# We will use string search instead of regex for safety if regex fails
start_marker = "def _compute_dynamic_safety(self, avg_speed: float) -> Tuple[float, float]:"
end_marker = "def validate_safety_constraints"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    new_content = content[:start_idx] + replacement + "\n\n    " + content[end_idx:]
    with open('Core/swarm_core_v5_5.py', 'w') as f:
        f.write(new_content)
    print("Fixed via string slicing")
else:
    print("Could not find markers")
