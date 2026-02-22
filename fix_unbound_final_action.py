# The error "UnboundLocalError: cannot access local variable 'final_action'"
# means  is not defined in all branches before use.
# It is used in .
# It should be defined in  or .

# Let's check where it might be missing.
# If bumper_action is None, and override_action is None, and forced_turn is None,
# then  is calculated and  is set via stabilizer.
# So it should be covered.

# However, maybe I messed up the indentation or structure during edits.
# Let's initialize  at the start of loop.

import re

with open('Core/swarm_core_v5_5.py', 'r') as f:
    content = f.read()

# Insert initialization
pattern = r"# === INICJALIZACJA ZMIENNYCH \(ZAPOBIEGA BŁĘDOM\) ==="
replacement = r"""# === INICJALIZACJA ZMIENNYCH (ZAPOBIEGA BŁĘDOM) ===
        final_action = None
        source = "UNKNOWN"""

content = content.replace(pattern, replacement)

# Also handle  variable which might be unbound too.

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.write(content)
