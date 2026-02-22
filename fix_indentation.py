import sys

with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

# The error "expected an indented block after 'if' statement on line 2545"
# Line 2545 is: if best_concept and (not hasattr(self, "_last_concept") or self._last_concept != best_concept.name):
# The next line (2546) is: if final_action == self._last_action_type:
# This means the body of the if at 2545 is missing!
# I likely deleted it when removing the duplicate.

# I need to restore the body OR remove the empty if.
# The body was the concept logging.
# I see I inserted  earlier.
# But then I ran  logic (or manual sed) which might have been too aggressive.

# Let's restore the body.
# It should be:
#                          self._last_concept = best_concept.name
#                          logger.info(
#                              f"[CONCEPT] Uzyto konceptu '{best_concept.name}' "
#                              f"(aktywacja={best_concept.activation:.2f}) "
#                              f"-> sugestia: {concept_suggestion.name if concept_suggestion else 'None'}"
#                          )

# Find the empty if
start_idx = -1
for i, line in enumerate(lines):
    if 'if best_concept and (not hasattr' in line:
        start_idx = i
        break

if start_idx != -1:
    body = [
        '                          self._last_concept = best_concept.name\n',
        '                          logger.info(\n',
        '                              f"[CONCEPT] Uzyto konceptu \'{best_concept.name}\' "\n',
        '                              f"(aktywacja={best_concept.activation:.2f}) "\n',
        '                              f"-> sugestia: {concept_suggestion.name if concept_suggestion else \'None\'}"\n',
        '                          )\n'
    ]

    # Check if next line is already indented (it's not, it's  at level 8, while if at 2545 is at level 22)
    # So we insert the body.

    new_lines = lines[:start_idx+1] + body + lines[start_idx+1:]

    with open('Core/swarm_core_v5_5.py', 'w') as f:
        f.writelines(new_lines)
    print("Restored concept logging body")
else:
    print("Could not find empty if")
