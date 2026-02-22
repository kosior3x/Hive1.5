with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    # Detect the duplicate block.
    # It starts with "self._last_concept = best_concept.name" AND previous line was ")" (end of logger call)
    # Actually, in the output:
    #                          )
    #                          self._last_concept = best_concept.name
    #                          logger.info(
    #                              f"[CONCEPT] Uzyto konceptu '{best_concept.name}' "
    #                              f"(aktywacja={best_concept.activation:.2f}) "
    #                              f"-> sugestia: {concept_suggestion.name if concept_suggestion else 'None'}"
    #                          )

    # We want to keep the FIRST block and remove the SECOND.
    # The first block starts with 'if best_concept and ...'

    # Let's just remove the lines 2589-2595 based on visual inspection of previous  output
    # But line numbers might shift.

    # Identify the duplicate:
    if "self._last_concept = best_concept.name" in line and "if best_concept" not in lines[i-1]:
        # This looks like the start of the duplicate block (it repeats assignment without check, or maybe the check covers both?)
        # Wait, the check  covers the indentation block.
        # Inside that block, we have:
        # assignment
        # logger call
        # assignment (Duplicate!)
        # logger call (Duplicate!)

        # So we should remove the second assignment and subsequent logger call.
        # Scan forward to remove.
        skip = True

    if skip:
        if ")" in line and "sugestia" not in line: # End of logger call
             skip = False
             continue # Skip this closing parenthesis too
        continue

    new_lines.append(line)

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.writelines(new_lines)
