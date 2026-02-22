import sys

with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

# Locate the line with syntax error
for i, line in enumerate(lines):
    if '"""Safety check - TERAZ TYLKO GDY NAPRAWDĘ BLISKO!"""' in line:
        # Check indentation
        indent = len(line) - len(line.lstrip())
        prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
        print(f"Line {i+1}: indent={indent}, prev={prev_indent}")

        # If it's inside valid_safety_constraints definition, it should be indented.
        # But wait, did I mess up the definition line?
        # The previous line is:
        #                                   rear_bumper: int = 0) -> Optional[Tuple[Action, str]]:
        # It seems fine.

        # Maybe an invisible char or encoding issue?
        # Or maybe the previous line does NOT end with : ?
        print(f"Prev line: {lines[i-1].strip()}")
        if not lines[i-1].strip().endswith(':'):
             print("Prev line does not end with :")
             # Add :
             lines[i-1] = lines[i-1].rstrip() + ':\n'

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.writelines(lines)
