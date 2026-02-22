with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if '"""Safety check - TERAZ TYLKO GDY NAPRAWDĘ BLISKO!"""' in line:
        continue # Remove this line
    new_lines.append(line)

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.writelines(new_lines)
