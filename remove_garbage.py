with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'min(self.config.SAFETY_LIDAR_MAX, lidar_safety))' in line and 'return' not in line:
        continue
    new_lines.append(line)

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.writelines(new_lines)
