import sys

with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

def replace_block(lines, start_marker, end_marker_next_class, new_code):
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == start_marker:
            start_idx = i
        if start_idx != -1 and line.strip().startswith('class ') and i > start_idx and line.strip() != start_marker:
            end_idx = i
            break

    if start_idx != -1:
        if end_idx == -1: # Last class in file or block
             # Find end by indentation or until end of file
             pass # Logic for single class replacement

        # We assume known order or finding next class
        # For StuckDetector, it is followed by MovementTracker
        return lines[:start_idx] + [new_code + '\n'] + lines[end_idx:]
    return lines

# Load new code
with open('update_modules.py', 'r') as f:
    new_code = f.read()

# Split new code into classes
classes = {}
current_class = None
buffer = []
for line in new_code.split('\n'):
    if line.strip().startswith('class '):
        if current_class:
            classes[current_class] = '\n'.join(buffer)
        current_class = line.strip().split(':')[0]
        buffer = [line]
    else:
        buffer.append(line)
if current_class:
    classes[current_class] = '\n'.join(buffer)

# Replace StuckDetector
# In file: class StuckDetector -> class MovementTracker
start_sd = -1
end_sd = -1
for i, line in enumerate(lines):
    if line.strip() == 'class StuckDetector:':
        start_sd = i
    if start_sd != -1 and line.strip().startswith('class MovementTracker:'):
        end_sd = i
        break

if start_sd != -1 and end_sd != -1:
    lines = lines[:start_sd] + [classes['class StuckDetector'] + '\n\n'] + lines[end_sd:]

# Replace LidarEngine
# In file: class LidarEngine -> class DualLinearApproximator (actually, LidarEngine is early in file)
# In file: class LidarEngine -> class FeatureExtractor (wait, check file structure)
# LidarEngine is usually before FeatureExtractor or after config.
# Let's find it.
start_le = -1
end_le = -1
for i, line in enumerate(lines):
    if line.strip() == 'class LidarEngine:':
        start_le = i
    if start_le != -1 and line.strip().startswith('class RunningNormalizer:'):
        end_le = i
        break

if start_le != -1 and end_le != -1:
    lines = lines[:start_le] + [classes['class LidarEngine'] + '\n\n'] + lines[end_le:]

# Replace SensorFusion
# In file: class SensorFusion -> class BumperSystem
start_sf = -1
end_sf = -1
for i, line in enumerate(lines):
    if line.strip() == 'class SensorFusion:':
        start_sf = i
    if start_sf != -1 and line.strip().startswith('class BumperSystem:'):
        end_sf = i
        break

if start_sf != -1 and end_sf != -1:
    lines = lines[:start_sf] + [classes['class SensorFusion'] + '\n\n'] + lines[end_sf:]

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.writelines(lines)

print("Modules updated")
