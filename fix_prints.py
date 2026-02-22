with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'class EncoderMonitor:' in line:
        # Search forward for init
        for j in range(i, min(i+20, len(lines))):
            if 'def __init__' in lines[j]:
                # Next line has print
                if 'print(f"REAL_BUMPER init' in lines[j+1]:
                    lines[j+1] = lines[j+1].replace('REAL_BUMPER', 'REAL_ENCODER')
                break
        break

with open('Core/swarm_core_v5_5.py', 'w') as f:
    f.writelines(lines)
