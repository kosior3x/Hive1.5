import sys

with open('Core/swarm_core_v5_5.py', 'r') as f:
    lines = f.readlines()

save_idx = -1
load_idx = -1
save_body_start_idx = -1

for i, line in enumerate(lines):
    if 'def save_state(self):' in line:
        save_idx = i
    elif 'def _load_state(self):' in line:
        load_idx = i
    elif '2. Przygotuj dane do pickle' in line:
        save_body_start_idx = i

if save_idx != -1 and load_idx != -1 and save_body_start_idx != -1:
    # Extract load body: from load_idx+1 to save_body_start_idx
    load_body = lines[load_idx:save_body_start_idx]

    # Extract save body: from save_body_start_idx to ... where?
    # Until next method or end of class.
    # The next method is probably _compute_dynamic_safety.
    next_method_idx = -1
    for i in range(save_body_start_idx, len(lines)):
        if 'def _compute_dynamic_safety' in line: # Wait, grep dynamic safety
             pass
        if lines[i].strip().startswith('def '):
             next_method_idx = i
             break

    # Actually, we can just grab everything from save_body_start_idx to next method
    # But wait, looking at file structure, save_body ended with state_manager.save

    # Let's find end of save body by indentation or knowing it ends before _compute_dynamic_safety
    end_of_save_idx = -1
    for i in range(save_body_start_idx, len(lines)):
        if 'def _compute_dynamic_safety' in lines[i]:
            end_of_save_idx = i
            break

    if end_of_save_idx == -1:
        # Maybe it's the last method?
        # Let's assume it goes until end of block logic.
        # But wait, _compute_dynamic_safety follows _load_state usually.
        pass

    if end_of_save_idx != -1:
        save_body = lines[save_body_start_idx:end_of_save_idx]

        # Construct new list
        # Before save_state
        part1 = lines[:save_idx+1]
        # save_body
        part2 = save_body
        # load_body (includes def _load_state)
        part3 = load_body
        # Rest
        part4 = lines[end_of_save_idx:]

        new_lines = part1 + part2 + ['\n'] + part3 + part4

        with open('Core/swarm_core_v5_5.py', 'w') as f:
            f.writelines(new_lines)
        print("Fixed save_state and _load_state order")
    else:
        print("Could not find end of save body")
else:
    print(f"Indices not found: save={save_idx}, load={load_idx}, body={save_body_start_idx}")
