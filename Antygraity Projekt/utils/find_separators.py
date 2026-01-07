
with open('extracted_content_full.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "File: " in line and "==================================================" in lines[i-1]:
            try:
                print(f"Line {i+1}: {line.strip().encode('cp1252', 'replace').decode('cp1252')}")
            except:
                print(f"Line {i+1}: {line.strip().encode('ascii', 'replace').decode('ascii')}")
