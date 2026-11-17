import json
import sys

nb_path = sys.argv[1] if len(sys.argv) > 1 else 'local_data/pulled_v57/name-that-llm-v2-submission.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lines = [l for l in src.splitlines() if l.strip()]
    first_line = lines[0] if lines else 'EMPTY'
    print(f"Cell {i} ({c.get('cell_type')}): {first_line[:90]}")
