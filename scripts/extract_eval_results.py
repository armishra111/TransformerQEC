"""Extract evaluation plots and results from 03_evaluation.ipynb outputs.

Reads EVAL_FILE| lines from saved cell outputs and writes files to results/.

Usage:  python extract_eval_results.py
"""
import json, base64
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_EVAL = ROOT / 'notebooks' / '03_evaluation.ipynb'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

with open(NB_EVAL, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = 0
for cell in nb['cells']:
    for out in cell.get('outputs', []):
        for line in out.get('text', []):
            if line.startswith('EVAL_FILE|'):
                _, filename, b64 = line.strip().split('|', 2)
                raw = base64.b64decode(b64)
                path = RESULTS / filename
                path.write_bytes(raw)
                print(f'Saved: {path} ({len(raw):,} bytes)')
                found += 1

if found == 0:
    print('No evaluation outputs found in 03_evaluation.ipynb.')
    print('Run the notebook on Colab first, then save it (Ctrl+S).')
else:
    print(f'\nDone - {found} file(s) extracted to results/')
