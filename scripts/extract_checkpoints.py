"""Extract checkpoints from nb 02 and prepare nb 03 for Colab.

1. Reads any new base64-encoded checkpoints from 02_model_and_training.ipynb
   outputs and saves them as .pkl files to results/
2. Collects ALL transformer_qec_d*.pkl files in results/
3. Injects all of them into 03_evaluation.ipynb so it works on Colab
   (the VSCode Colab extension syncs the notebook but not sibling .pkl files)

Usage:  python extract_checkpoints.py
"""
import json, base64
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_TRAIN = ROOT / 'notebooks' / '02_model_and_training.ipynb'
NB_EVAL = ROOT / 'notebooks' / '03_evaluation.ipynb'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

# --- Step 1: Extract any new checkpoints from nb 02 outputs ---
if NB_TRAIN.exists():
    with open(NB_TRAIN, 'r', encoding='utf-8') as f:
        nb02 = json.load(f)
    for cell in nb02['cells']:
        for out in cell.get('outputs', []):
            for line in out.get('text', []):
                if line.startswith('CKPT|'):
                    _, filename, b64 = line.strip().split('|', 2)
                    path = RESULTS / filename
                    if not path.exists():
                        path.write_bytes(base64.b64decode(b64))
                        print(f'Extracted from nb 02: {path}')

# --- Step 2: Collect ALL .pkl checkpoint files on disk ---
checkpoints = {}  # filename -> base64 string
for pkl_path in sorted(RESULTS.glob('transformer_qec_d*.pkl')):
    raw = pkl_path.read_bytes()
    checkpoints[pkl_path.name] = base64.b64encode(raw).decode('ascii')
    print(f'Found: {pkl_path.name} ({len(raw):,} bytes)')

if not checkpoints:
    print('\nNo .pkl checkpoint files found in results/.')
    print('Train a model with 02_model_and_training.ipynb first.')
    raise SystemExit(1)

# --- Step 3: Inject all checkpoints into nb 03 ---
ckpt_lines = []
for filename, b64 in checkpoints.items():
    ckpt_lines.append(f'    "{filename}": "{b64}",')
ckpt_dict = "\n".join(ckpt_lines)

inject_source = (
    '# Auto-generated: checkpoint data for Colab compatibility.\n'
    '# This cell recreates .pkl files when running on a remote server.\n'
    '# Locally the files already exist so this is a no-op.\n'
    '# Re-run extract_checkpoints.py after retraining to update.\n'
    'import base64, os\n'
    'from pathlib import Path\n'
    '\n'
    '_CHECKPOINTS = {\n'
    f'{ckpt_dict}\n'
    '}\n'
    '\n'
    '_results = Path(r"C:\\Books\\Quant_Prep\\jupyter_notebooks\\QEC_ML\\results")\n'
    'if not _results.exists():\n'
    '    _results = Path(".")\n'
    '\n'
    'for _fname, _b64 in _CHECKPOINTS.items():\n'
    '    _path = _results / _fname\n'
    '    if not _path.exists():\n'
    '        _path.write_bytes(base64.b64decode(_b64))\n'
    '        print(f"Restored: {_fname}")\n'
    '    else:\n'
    '        print(f"Already exists: {_fname}")\n'
)

with open(NB_EVAL, 'r', encoding='utf-8') as f:
    nb03 = json.load(f)

MARKER = "# Auto-generated: checkpoint data"
source_lines = inject_source.split('\n')
source_json = [line + '\n' for line in source_lines]
if source_json:
    source_json[-1] = source_json[-1].rstrip('\n')

# Find existing injected cell or insert a new one
updated = False
for i, cell in enumerate(nb03['cells']):
    src = ''.join(cell.get('source', []))
    if MARKER in src:
        nb03['cells'][i]['source'] = source_json
        nb03['cells'][i]['outputs'] = []
        updated = True
        print('Updated checkpoint cell in 03_evaluation.ipynb')
        break

if not updated:
    insert_idx = len(nb03['cells']) - 1
    for i, cell in enumerate(nb03['cells']):
        src = ''.join(cell.get('source', []))
        if 'DISTANCES = [' in src:
            insert_idx = i
            break

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_json,
    }
    nb03['cells'].insert(insert_idx, new_cell)
    print('Injected checkpoint cell into 03_evaluation.ipynb')

with open(NB_EVAL, 'w', encoding='utf-8') as f:
    json.dump(nb03, f, indent=1, ensure_ascii=False)

print(f'\nDone — {len(checkpoints)} checkpoint(s) injected into 03_evaluation.ipynb:')
for name in checkpoints:
    print(f'  - {name}')
