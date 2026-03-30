"""Extract checkpoints from nb 02 and prepare nb 03 for Colab.

1. Reads any new base64-encoded checkpoints from 02_model_and_training.ipynb
   outputs and saves them as .pkl files to results/
2. Collects ALL transformer_qec_d*.pkl files in results/
3. Injects a download cell into 03_evaluation.ipynb so it works on Colab
   (the VSCode Colab extension syncs the notebook but not sibling .pkl files)

Note: The old approach embedded base64-encoded pkl data directly in the cell
source.  This created a ~69 MB cell that stalls Colab's Python parser.
Instead we now inject a lightweight cell that downloads the pkl files from
the GitHub repo at runtime.

Usage:  python extract_checkpoints.py
"""
import json, base64, subprocess
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
pkl_files = sorted(RESULTS.glob('transformer_qec_d*.pkl'))
for pkl_path in pkl_files:
    print(f'Found: {pkl_path.name} ({pkl_path.stat().st_size:,} bytes)')

if not pkl_files:
    print('\nNo .pkl checkpoint files found in results/.')
    print('Train a model with 02_model_and_training.ipynb first.')
    raise SystemExit(1)

# --- Step 3: Detect GitHub remote for download URL ---
repo = "armishra111/TransformerQEC"
branch = "main"
try:
    url = subprocess.check_output(
        ['git', 'remote', 'get-url', 'origin'],
        cwd=str(ROOT), text=True
    ).strip()
    # Parse "https://github.com/OWNER/REPO" or "git@github.com:OWNER/REPO.git"
    if 'github.com' in url:
        parts = url.rstrip('.git').replace(':', '/').split('/')
        repo = f'{parts[-2]}/{parts[-1]}'
    branch_out = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=str(ROOT), text=True
    ).strip()
    if branch_out:
        branch = branch_out
except Exception:
    pass  # fall back to defaults

print(f'GitHub repo: {repo}, branch: {branch}')

# --- Step 4: Inject download cell into nb 03 ---
file_list = ', '.join(f'"{p.name}"' for p in pkl_files)

inject_source = (
    '# Auto-generated: download checkpoint files for Colab compatibility.\n'
    '# This cell downloads .pkl files from GitHub when running on a remote\n'
    '# server where only the notebook is synced (not sibling data files).\n'
    '# Locally the files already exist so this is a no-op.\n'
    '# Re-run extract_checkpoints.py after retraining to update.\n'
    'import urllib.request\n'
    'from pathlib import Path\n'
    '\n'
    f'_REPO = "{repo}"\n'
    f'_BRANCH = "{branch}"\n'
    f'_FILES = [{file_list}]\n'
    '\n'
    '_results = Path("../results").resolve()\n'
    'if not _results.exists():\n'
    '    _results = Path(".")\n'
    '    _results.mkdir(exist_ok=True)\n'
    '\n'
    'for _fname in _FILES:\n'
    '    _path = _results / _fname\n'
    '    if not _path.exists():\n'
    '        _url = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}/results/{_fname}"\n'
    '        print(f"Downloading {_fname} ...")\n'
    '        urllib.request.urlretrieve(_url, str(_path))\n'
    '        print(f"  Saved to {_path}")\n'
    '    else:\n'
    '        print(f"Already exists: {_fname}")\n'
)

with open(NB_EVAL, 'r', encoding='utf-8') as f:
    nb03 = json.load(f)

# Match either old (inline base64) or new (download) marker
OLD_MARKER = "# Auto-generated: checkpoint data"
NEW_MARKER = "# Auto-generated: download checkpoint"
source_lines = inject_source.split('\n')
source_json = [line + '\n' for line in source_lines]
if source_json:
    source_json[-1] = source_json[-1].rstrip('\n')

# Find existing injected cell or insert a new one
updated = False
for i, cell in enumerate(nb03['cells']):
    src = ''.join(cell.get('source', []))
    if OLD_MARKER in src or NEW_MARKER in src:
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

nb_json = json.dumps(nb03, indent=1, ensure_ascii=False)
with open(NB_EVAL, 'w', encoding='utf-8') as f:
    f.write(nb_json)

print(f'\nDone — download cell for {len(pkl_files)} checkpoint(s) '
      f'injected into 03_evaluation.ipynb:')
for p in pkl_files:
    print(f'  - {p.name}')
