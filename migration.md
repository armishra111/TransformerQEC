# TPU Research Cloud Migration

30-day TRC access. Goal: move TransformerQEC off Colab onto persistent TRC TPU VM.

---

## 1. How to use TPU Research Cloud

### 1.1 Prereqs

- TRC approval email naming your **GCP project** + allowed **zone** + **TPU type** (e.g. `v3-8` in `us-central1-f`, or `v4-8` in `us-central2-b`).
- `gcloud` CLI installed + authed:
  ```bash
  gcloud auth login
  gcloud config set project <TRC_PROJECT_ID>
  ```
- Billing enabled on project (TRC covers TPU cost; storage + egress still billed — keep GCS in same region as TPU).

### 1.2 Pick TPU type

| Type   | Chips | HBM/chip | bf16 TFLOPs (total) | Use for TransformerQEC |
|--------|-------|----------|---------------------|------------------------|
| v2-8   | 8     | 8 GB     | ~180                | d≤7 only; OOM at d=11. skip |
| v3-8   | 8     | 16 GB    | ~420                | fallback. d=11 works B≤48 |
| v4-8   | 8     | 32 GB    | ~2200               | **preferred.** d=11 B=128 easy |
| v5e-8  | 8     | 16 GB    | ~1600               | acceptable. ≈v4-8 for this size |
| v6e-1  | 1     | 32 GB    | ~918                | last resort. ~40% v4-8 throughput |
| v6e-4  | 4     | 32 GB    | ~3700               | strong. beats v4-8 if granted |
| v6e-8  | 8     | 32 GB    | ~7300               | ideal for full d=3–11 sweep |
| v5p-8 / v4-32+ pods | multi | — | — | overkill; needs `jax.distributed` |

Single-host (v3-8 / v4-8 / v6e-8) simplest. No `pjit`/multi-host plumbing.

**Ask TRC in this order:** v6e-8 → v4-8 → v6e-4 → v5e-8 → v3-8 → v6e-1.

**Why not v6e-1?** Single chip, no 8-way `pmap` data parallelism. ~40% v4-8 throughput for 1.3M param model. Only take if nothing larger granted.

### 1.2a d-scaling memory budget (rotated_memory_z, circuit-level)

Seq length L ≈ (d²−1)/2 · (d+1). Attention O(L²).

| d  | L    | Attn matrix B=64 bf16 / layer | Fits v3-8? | Fits v4-8? |
|----|------|-------------------------------|------------|------------|
| 3  | ~16  | 130 KB                        | yes        | yes        |
| 5  | ~72  | 2.7 MB                        | yes        | yes        |
| 7  | ~192 | 19 MB                         | yes        | yes        |
| 9  | ~400 | 82 MB                         | yes B=64   | yes B=128  |
| 11 | ~720 | 265 MB                        | tight B=32 | yes B=128  |
| 13 | ~1200| 737 MB                        | OOM        | yes B=48   |

`nn.remat` already on (model.py:302). bf16 already on. Static pad to `L_max=750` for full sweep to avoid XLA recompile per d.

### 1.3 Provision TPU VM

```bash
export TPU_NAME=qec-tpu
export ZONE=us-central1-f          # use TRC-assigned zone
export ACCEL=v3-8                  # use TRC-assigned type
export RUNTIME=tpu-vm-base         # v4: tpu-ubuntu2204-base

gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=$ACCEL \
  --version=$RUNTIME
```

SSH:
```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE
```

### 1.4 Verify TPU visible from JAX

On VM:
```bash
python3 -c "import jax; print(jax.devices()); print(jax.device_count())"
```
Expect 8 `TpuDevice(...)`. If CPU only → libtpu/JAX version mismatch (§1.7).

### 1.5 GCS bucket (data + checkpoints)

Colab used Google Drive. TRC uses GCS.

```bash
export BUCKET=qec-<yourhandle>
gsutil mb -l us-central1 gs://$BUCKET   # same region as TPU
gsutil mb -l us-central1 gs://$BUCKET-ckpt
```

Read/write from Python via `tf.io.gfile` or `gcsfs`:
```python
import gcsfs, pickle
fs = gcsfs.GCSFileSystem()
with fs.open('gs://qec-xxx-ckpt/best_d5_20260422_120000_checkpoint.pkl','wb') as f:
    pickle.dump(ckpt, f)
```

Or plain `gsutil cp local.pkl gs://...` after each save.

### 1.6 Install project deps

```bash
git clone <repo-url> && cd TransformerQEC
pip install --upgrade pip
pip install -r requirements.txt
# Match JAX TPU wheel to runtime libtpu:
pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Sanity:
```bash
python3 -c "import jax, flax, optax, stim, pymatching; print('ok', jax.devices())"
pytest tests/test_equivariance.py -q    # expect 4/4 pass
```

### 1.7 Version pin cheat sheet

- `jax==0.4.x` + matching `libtpu-nightly` (installed via `jax[tpu]` extra).
- `flax>=0.8`, `optax>=0.2`, `numpy<2`, `stim>=1.14,<2`, `pymatching>=2`.
- If `jax.devices()` returns CPU: runtime/libtpu drift. Re-create VM with newer `--version=` or pin older JAX.

### 1.8 Run long jobs (no 12h Colab timeout)

```bash
tmux new -s train
python3 scripts/train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
# detach: Ctrl+b d ; reattach: tmux attach -t train
```

VM preemption: TRC v2/v3 preemptible on 24h cycle. Checkpoint every N steps; resume logic required (timestamped `best_d{d}_{STAMP}_checkpoint.pkl` already in place).

### 1.9 Cost hygiene

- TPU: free under TRC.
- GCS storage: ~$0.02/GB/month — prune old checkpoints.
- Egress: free within same region. Never pull 10GB to laptop; eval results only.
- **Delete VM when idle**: `gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE`. VM persistent disk survives; recreate VM fast. Running VM consumes TRC quota hours.

### 1.10 Monitor

```bash
# TPU chip util from inside VM
pip install tpu-info
tpu-info

# GCS bucket size
gsutil du -sh gs://$BUCKET-ckpt
```

---

## 2. How to migrate files from Colab TPUs to TRC

Project already JAX/Flax + `jax[tpu]` → code stays. Migration = I/O paths + entry points.

### 2.1 File inventory

Current Colab-coupled artifacts:
- `notebooks/02_model_and_training.ipynb` — training. Has `drive.mount`, Drive pkl writes, notebook cell structure.
- `notebooks/03_evaluation.ipynb` — eval. Drive pkl reads, matplotlib inline.
- `notebooks/model.py` — **pure JAX, zero Colab deps. Port as-is.**
- `research_symmetries.py`, `tests/` — **portable.**
- `requirements.txt` — **portable.**
- `results/` — checkpoints (legacy + `best_d*_checkpoint.pkl`). Upload to GCS.

### 2.2 Replace Drive with GCS

| Colab | TRC |
|-------|-----|
| `from google.colab import drive; drive.mount('/content/drive')` | **delete** |
| `/content/drive/MyDrive/TransformerQEC/...` | `gs://$BUCKET/TransformerQEC/...` or `/home/$USER/TransformerQEC/...` |
| `pickle.dump(ckpt, open('/content/drive/.../best_d5.pkl','wb'))` | `gcsfs.GCSFileSystem().open('gs://.../best_d5.pkl','wb')` |
| `!pip install -q -r /content/drive/MyDrive/TransformerQEC/requirements.txt` | `pip install -r requirements.txt` (one-shot at VM setup) |

### 2.3 Notebooks → scripts (fastest iter on TRC)

Notebook server on TPU VM works but ssh tunneling + kernel crashes waste cycles. Convert.

```bash
jupyter nbconvert --to script notebooks/02_model_and_training.ipynb \
  --output-dir scripts/ --output train
jupyter nbconvert --to script notebooks/03_evaluation.ipynb \
  --output-dir scripts/ --output evaluate
```

Manual edits after conversion:
1. Delete `drive.mount` cell.
2. Delete `!pip install` lines (done once at VM setup).
3. Replace `get_ipython().run_line_magic(...)` calls.
4. Swap Drive paths → GCS or local.
5. Wrap main in `if __name__ == "__main__":`.
6. Argparse `--distance`, `--epochs`, `--p_values`, `--ckpt_dir`.
7. Matplotlib: `matplotlib.use('Agg')` before `pyplot` import. Save PNGs instead of `plt.show()`.

### 2.4 Upload existing Colab artifacts

From laptop (or Colab cell with `gcsfs`):
```bash
gsutil -m cp -r results/legacy gs://$BUCKET-ckpt/legacy/
gsutil -m cp results/best_d*_checkpoint.pkl gs://$BUCKET-ckpt/
gsutil -m cp -r notebooks/*.ipynb gs://$BUCKET/notebooks/
```

From TRC VM pull back what training needs:
```bash
mkdir -p ~/ckpts
gsutil -m cp gs://$BUCKET-ckpt/best_d5_*.pkl ~/ckpts/
```

### 2.5 Data pipeline: nothing to migrate

On-the-fly STIM sampling (Cycle D) means **no 6M-sample dataset to move**. `build_samplers(d, p_values)` runs on TPU VM host CPU, streams to TPU. Keep as-is.

Val set: regenerated from fixed seed → reproducible, no upload needed.

### 2.6 Checkpoint schema stays

Pickle dict unchanged:
```python
{'params': ..., 'config': {'model_version': 'dipe-no-mask-otf', 'timestamp': RUN_STAMP, ...}}
```
Eval `EXPECTED_MODEL_VERSIONS = {'dipe-no-mask', 'dipe-no-mask-otf'}` guard works unchanged.

### 2.7 Resume logic for preemption

Training script entry point:
```python
from pathlib import Path
import glob, re, pickle
ckpts = sorted(glob.glob(f'{CKPT_DIR}/best_d{D}_*_checkpoint.pkl'))
start_epoch = 0
if ckpts:
    ckpt = pickle.load(open(ckpts[-1],'rb'))
    params = ckpt['params']
    start_epoch = ckpt['config'].get('epoch', 0)
    print(f'Resumed from {ckpts[-1]} @ epoch {start_epoch}')
```
Then training loop starts at `start_epoch`. Also rsync `$CKPT_DIR` → GCS every epoch so VM preemption is recoverable from bucket.

### 2.8 Run end-to-end

```bash
# Train d=5 on TRC
tmux new -s d5
python3 scripts/train.py --distance 5 --epochs 20 \
    --ckpt_dir gs://$BUCKET-ckpt --log_dir logs/
# detach

# When done, evaluate
python3 scripts/evaluate.py --distances 3 5 7 \
    --ckpt_dir gs://$BUCKET-ckpt --out results/ler_curves.csv
gsutil cp results/ler_curves.csv gs://$BUCKET/results/
```

### 2.9 Teardown after 30 days

```bash
gsutil -m cp -r gs://$BUCKET-ckpt ./final_ckpts_backup/
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE
# Keep bucket (cheap) or: gsutil -m rm -r gs://$BUCKET*
```

---

---

## 3. Full d ∈ {3,5,7,9,11} sweep — wall-clock plan

AlphaQubit ceiling was d=11 sim. Matching it = publishable comparison axis.

### 3.1 Estimated wall-clock per run (20 epochs, OTF STIM, B=128)

| d  | v6e-8  | v4-8   | v3-8   |
|----|--------|--------|--------|
| 3  | 10 min | 30 min | 45 min |
| 5  | 30 min | 1.5 h  | 2.5 h  |
| 7  | 1.5 h  | 4 h    | 7 h    |
| 9  | 4 h    | 10 h   | 18 h   |
| 11 | 10 h   | 24 h   | 3 days |
| **Total** | **~16 h** | **~1.5 d** | **~4 d** |

All fit 30-day TRC window. v4-8 and v6e-8 leave room for re-runs, ablations, TTA eval, unified multi-d training.

### 3.2 Sweep driver

```bash
for D in 3 5 7 9 11; do
  tmux new -d -s "train_d${D}" \
    "python3 scripts/train.py --distance $D --epochs 20 \
       --batch 128 --ckpt_dir gs://$BUCKET-ckpt --log_dir logs/ \
       2>&1 | tee logs/train_d${D}_$(date +%Y%m%d_%H%M%S).log"
done
```

Run serially on single-host TPU — JAX grabs all 8 chips per process, so parallel distances compete. One at a time.

### 3.3 Circuit-level gotchas

- **STIM sampler slower at high d.** Circuit size ~d³. d=11 circuit-level sampling ~50 ms/shot single-threaded. Bump `ThreadPoolExecutor(max_workers=4)` in training cell-14.
- **Detector count grows.** Pre-allocate `coords` buffers at `L_max` once. Pad shorter-d shots; model already CLS-prefix-tolerant.
- **Val set per d.** Generate once, cache to `gs://$BUCKET/val_cache/val_d{d}_p{p}.npz`. 100k shots × L=720 × 1 byte ≈ 70 MB per p-bin — trivial GCS cost.
- **Validation stability at low p.** d=11 @ p=0.001 → LER ~1e-8 range. Need 10M+ shots per p-bin for non-zero count. Use Wilson CI (already in `03_evaluation.ipynb` cell-12) and accept zero-count bins as upper bounds.

### 3.4 Unified multi-d (DIPE payoff)

DIPE designed so one model handles all d. After per-d baselines land, train **single unified model** on mixed-d stream:

```python
# training loop — sample d uniformly per chunk
D_SCHEDULE = [3, 5, 7, 9, 11]
d = jax.random.choice(key, jnp.array(D_SCHEDULE))
samplers = build_samplers(int(d), P_VALUES)
```

Checkpoint tag: `model_version: 'dipe-no-mask-otf-unified'`. Eval: same checkpoint evaluated at each d. If LER at each d within CI of per-d baselines → DIPE claim validated, above AlphaQubit on unification axis.

---

## Quick-start checklist

- [ ] TRC email read; project ID + zone + accel type noted.
- [ ] Accel type in priority order: v6e-8 → v4-8 → v6e-4 → v5e-8 → v3-8 → v6e-1.
- [ ] `gcloud` authed; project set.
- [ ] GCS buckets created (same region as TPU).
- [ ] TPU VM provisioned, `jax.devices()` shows expected chip count.
- [ ] Repo cloned on VM; `requirements.txt` installed; `jax[tpu]` pinned; tests pass.
- [ ] Legacy checkpoints uploaded to `gs://$BUCKET-ckpt`.
- [ ] Notebooks converted to `scripts/train.py` + `scripts/evaluate.py`; Drive paths purged.
- [ ] Static pad `L_max=750` set (skips XLA recompile across d).
- [ ] Resume-from-checkpoint logic added.
- [ ] `tmux` + `tee` log pattern in use.
- [ ] Per-d baselines trained d ∈ {3,5,7,9,11}.
- [ ] Unified multi-d run launched (`dipe-no-mask-otf-unified` tag).
- [ ] Eval: Wilson-CI plots + TTA overlay for all d; CSVs pushed to `gs://$BUCKET/results/`.
- [ ] VM delete command saved for teardown day.

---

## Day-by-day 30-day budget (suggested)

| Days  | Work |
|-------|------|
| 1     | TRC setup, VM provisioned, deps installed, tests pass, d=5 smoke run |
| 2–4   | Per-d baselines d ∈ {3,5,7,9,11} sequential, full 20-epoch runs |
| 5–7   | Eval all checkpoints, Wilson-CI plots, TTA overlay, CSV export |
| 8–12  | Unified multi-d training, ablations |
| 13–18 | Calibrated Physics Penalty (queued Cycle in PLAN.md); retrain + eval |
| 19–23 | Hybrid RoPE + reflection-symmetric bias (queued FIX 12); retrain + eval |
| 24–27 | QCT input embedding + hardware adjacency masking |
| 28–29 | Final eval sweep, figures, push all artifacts to GCS + local backup |
| 30    | Teardown: `gcloud compute tpus tpu-vm delete`, keep bucket |

Buffer for preemption recovery: checkpoint every epoch, rsync to GCS every epoch, resume logic tested day 1.
