"""Stress test for notebook 02 and 03 compatibility.

Exercises: data generation, model init, forward/backward pass,
checkpoint save/load, and evaluation inference — all at d=3 (fast).
"""
import sys, os, json, pickle, io, tempfile
import numpy as np

# ---------------------------------------------------------------------------
# Extract source code from notebook cells
# ---------------------------------------------------------------------------
NB_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

def get_cell_sources(notebook_name):
    path = os.path.join(NB_DIR, notebook_name)
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    return [''.join(c['source']) for c in nb['cells']]

nb02 = get_cell_sources('02_model_and_training.ipynb')
nb03 = get_cell_sources('03_evaluation.ipynb')

# ---------------------------------------------------------------------------
# Execute notebook 02 cells in order (skip pip install, plots, base64 embed)
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 1: Notebook 02 — imports, data gen, model, training setup")
print("=" * 60)

# Cell 2: imports
exec(nb02[2])
print("[OK] Imports")

# Cell 4: data generation functions (make_circuit, sample_syndromes, get_detector_coords, generate_dataset)
exec(nb02[4])
print("[OK] Data generation functions defined")

# Cell 6: model classes
exec(nb02[6])
print("[OK] Model classes defined")

# Cell 8: training setup functions
exec(nb02[8])
print("[OK] Training setup functions defined")

# ---------------------------------------------------------------------------
# Test data generation for all distances
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 2: Data generation for d=3, d=5, d=7")
print("=" * 60)

for d in [3, 5, 7]:
    syn, lab, pe, coords = generate_dataset(d, [0.05], 100)
    assert syn.shape[0] == 100, f"d={d}: wrong sample count {syn.shape[0]}"
    assert syn.shape[1] == coords.shape[0], (
        f"d={d}: seq_len mismatch syn={syn.shape[1]} vs coords={coords.shape[0]}")
    assert coords.shape[1] == 3, f"d={d}: coords should have 3 columns"
    assert coords.min() >= 0.0 and coords.max() <= 1.0, (
        f"d={d}: coords not normalized [{coords.min()}, {coords.max()}]")
    print(f"  d={d}: syn={syn.shape}, coords={coords.shape}, "
          f"coord_range=[{coords.min():.3f}, {coords.max():.3f}] [OK]")

# Verify coords are independent of p
c1 = get_detector_coords(3)
c2 = get_detector_coords(3)
assert np.allclose(c1, c2), "Coords not deterministic!"
print("  Coords deterministic across calls [OK]")

# ---------------------------------------------------------------------------
# Test model init + forward pass for all distances
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 3: Model init + forward pass for d=3, d=5, d=7")
print("=" * 60)

for d in [3, 5, 7]:
    syn, lab, pe, coords = generate_dataset(d, [0.05], 16)
    seq_len = syn.shape[1]
    coords_d = jax.device_put(coords)

    model = TransformerQEC()
    key = jax.random.PRNGKey(0)

    # Init
    dummy_syn = jnp.zeros((1, seq_len))
    dummy_p = jnp.zeros((1,))
    variables = model.init(key, dummy_syn, dummy_p, coords_d)
    params = variables['params']
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))

    # Forward pass
    syn_d = jax.device_put(syn)
    pe_d = jax.device_put(pe)
    logits = model.apply({'params': params}, syn_d, pe_d, coords_d)

    assert logits.shape == (16, 2), f"d={d}: wrong logits shape {logits.shape}"
    assert not np.any(np.isnan(np.array(logits))), f"d={d}: NaN in logits"
    print(f"  d={d}: params={param_count:,}, logits={logits.shape}, "
          f"no NaN [OK]")

# ---------------------------------------------------------------------------
# Test gradient computation (training step)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 4: Gradient computation (d=3)")
print("=" * 60)

d = 3
syn, lab, pe, coords = generate_dataset(d, [0.05], 32)
seq_len = syn.shape[1]
coords_d = jax.device_put(coords)
model = TransformerQEC()
key = jax.random.PRNGKey(42)

state = create_train_state(key, model, seq_len, coords_d, num_steps=100)

syn_d = jax.device_put(syn)
lab_d = jax.device_put(lab)
pe_d = jax.device_put(pe)

def loss_fn(params):
    logits = model.apply({'params': params}, syn_d, pe_d, coords_d)
    return cross_entropy_loss(logits, lab_d), logits

(loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
loss_val = float(loss)
assert not np.isnan(loss_val), "NaN loss"
assert loss_val > 0, "Loss should be positive"

# Check grads are non-zero
grad_norms = {k: float(jnp.linalg.norm(jax.tree_util.tree_leaves(v)[0]))
              for k, v in grads.items() if jax.tree_util.tree_leaves(v)}
assert all(n > 0 for n in grad_norms.values()), "Some gradients are zero"
print(f"  Loss={loss_val:.4f}, grad norms: {grad_norms}")
print("  [OK] Gradients flow correctly")

# Apply gradients
new_state = state.apply_gradients(grads=grads)
assert new_state.step == 1, "Step not incremented"
print("  [OK] Optimizer step works")

# ---------------------------------------------------------------------------
# Test make_epoch_fns (scan-based training + eval)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 5: Scan-based epoch functions (d=3)")
print("=" * 60)

d = 3
BATCH_SIZE = 16
syn, lab, pe, coords = generate_dataset(d, [0.05], 64)
seq_len = syn.shape[1]
coords_d = jax.device_put(coords)
model = TransformerQEC()
key = jax.random.PRNGKey(42)

state = create_train_state(key, model, seq_len, coords_d, num_steps=100)

n_train = (len(syn) // BATCH_SIZE) * BATCH_SIZE
train_syn_d = jax.device_put(syn[:n_train])
train_lab_d = jax.device_put(lab[:n_train])
train_pe_d = jax.device_put(pe[:n_train])

train_epoch, eval_epoch = make_epoch_fns(
    model, coords_d, train_syn_d, train_lab_d, train_pe_d)

# Train epoch
n_batches = n_train // BATCH_SIZE
perm = jax.random.permutation(jax.random.PRNGKey(0), n_train)
index_batches = perm.reshape(n_batches, BATCH_SIZE)

state, train_loss, train_acc = train_epoch(state, index_batches)
print(f"  Train: loss={float(train_loss):.4f}, acc={float(train_acc):.4f}")
assert not np.isnan(float(train_loss)), "NaN train loss"

# Eval epoch
val_batches = (
    jax.device_put(syn[:n_train].reshape(n_batches, BATCH_SIZE, seq_len)),
    jax.device_put(lab[:n_train].reshape(n_batches, BATCH_SIZE)),
    jax.device_put(pe[:n_train].reshape(n_batches, BATCH_SIZE)),
)
val_loss, val_acc = eval_epoch(state.params, val_batches)
print(f"  Val:   loss={float(val_loss):.4f}, acc={float(val_acc):.4f}")
assert not np.isnan(float(val_loss)), "NaN val loss"
print("  [OK] Scan-based epoch functions work")

# ---------------------------------------------------------------------------
# Test checkpoint save/load cycle
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 6: Checkpoint save/load cycle")
print("=" * 60)

checkpoint = {
    'params': jax.device_get(state.params),
    'config': {
        'distance': d,
        'seq_len': seq_len,
        'd_model': model.d_model,
        'num_heads': model.num_heads,
        'num_layers': model.num_layers,
        'ffn_dim': model.ffn_dim,
    },
    'coords': coords,
}

# Save and reload
buf = io.BytesIO()
pickle.dump(checkpoint, buf)
raw = buf.getvalue()
print(f"  Checkpoint size: {len(raw):,} bytes")

buf2 = io.BytesIO(raw)
loaded = pickle.load(buf2)

assert 'coords' in loaded, "coords missing from checkpoint"
assert loaded['config']['distance'] == d
assert np.allclose(loaded['coords'], coords), "coords mismatch after load"
assert 'max_seq_len' not in loaded['config'], "max_seq_len should not be in config"
print("  [OK] Checkpoint save/load preserves all fields")

# ---------------------------------------------------------------------------
# Test notebook 03 compatibility: load checkpoint and run inference
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 7: Notebook 03 — checkpoint loading + inference")
print("=" * 60)

# Save checkpoint to temp file
with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
    pickle.dump(checkpoint, f)
    tmp_path = f.name

# Simulate notebook 03 checkpoint loading (cell 9 logic)
with open(tmp_path, 'rb') as f:
    ckpt = pickle.load(f)

cfg = ckpt['config']
sl = cfg['seq_len']

if 'coords' in ckpt:
    eval_coords = ckpt['coords']
else:
    eval_coords = get_detector_coords(cfg['distance'])

eval_model = TransformerQEC(
    d_model=cfg.get('d_model', 64),
    num_heads=cfg.get('num_heads', 4),
    num_layers=cfg.get('num_layers', 4),
    ffn_dim=cfg.get('ffn_dim', 256))

print(f"  Loaded: d={cfg['distance']}, seq_len={sl}, coords={eval_coords.shape}")

# Simulate notebook 03 inference (cell 11 logic)
eval_params = jax.device_put(ckpt['params'])
eval_coords_d = jax.device_put(eval_coords)

# Generate test data
test_circuit = make_circuit(d, 0.05)
test_syn, test_lab = sample_syndromes(test_circuit, 128)

test_syn_d = jax.device_put(test_syn)
test_pe_d = jax.device_put(np.full(128, 0.05, dtype=np.float32))

# Forward pass with loaded checkpoint
eval_logits = eval_model.apply(
    {'params': eval_params}, test_syn_d, test_pe_d, eval_coords_d)
preds = eval_logits.argmax(-1)

assert eval_logits.shape == (128, 2), f"Wrong eval logits shape: {eval_logits.shape}"
assert not np.any(np.isnan(np.array(eval_logits))), "NaN in eval logits"
acc = float((preds == jax.device_put(test_lab)).mean())
print(f"  Inference: logits={eval_logits.shape}, acc={acc:.4f}")
print("  [OK] Notebook 03 inference works with notebook 02 checkpoints")

# Simulate scan-based predict_all from cell 11
eval_batch = 64
n_eval_batches = 128 // eval_batch

@jax.jit
def predict_all(params, syn_batched, p_batched, lab_batched, mask_batched):
    def body(correct, batch):
        syn, pe, lab, mask = batch
        preds = eval_model.apply(
            {'params': params}, syn, pe, eval_coords_d).argmax(-1)
        correct = correct + jnp.where(mask, preds == lab, False).sum()
        return correct, None
    correct, _ = jax.lax.scan(
        body, jnp.int32(0),
        (syn_batched, p_batched, lab_batched, mask_batched))
    return correct

syn_batched = jax.device_put(test_syn.reshape(n_eval_batches, eval_batch, sl))
p_batched = jax.device_put(
    np.full(128, 0.05, dtype=np.float32).reshape(n_eval_batches, eval_batch))
lab_batched = jax.device_put(test_lab.reshape(n_eval_batches, eval_batch))
mask_batched = jax.device_put(
    np.ones(128, dtype=np.bool_).reshape(n_eval_batches, eval_batch))

correct = int(predict_all(eval_params, syn_batched, p_batched, lab_batched, mask_batched))
scan_acc = correct / 128
print(f"  Scan inference: correct={correct}/128, acc={scan_acc:.4f}")
assert abs(scan_acc - acc) < 1e-6, f"Scan acc {scan_acc} != direct acc {acc}"
print("  [OK] Scan-based predict_all matches direct inference")

# ---------------------------------------------------------------------------
# Test cross-distance: train d=3, eval d=5 (different coords, no crash)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 8: Cross-distance model instantiation (d=3, d=5, d=7)")
print("=" * 60)

for d in [3, 5, 7]:
    syn, lab, pe, coords = generate_dataset(d, [0.05], 8)
    sl = syn.shape[1]
    cd = jax.device_put(coords)
    m = TransformerQEC()
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, sl)), jnp.zeros((1,)), cd)['params']
    out = m.apply({'params': p}, jax.device_put(syn), jax.device_put(pe), cd)
    assert out.shape == (8, 2)
    print(f"  d={d}: seq_len={sl}, forward pass [OK]")

# Cleanup
os.unlink(tmp_path)

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)