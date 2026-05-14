"""Baseline reproduction training driver (cc3abc5 architecture).

Bulk-materialized dataset, normalized [0,1] coords + seq_len-scaled RoPE,
cross-entropy loss, no DIPE, no locality mask, no nn.remat. Defaults
match the per-distance configs that produced results/legacy/*.pkl:

    d=3:  d_model=128 num_heads=4 num_layers=4 ffn_dim=1024
    d=5:  d_model=128 num_heads=4 num_layers=6 ffn_dim=512
    d=7:  d_model=128 num_heads=4 num_layers=6 ffn_dim=512

Usage:
    python -m scripts.train --distance 5 \\
        --ckpt_dir gs://qec-armishra1-ckpt \\
        --epochs 13 --total_samples 6000000
"""
from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from scripts._common import (  # noqa: E402
    cross_entropy_loss,
    generate_dataset,
)
from scripts.baseline_model import LEGACY_CONFIGS, TransformerQEC  # noqa: E402
from scripts.checkpoint import (  # noqa: E402
    delete_path,
    load_pickle,
    pick_latest,
    save_pickle,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline TransformerQEC training (reproduces commit cc3abc5)."
    )
    p.add_argument("--distance", type=int, required=True)
    # Defaults filled per-distance from LEGACY_CONFIGS after parse — keep
    # `None` here so explicit user flags override the per-distance default.
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--ffn_dim", type=int, default=None)
    p.add_argument("--num_p", type=int, default=20)
    p.add_argument("--p_min", type=float, default=0.002)
    p.add_argument("--p_max", type=float, default=0.017)
    p.add_argument("--total_samples", type=int, default=6_000_000)
    p.add_argument("--val_ratio", type=int, default=20,
                   help="Validation samples per p = SHOTS_PER_P / val_ratio.")
    p.add_argument("--epochs", type=int, default=13)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--peak_lr", type=float, default=1.5e-4)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ckpt_dir",
        type=str,
        default="./results",
        help="Local path or gs:// URI to write checkpoints into.",
    )
    p.add_argument(
        "--ckpt_name",
        type=str,
        default=None,
        help=(
            "Output checkpoint filename (no path). Defaults to legacy form "
            "'transformer_qec_d{D}.pkl' to drop in next to the existing pkls."
        ),
    )
    p.add_argument("--resume", action="store_true",
                   help="Resume params from latest ckpt in --ckpt_dir.")
    p.add_argument("--save_intermediate_every", type=int, default=3,
                   help="Safety ckpt every N epochs (0 = disable).")
    p.add_argument("--curves_png", type=str, default=None)
    return p.parse_args()


def _resolve_config(args) -> tuple[int, int, int, int]:
    defaults = LEGACY_CONFIGS.get(args.distance, LEGACY_CONFIGS[3])
    d_model = args.d_model if args.d_model is not None else defaults["d_model"]
    num_heads = args.num_heads if args.num_heads is not None else defaults["num_heads"]
    num_layers = args.num_layers if args.num_layers is not None else defaults["num_layers"]
    ffn_dim = args.ffn_dim if args.ffn_dim is not None else defaults["ffn_dim"]
    return d_model, num_heads, num_layers, ffn_dim


def _create_train_state(model, key, seq_len: int, coords_d,
                        num_steps: int, warmup_steps: int, peak_lr: float):
    dummy_syn = jnp.zeros((1, seq_len))
    dummy_p = jnp.zeros((1,))
    params = model.init(key, dummy_syn, dummy_p, coords_d)["params"]
    warmup_steps = min(warmup_steps, max(1, num_steps // 5))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
        end_value=1e-6,
    )
    muon_adam_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr * 0.2,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
        end_value=2e-7,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.contrib.muon(
            learning_rate=schedule,
            adam_learning_rate=muon_adam_schedule,
            weight_decay=0.1,
        ),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _make_epoch_fns(model, coords_d):

    def _train_body(state, batch_indices, train_syn_d, train_lab_d, train_pe_d):
        syn = train_syn_d[batch_indices]
        lab = train_lab_d[batch_indices]
        pe = train_pe_d[batch_indices]

        def loss_fn(params):
            logits = model.apply({"params": params}, syn, pe, coords_d)
            return cross_entropy_loss(logits, lab), logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        acc = (logits.argmax(-1) == lab).mean()
        return state, (loss, acc)

    @jax.jit
    def train_epoch(state, index_batches, train_syn_d, train_lab_d, train_pe_d):
        def body(state, idx):
            return _train_body(state, idx, train_syn_d, train_lab_d, train_pe_d)
        state, (losses, accs) = jax.lax.scan(body, state, index_batches)
        return state, losses.mean(), accs.mean()

    @jax.jit
    def eval_epoch(params, batches):
        def body(metrics, batch):
            loss_sum, acc_sum = metrics
            syn, lab, pe = batch
            logits = model.apply({"params": params}, syn, pe, coords_d)
            loss = cross_entropy_loss(logits, lab)
            acc = (logits.argmax(-1) == lab).mean()
            return (loss_sum + loss, acc_sum + acc), None

        n = batches[0].shape[0]
        (total_loss, total_acc), _ = jax.lax.scan(
            body, (jnp.float32(0.0), jnp.float32(0.0)), batches
        )
        return total_loss / n, total_acc / n

    return train_epoch, eval_epoch


def main() -> None:
    args = _parse_args()
    d_model, num_heads, num_layers, ffn_dim = _resolve_config(args)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(
        f"d={args.distance} | model: d_model={d_model} heads={num_heads} "
        f"layers={num_layers} ffn={ffn_dim}"
    )

    p_train = np.geomspace(args.p_min, args.p_max, args.num_p).tolist()
    shots_per_p = args.total_samples // args.num_p
    val_shots_per_p = max(1, shots_per_p // args.val_ratio)
    print(f"p_train ({args.num_p}): {[f'{p:.4f}' for p in p_train]}")
    print(
        f"shots_per_p train={shots_per_p:,} val={val_shots_per_p:,} "
        f"(total train={shots_per_p * args.num_p:,})"
    )

    print("Generating training data...")
    train_syn, train_lab, train_p, coords = generate_dataset(
        args.distance, p_train, shots_per_p
    )
    print("Generating validation data...")
    val_syn, val_lab, val_p, _ = generate_dataset(
        args.distance, p_train, val_shots_per_p
    )
    seq_len = train_syn.shape[1]
    print(
        f"train: {train_syn.shape[0]:,} samples seq_len={seq_len} "
        f"pos_rate={train_lab.mean():.4f}"
    )
    print(
        f"val:   {val_syn.shape[0]:,} samples pos_rate={val_lab.mean():.4f}"
    )

    model = TransformerQEC(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        dtype=jnp.bfloat16,
        pos_encoding="rope",
    )
    coords_d = jax.device_put(coords)

    steps_per_epoch = len(train_syn) // args.batch_size
    num_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    key = jax.random.PRNGKey(args.seed)

    state = _create_train_state(
        model, key, seq_len, coords_d, num_steps, warmup_steps, args.peak_lr
    )
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f"params: {param_count:,}")
    print(f"steps/epoch={steps_per_epoch} warmup_steps={warmup_steps} total={num_steps:,}")

    if args.resume:
        hit = pick_latest(args.ckpt_dir, args.distance)
        if hit is None:
            print(f"--resume: no ckpt in {args.ckpt_dir} for d={args.distance}; fresh start.")
        else:
            stamp, path = hit
            ckpt = load_pickle(path)
            params = jax.tree.map(jnp.asarray, ckpt["params"])
            state = state.replace(params=params)
            print(f"Resumed params from {path} (stamp={stamp}).")

    # Pre-batch val for scan
    n_train = (len(train_syn) // args.batch_size) * args.batch_size
    n_val = (len(val_syn) // args.batch_size) * args.batch_size
    n_train_batches = n_train // args.batch_size
    n_val_batches = n_val // args.batch_size

    train_syn_d = jax.device_put(train_syn[:n_train])
    train_lab_d = jax.device_put(train_lab[:n_train])
    train_p_d = jax.device_put(jnp.asarray(train_p[:n_train], dtype=jnp.bfloat16))

    val_batches = (
        jax.device_put(val_syn[:n_val]).reshape(n_val_batches, args.batch_size, seq_len),
        jax.device_put(val_lab[:n_val]).reshape(n_val_batches, args.batch_size),
        jax.device_put(jnp.asarray(val_p[:n_val], dtype=jnp.bfloat16)).reshape(
            n_val_batches, args.batch_size
        ),
    )
    print(f"{n_train_batches} train batches/epoch, {n_val_batches} val batches/epoch")

    train_epoch, eval_epoch = _make_epoch_fns(model, coords_d)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_loss = float("inf")
    best_params = None
    best_epoch = 0
    best_val_acc = 0.0

    base_cfg = {
        "distance": args.distance,
        "seq_len": seq_len,
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "ffn_dim": model.ffn_dim,
        "pos_encoding": model.pos_encoding,
    }

    train_key = jax.random.PRNGKey(args.seed + 1)
    prev_safety_path = None
    t0 = time.time()

    for epoch in range(args.epochs):
        train_key, subkey = jax.random.split(train_key)
        perm = jax.random.permutation(subkey, n_train)
        index_batches = perm.reshape(n_train_batches, args.batch_size)

        state, train_loss, train_acc = train_epoch(
            state, index_batches, train_syn_d, train_lab_d, train_p_d
        )
        val_loss, val_acc = eval_epoch(state.params, val_batches)
        val_acc.block_until_ready()
        elapsed = time.time() - t0

        train_losses.append(float(train_loss))
        train_accs.append(float(train_acc))
        val_losses.append(float(val_loss))
        val_accs.append(float(val_acc))

        tag = " [incl. JIT compilation]" if epoch == 0 else ""
        print(
            f"Epoch {epoch + 1:2d}/{args.epochs} | "
            f"train loss={train_losses[-1]:.4f} acc={train_accs[-1]:.4f} | "
            f"val loss={val_losses[-1]:.4f} acc={val_accs[-1]:.4f} | "
            f"{elapsed:.1f}s{tag}",
            flush=True,
        )

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_val_acc = val_accs[-1]
            best_epoch = epoch + 1
            best_params = jax.tree.map(lambda x: x.copy(), state.params)
            print(
                f"  ^ New best (epoch {best_epoch}, val_loss={best_val_loss:.6f}, "
                f"val_acc={best_val_acc:.4f})"
            )

        if args.save_intermediate_every and (
            (epoch + 1) % args.save_intermediate_every == 0
        ):
            safety_ckpt = {
                "params": jax.device_get(state.params),
                "config": base_cfg,
                "coords": coords,
            }
            safety_path = (
                f"{args.ckpt_dir.rstrip('/')}/safety_d{args.distance}_epoch{epoch + 1}.pkl"
            )
            save_pickle(safety_ckpt, safety_path)
            if prev_safety_path:
                delete_path(prev_safety_path)
            prev_safety_path = safety_path
            print(f"  Safety ckpt saved: {safety_path}")

    del train_syn_d, train_lab_d, train_p_d

    final_name = args.ckpt_name or f"transformer_qec_d{args.distance}.pkl"
    best_checkpoint = {
        "params": jax.device_get(best_params),
        "config": base_cfg,
        "coords": coords,
        "epoch": best_epoch,
        "val_loss": float(best_val_loss),
        "val_acc": float(best_val_acc),
    }
    best_path = f"{args.ckpt_dir.rstrip('/')}/{final_name}"
    save_pickle(best_checkpoint, best_path)
    print(f"\nBest model (epoch {best_epoch}) -> {best_path}")

    curves_target = args.curves_png or os.path.join(
        ".", f"train_curves_d{args.distance}.png"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    epochs = range(1, args.epochs + 1)
    ax1.plot(epochs, train_losses, "b-o", label="Train", markersize=4)
    ax1.plot(epochs, val_losses, "r-o", label="Val", markersize=4)
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_accs, "b-o", label="Train", markersize=4)
    ax2.plot(epochs, val_accs, "r-o", label="Val", markersize=4)
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Classification Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(curves_target, dpi=150, bbox_inches="tight")
    print(f"Saved curves -> {curves_target}")


if __name__ == "__main__":
    main()
