"""Baseline evaluation driver — reproduces results/evaluation_results.csv.

Loads either:
  - legacy filename `transformer_qec_d{D}.pkl` (results/legacy/, the CSV-producing checkpoints)
  - timestamped filename `best_d{D}_{YYYYMMDD_HHMMSS}_checkpoint.pkl` (from scripts/train.py)

Runs MWPM via PyMatching as the baseline decoder, then runs the
TransformerQEC checkpoint on the same syndromes. Emits CSV with Wilson
95% CIs and PNG plots. No D4xT TTA in this script — keep the baseline
reproducer minimal; layer TTA on once CSV reproduction is confirmed.

Usage:
    python -m scripts.evaluate --distances 3 5 7 \\
        --ckpt_dir gs://qec-armishra1-ckpt \\
        --out_dir  gs://qec-armishra1/results
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts._common import (  # noqa: E402
    get_detector_coords,
    make_circuit,
    sample_syndromes,
    wilson_ci,
)
from scripts.baseline_model import TransformerQEC  # noqa: E402
from scripts.checkpoint import (  # noqa: E402
    _is_gcs,
    discover_checkpoints,
    load_pickle,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline MWPM vs Transformer evaluation.")
    p.add_argument("--distances", type=int, nargs="+", required=True)
    p.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="Local or gs:// directory containing transformer_qec_d{D}.pkl "
             "or best_d{D}_*_checkpoint.pkl",
    )
    p.add_argument(
        "--out_dir", type=str, default=None,
        help="Where to write CSV/PNG/txt outputs. Default = ckpt_dir.",
    )
    p.add_argument("--num_p", type=int, default=10)
    p.add_argument("--p_min", type=float, default=0.001)
    p.add_argument("--p_max", type=float, default=0.01)
    p.add_argument("--num_test", type=int, default=100_000)
    p.add_argument("--eval_batch", type=int, default=128)
    p.add_argument(
        "--checkpoint_pins", type=str, default=None,
        help='JSON dict, e.g. "{\\"5\\":\\"legacy\\"}" to pin which ckpt per d.',
    )
    return p.parse_args()


def _save_text(text: str, path: str) -> None:
    if _is_gcs(path):
        import gcsfs
        with gcsfs.GCSFileSystem().open(path, "w") as f:
            f.write(text)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


def _save_csv(rows, fieldnames, path: str) -> None:
    if _is_gcs(path):
        import gcsfs
        with gcsfs.GCSFileSystem().open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)


def _save_fig(fig, path: str) -> None:
    if _is_gcs(path):
        import gcsfs
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            local = tmp.name
        fig.savefig(local, dpi=150, bbox_inches="tight")
        gcsfs.GCSFileSystem().put(local, path)
        os.unlink(local)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def _load_models(ckpt_dir: str, distances, pins: dict):
    by_d = discover_checkpoints(ckpt_dir)
    out = {}
    for d in distances:
        candidates = by_d.get(d, [])
        if not candidates:
            print(f"d={d}: no ckpt in {ckpt_dir}; skipping.")
            continue
        pin = pins.get(str(d))
        if pin is not None:
            match = [(s, p) for s, p in candidates if s == pin]
            if not match:
                print(f"d={d}: pinned stamp {pin!r} not found; skipping.")
                continue
            stamp, path = match[0]
        else:
            stamp, path = candidates[-1]
            if len(candidates) > 1:
                stamps = [s for s, _ in candidates]
                print(f"d={d}: {len(candidates)} candidates {stamps}; using {stamp}")

        ckpt = load_pickle(path)
        cfg = ckpt["config"]
        # Reconcile d_model with the actual param shape (guards against config drift).
        cfg["d_model"] = int(ckpt["params"]["Dense_0"]["kernel"].shape[-1])
        coords = ckpt.get("coords", get_detector_coords(d))

        model = TransformerQEC(
            d_model=cfg["d_model"],
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 4),
            ffn_dim=cfg.get("ffn_dim", 1024),
            pos_encoding=cfg.get("pos_encoding", "rope"),
        )
        out[d] = {
            "params": ckpt["params"],
            "model": model,
            "coords": coords,
            "seq_len": coords.shape[0],
            "stamp": stamp,
            "path": path,
        }
        print(
            f"d={d}: loaded stamp={stamp} d_model={cfg['d_model']} "
            f"layers={cfg.get('num_layers')} ffn={cfg.get('ffn_dim')} "
            f"seq_len={coords.shape[0]}"
        )
    return out


def _mwpm_decode(circuit, syndromes_bool):
    import pymatching
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher.decode_batch(syndromes_bool)[:, 0]


def _eval_one(d: int, info: dict, p_eval, num_test: int, eval_batch: int):
    model = info["model"]
    params = jax.device_put(info["params"])
    seq_len = info["seq_len"]
    coords_d = jax.device_put(info["coords"])

    n_padded = ((num_test + eval_batch - 1) // eval_batch) * eval_batch
    pad_n = n_padded - num_test
    n_eval_batches = n_padded // eval_batch

    @jax.jit
    def predict_all(params, syn_batched, p_batched, lab_batched, mask_batched):
        def body(correct, batch):
            syn, pe, lab, mask = batch
            preds = model.apply({"params": params}, syn, pe, coords_d).argmax(-1)
            correct = correct + jnp.where(mask, preds == lab, False).sum()
            return correct, None
        correct, _ = jax.lax.scan(
            body, jnp.int32(0),
            (syn_batched, p_batched, lab_batched, mask_batched),
        )
        return correct

    out = {}
    print(f"d={d}: eval_batch={eval_batch} n_eval_batches={n_eval_batches} pad={pad_n}")
    for p in p_eval:
        circuit = make_circuit(d, p)
        syndromes, labels = sample_syndromes(circuit, num_test)

        mwpm_preds = _mwpm_decode(circuit, syndromes.astype(np.bool_))
        mwpm_failures = int((mwpm_preds != labels).sum())
        mwpm_ler = mwpm_failures / num_test
        mwpm_lo, mwpm_hi = wilson_ci(mwpm_failures, num_test)

        mask = np.ones(n_padded, dtype=np.bool_)
        p_arr = np.full(n_padded, p, dtype=np.float32)
        if pad_n:
            syndromes = np.concatenate(
                [syndromes, np.zeros((pad_n, seq_len), dtype=np.float32)]
            )
            labels = np.concatenate(
                [labels, np.zeros(pad_n, dtype=labels.dtype)]
            )
            mask[num_test:] = False
            p_arr[num_test:] = 0.0

        syn_d = jax.device_put(syndromes.reshape(n_eval_batches, eval_batch, seq_len))
        p_d = jax.device_put(p_arr.reshape(n_eval_batches, eval_batch))
        lab_d = jax.device_put(labels.reshape(n_eval_batches, eval_batch))
        mask_d = jax.device_put(mask.reshape(n_eval_batches, eval_batch))

        correct = int(predict_all(params, syn_d, p_d, lab_d, mask_d))
        tf_failures = num_test - correct
        tf_ler = tf_failures / num_test
        tf_lo, tf_hi = wilson_ci(tf_failures, num_test)

        del syn_d, p_d, lab_d, mask_d

        out[(d, p)] = {
            "mwpm_ler": mwpm_ler,
            "mwpm_ci_lo": mwpm_lo,
            "mwpm_ci_hi": mwpm_hi,
            "mwpm_failures": mwpm_failures,
            "transformer_ler": tf_ler,
            "tf_ci_lo": tf_lo,
            "tf_ci_hi": tf_hi,
            "tf_failures": tf_failures,
            "num_test": num_test,
        }
        print(
            f"d={d} p={p:.4f} | MWPM={mwpm_ler:.6f} [{mwpm_lo:.6f},{mwpm_hi:.6f}] "
            f"| Transformer={tf_ler:.6f} [{tf_lo:.6f},{tf_hi:.6f}]"
        )
    return out


def _plot_per_decoder(results, distances, p_eval, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for ax, (key, lo_key, hi_key, title) in zip(
        axes,
        [
            ("mwpm_ler", "mwpm_ci_lo", "mwpm_ci_hi", "MWPM Decoder"),
            ("transformer_ler", "tf_ci_lo", "tf_ci_hi", "Transformer Decoder"),
        ],
    ):
        for d in distances:
            ps = [p for p in p_eval if (d, p) in results]
            lers = [results[(d, p)][key] for p in ps]
            los = [results[(d, p)][lo_key] for p in ps]
            his = [results[(d, p)][hi_key] for p in ps]
            keep = [v > 0 for v in lers]
            ps_k = [pi for pi, k in zip(ps, keep) if k]
            le_k = [li for li, k in zip(lers, keep) if k]
            lo_k = [max(li, 1e-12) for li, k in zip(los, keep) if k]
            hi_k = [hi for hi, k in zip(his, keep) if k]
            line, = ax.plot(ps_k, le_k, "o-", label=f"d={d}", markersize=5)
            ax.fill_between(ps_k, lo_k, hi_k, alpha=0.2, color=line.get_color(), linewidth=0)
        ax.set(
            xlabel="Physical error rate (p)",
            ylabel="Logical error rate",
            xscale="log", yscale="log",
            title=title,
        )
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def _plot_combined(results, distances, p_eval, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = {3: "C0", 5: "C1", 7: "C2", 9: "C3", 11: "C4"}

    def nz(ps, vals, los, his):
        out = [(p, v, l, h) for p, v, l, h in zip(ps, vals, los, his) if v > 0]
        if not out:
            return [], [], [], []
        ps_k, vs_k, ls_k, hs_k = zip(*out)
        return list(ps_k), list(vs_k), [max(l, 1e-12) for l in ls_k], list(hs_k)

    for d in distances:
        ps = [p for p in p_eval if (d, p) in results]
        mw = [results[(d, p)]["mwpm_ler"] for p in ps]
        mlo = [results[(d, p)]["mwpm_ci_lo"] for p in ps]
        mhi = [results[(d, p)]["mwpm_ci_hi"] for p in ps]
        tf = [results[(d, p)]["transformer_ler"] for p in ps]
        tlo = [results[(d, p)]["tf_ci_lo"] for p in ps]
        thi = [results[(d, p)]["tf_ci_hi"] for p in ps]

        pm, lm, ml, mh = nz(ps, mw, mlo, mhi)
        if pm:
            ax.plot(pm, lm, "s--", color=colors.get(d, "C5"), alpha=0.7, label=f"MWPM d={d}")
            ax.fill_between(pm, ml, mh, alpha=0.12, color=colors.get(d, "C5"), linewidth=0)
        pt, lt, tl, th = nz(ps, tf, tlo, thi)
        if pt:
            ax.plot(pt, lt, "o-", color=colors.get(d, "C5"), label=f"Transformer d={d}")
            ax.fill_between(pt, tl, th, alpha=0.2, color=colors.get(d, "C5"), linewidth=0)

    ax.set(
        xlabel="Physical error rate (p)",
        ylabel="Logical error rate",
        xscale="log", yscale="log",
        title="Transformer vs MWPM (shaded = 95% Wilson CI)",
    )
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)


def _threshold_crossing(results, p_eval, key, d1, d2):
    ps, l1, l2 = [], [], []
    for p in p_eval:
        if (d1, p) in results and (d2, p) in results:
            a = results[(d1, p)][key]
            b = results[(d2, p)][key]
            if a > 0 and b > 0:
                ps.append(p); l1.append(a); l2.append(b)
    if len(ps) < 2:
        return None
    log_ps = np.log(ps)
    diff = np.log(l2) - np.log(l1)
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            t = diff[i] / (diff[i] - diff[i + 1])
            return float(np.exp(log_ps[i] + t * (log_ps[i + 1] - log_ps[i])))
    return None


def main() -> None:
    import json
    args = _parse_args()
    pins = json.loads(args.checkpoint_pins) if args.checkpoint_pins else {}
    print(f"JAX backend: {jax.default_backend()}")

    out_dir = (args.out_dir or args.ckpt_dir).rstrip("/")
    p_eval = np.geomspace(args.p_min, args.p_max, args.num_p).tolist()

    models = _load_models(args.ckpt_dir, args.distances, pins)
    if not models:
        raise SystemExit("No checkpoints loaded; aborting.")

    results = {}
    for d in args.distances:
        if d not in models:
            continue
        jax.clear_caches()
        gc.collect()
        results.update(_eval_one(d, models[d], p_eval, args.num_test, args.eval_batch))

    rows = []
    for d in args.distances:
        for p in p_eval:
            if (d, p) not in results:
                continue
            r = results[(d, p)]
            improvement = (
                (r["mwpm_ler"] - r["transformer_ler"]) / r["mwpm_ler"] * 100
                if r["mwpm_ler"] > 0
                else float("nan")
            )
            rows.append({
                "d": d, "p": p, "num_test": r["num_test"],
                "mwpm_failures": r["mwpm_failures"],
                "mwpm_ler": r["mwpm_ler"],
                "mwpm_ci_lo": r["mwpm_ci_lo"], "mwpm_ci_hi": r["mwpm_ci_hi"],
                "tf_failures": r["tf_failures"],
                "transformer_ler": r["transformer_ler"],
                "tf_ci_lo": r["tf_ci_lo"], "tf_ci_hi": r["tf_ci_hi"],
                "improvement_pct": improvement,
            })

    fieldnames = [
        "d", "p", "num_test",
        "mwpm_failures", "mwpm_ler", "mwpm_ci_lo", "mwpm_ci_hi",
        "tf_failures", "transformer_ler", "tf_ci_lo", "tf_ci_hi",
        "improvement_pct",
    ]
    csv_path = f"{out_dir}/evaluation_results.csv"
    _save_csv(rows, fieldnames, csv_path)
    print(f"Saved CSV -> {csv_path}")

    _plot_per_decoder(results, args.distances, p_eval, f"{out_dir}/logical_error_rates.png")
    print(f"Saved -> {out_dir}/logical_error_rates.png")
    _plot_combined(results, args.distances, p_eval, f"{out_dir}/transformer_vs_mwpm.png")
    print(f"Saved -> {out_dir}/transformer_vs_mwpm.png")

    present = [d for d in args.distances if d in models]
    pairs = list(zip(present[:-1], present[1:]))
    lines = ["Threshold estimates (from curve crossings):", "=" * 55]
    for key, name in [("mwpm_ler", "MWPM"), ("transformer_ler", "Transformer")]:
        for d1, d2 in pairs:
            p_th = _threshold_crossing(results, p_eval, key, d1, d2)
            if p_th:
                lines.append(f"{name:>12} d={d1}/{d2} crossing: p_th ~ {p_th:.4f}")
            else:
                lines.append(
                    f"{name:>12} d={d1}/{d2} crossing: not found in evaluated range"
                )
    print("\n".join(lines))
    _save_text("\n".join(lines) + "\n", f"{out_dir}/threshold_estimates.txt")
    print(f"Saved -> {out_dir}/threshold_estimates.txt")


if __name__ == "__main__":
    main()
