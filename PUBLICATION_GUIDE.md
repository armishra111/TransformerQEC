# Publication Guide — TransformerQEC

Combined pre-submission review from two independent agents:
1. **qec-decoder-paper-reviewer** — broad venue-fit review (Nature / Quantum / NeurIPS / PRX Quantum)
2. **prx-qec-ml-reviewer** — Physical Review family specifically (PRX / PRX Quantum / PR Research / PRA / PRL)

Both reviewers assumed CRITICAL fixes (FIX 1 DIPE, FIX 2 graph-masked soft physics penalty, FIX 3 physics-invariance tests) from `DEEP_RESEARCH_FIX_SUGGS.md` have been executed successfully and the $d=7$ collapse is closed.

---

## 1. TL;DR Verdict

> **Assuming CRITICAL fixes land and close the $d=7$ collapse, the work is a strong workshop-tier submission (ML4PS / QIP poster / ICLR Physics4ML) but is NOT yet publishable in any Physical Review venue or top-tier journal.** The realistic path to a PR-family publication is *Physical Review Research* after landing FIX 4, 6, 7 and fixing statistical rigor. *PRX Quantum* is a ~6-month stretch that requires circuit-level noise. *PRX*, *PRL*, and *Nature* are not on the trajectory.

Both reviewers independently converged on the same target venue: **Physical Review Research** (or the *Quantum* journal as a parallel option), framed as "Distance-invariant neural decoding: a physics-informed inductive-bias stack for parameter-efficient transformer decoders."

---

## 2. Per-Venue Assessment

| Venue | Verdict | Primary Blocker |
|---|---|---|
| **Nature** | Not on trajectory | Phenomenological-only noise; novelty budget too small against AlphaQubit; no hardware validation |
| **PRX** | No | Broad-impact novelty bar unmet; architectural delta too incremental |
| **PRX Quantum** | Not yet (~6 months work) | Phenomenological-only noise is a dealbreaker vs. AlphaQubit's circuit-level SI1000 + Sycamore benchmark; needs FIX 7 + one circuit-level ablation |
| **PRL** | Doesn't fit | ~4-page format cannot contain the story; not a PRL-scope physics claim |
| **Physical Review Research** | Plausible target after FIX 1–3, 4, 6, 7 + statistical rigor | Needs shot-count disclosure, Wilson CIs, error bars on thresholds, FIX 7 (unified backbone) |
| **Physical Review A** | Weak fit | Scope skews atomic/quantum optics; decoder-architecture papers usually go to PRX Quantum or Quantum |
| **Quantum (journal)** | Plausible after FIX 1–3, 4, 6, 7, 8 + rigor | Parallel option to PR Research; more tolerant of phenomenological-only framing if scope is stated explicitly |
| **NeurIPS 2026 (main)** | Uphill but possible if FIX 7 lands cleanly | Risk: "incremental RoPE variant" reviewer; mitigate by leading with unified multi-distance backbone |
| **NeurIPS ML4PS / ICLR Physics4ML / QIP poster** | Strong fit now (after CRITICAL fixes) | No additional blockers |

---

## 3. Critical Errors (Must Fix Before Any Submission)

These are bugs and rigor gaps found in the current repository that are **not** in the existing `DEEP_RESEARCH_FIX_SUGGS.md` fix list. A PR referee will find each of these in minutes.

### ERROR 1: "Exploiting Y-Errors" Claim May Be Factually Wrong
**Location:** `README.md §4 Insights & Future Directions`
> "The Transformer outperforms MWPM below threshold by learning correlated defect signatures of $Y$ errors, which standard MWPM strictly treats as independent $X$ and $Z$ defect pairs."

**Problem:** Under *pure* phenomenological noise (independent $X$ and $Z$ channels on data qubits + independent measurement flips), there is no $Y$-channel content in the noise model at all. If the STIM circuit is generated without depolarizing or Y-channel noise, this claim is *empirically impossible* — the model cannot exploit correlations that are not present in the data.

**Action:**
- **Verify** the noise model in `02_model_and_training.ipynb:make_circuit`. Check whether `after_clifford_depolarization` or a Y-channel term is present.
- If **no** Y-channel content: delete this claim from the README. Replace with an honest statement about what the transformer *is* learning (e.g., spatial correlations in measurement errors, hook errors if present, boundary effects).
- If you want the Y-error story to be the paper's headline, **rerun under a depolarizing or circuit-level noise model** that actually contains $Y$ errors, then back the claim with an ablation comparing X/Z-only vs. XYZ-depolarizing training.

**Why this is critical:** Any referee who reads the README and cross-references the STIM circuit will flag this in 60 seconds, and it damages the credibility of every other empirical claim in the paper.

### ERROR 2: Missing Shot-Count Disclosure
**Location:** `results/evaluation_results.csv`, `README.md §3`

**Problem:** The CSV reports point-estimate LER values with zero metadata on shot counts per $(d, p)$ cell. At $d=7, p=0.0015$, the expected LER under correct scaling is $\sim 2 \times 10^{-6}$ — resolving this to one significant figure requires $\gtrsim 5 \times 10^{6}$ samples for a Wilson 95% CI that doesn't span an order of magnitude. Without disclosed shot counts, every low-$p$ number in the CSV is statistically ambiguous.

**Action:** Add a supplementary table reporting:
- `N_shots` (total samples) per $(d, p)$ cell
- `N_failures` (observed logical errors) per cell
- Wilson 95% CI lower / upper bounds instead of point estimates

**Why this is critical:** No PR-tier reviewer accepts decoder evaluations without this. It's a rejection-round-1 blocker.

### ERROR 3: Zero-Failure Cells Polluting Log-Log Fit
**Location:** `results/evaluation_results.csv` (rows marked `nan` in `improvement_pct`)

**Problem:** At $d=5$ and $d=7$, low-$p$ rows show `transformer_ler = 0.0` — these are bins where both decoders recorded zero observed failures at whatever shot count was used. Currently they are either (a) treated as data points in the log-log fit (inflating $R^2$) or (b) silently dropped (distorting the slope estimate).

**Action:**
- Treat zero-failure cells as **upper bounds** (Wilson upper CI) in the fit, or exclude them explicitly and disclose the exclusion in the methodology.
- **Re-run** the log-log regression after CRITICAL fixes land and confirm the $d=7$ slope lands within **1%** of theoretical $(d+1)/2 = 4.0$. The current 4.205 slope is 5% high, consistent with capacity-bound behavior.

### ERROR 4: Extrapolated Thresholds Overclaimed
**Location:** `README.md §4`, `results/threshold_estimates.txt`

**Problem:** The reported decoder thresholds ($p_{th} \in [0.0131, 0.0395]$) sit **outside** the training range ($p \in [0.002, 0.017]$) and the evaluation range ($p \le 0.01$). These are log-log extrapolations, not measurements. The README currently frames them as empirical evidence that "the model learns generalized topological homology." This overclaims.

**Action:**
- Add bootstrap error bars on the threshold estimate (resample per-$p$ shots, re-fit, report 2.5/97.5 percentiles).
- Add an explicit disclaimer that the pseudothreshold is an extrapolation of the log-log fit, not a measurement, while keeping the evaluation range bounded to the physical regime $p < 0.02$.
- Soften the "generalized topological homology" language in the Insights section.

### ERROR 5: CLS Information Bottleneck (Architectural Bug)
**Location:** `model.py:TransformerQEC.__call__` (line 308), `x[:, 0]`

**Problem:** The classifier reads exactly one token — the prepended CLS — after 4 transformer layers. With 336 detectors at $d=7$, forcing the decision through a single 128-dim slot is an obvious bottleneck. Compounding this, the CLS is inserted with **identity rotation** (`cls_cos=1, cls_sin=0` at `model.py:297–300`), so the final layer's CLS attention has *no* positional discrimination over detector tokens.

**Action:** This is FIX 4 in `DEEP_RESEARCH_FIX_SUGGS.md`. Must land before submission — any architecture-fluent reviewer will identify this in 60 seconds.

---

## 4. Immediate Fixes (Priority Order)

Ordered by "must-do before any PR submission" → "story-making experiments" → "rigor polish":

### Tier A — Rejection-Round-1 Blockers (Do First)

1. **Verify and repair ERROR 1** — audit STIM noise model, delete or rerun the Y-errors claim.
2. **ERROR 2 — shot-count disclosure table** — add to supplementary material.
3. **ERROR 3 — Wilson CIs + zero-failure handling** — re-derive LER numbers as intervals, not points.
4. **ERROR 4 — bootstrap threshold error bars + language softening** — soften "generalized topological homology" framing.
5. **FIX 3** (physics-invariance regression tests) — must land before FIX 1 / FIX 2 to gate them in CI.
6. **FIX 1** (DIPE — integer-lattice RoPE anchoring).
7. **FIX 2** (graph-masked locality + soft physics penalty).
8. **FIX 4** (attention pooling — ERROR 5 fix).

### Tier B — Story-Making Experiments (Required for PR Research)

9. **FIX 7 (Unified Multi-Distance Backbone)** — **this is the headline experiment**. Train a single set of weights on interleaved batches from $d \in \{3, 5, 7, 9\}$, match or beat per-distance checkpoints, ideally demonstrate zero-shot generalization to held-out $d=11$. Without this, the parameter-efficiency framing collapses under scrutiny against AlphaQubit.
10. **FIX 6 (MWPM Soft-Label Distillation)** — gives a causal decomposition of where the transformer improves ("correlated-error structure MWPM misses") vs. "bigger is better." Enables the attention-map interpretability figure.
11. **Attention-map visualization at $d=5$** — show that learned $\beta$ tracks $-\log p$ and attention mass concentrates on DEM-adjacent detectors. This is the figure that converts "reasonable engineering" into "physics-informed contribution" and directly supports the strongest paper framing.

### Tier C — PR Research Polish

12. **Re-fit log-log regression** after Tier A + B land; confirm $d=7$ slope within 1% of 4.0.
13. **Pareto plot**: LER at fixed $p$ vs. parameter count, with your model and AlphaQubit-style baselines at *matched* noise model and training regime. This is the only way to make the "parameter efficiency" claim credible and comparable.
14. **FIX 8 (D₄ TTA)** — free 1–3% LER improvement. Adds a table row.

### Tier D — PRX Quantum Stretch (Only If Targeting That Venue)

15. **One circuit-level ablation at $d=5$** under STIM SI1000 circuit-level noise. Not full circuit-level retraining — just one datapoint showing the architecture transfers. Without this, PRX Quantum is out.
16. **Latency benchmark on TPU** — profile actual inference throughput, not just FLOPs. Required to back the "$<1\mu s$ decoder" claim that the fix list references.

### Not On Critical Path

- FIX 5 (QCT) — nice-to-have engineering; reviewers will not insist.
- FIX 9 (stratified sampling) — partially subsumed by the Wilson CI + shot-count rigor work in Tier A.
- FIX 10 (`nn.remat`) — engineering; reviewers won't care unless you go to $d \ge 9$.
- FIX 11 (Mamba) — out of scope for this paper; future work.

---

## 5. Recommended Paper Framing

Both reviewers converged on the same framing:

> **"Distance-invariant neural decoding: a physics-informed inductive-bias stack for parameter-efficient transformer decoders"**

**Structure:**
- **Headline:** Unified multi-distance backbone (FIX 1 + FIX 7) that matches or beats per-distance checkpoints with a single set of weights.
- **Methods section:** (2+1)D Anisotropic RoPE → DIPE → graph-masked locality with soft physics penalty.
- **Interpretability section:** MWPM distillation (FIX 6) + attention visualization showing learned $\beta \propto -\log p$.
- **Supporting table:** Pareto plot for parameter efficiency; 1.3M is a data point, *not* the title.

**Framings to avoid:**
- **"Small model beats big model"** — headline-bait that invites the AlphaQubit-comparison defense. You will spend the review debating whether 1.3M vs. 5.4M is a fair comparison (it isn't, since AlphaQubit was trained on circuit-level noise + Sycamore data).
- **"A new SOTA decoder"** — phenomenological-only and no hardware data; SOTA claims invite SOTA scrutiny and will lose.
- **Leading with 2.5D RoPE as the contribution** — 2D RoPE exists (RoFormer, Vision-RoPE); spatial/temporal band splitting is standard in video transformers. The physical justification is correct but reviewers will see this as "sensible RoPE instantiation for lattice data," not an architectural invention.

**Interpretability sleeper option:** If the main paper is delayed, the attention-map visualization at $d=5$ (showing DEM-alignment of attention + $\beta$ tracking $-\log p$) could be an independent NeurIPS ML4PS / ICLR Physics4ML short paper on its own, using the current baseline.

---

## 6. Timeline to Each Venue

| Target | Minimum Fix Set | Effort |
|---|---|---|
| **NeurIPS ML4PS / QIP poster / ICLR Physics4ML** | CRITICAL fixes + Tier A rigor | 2–4 weeks |
| **PR Research** | CRITICAL + FIX 4, 6, 7 + Tier A + B + C | 3–4 months |
| **Quantum (journal)** | Same as PR Research + scope statement + one circuit-level ablation | 4–5 months |
| **NeurIPS 2026 main track** | Same as PR Research + strong unified-backbone ablation | 3–4 months (submission deadline permitting) |
| **PRX Quantum** | All of the above + circuit-level noise retraining + latency benchmark + hardware data if available | 6+ months |
| **Nature** | Not on trajectory without hardware validation on actual quantum processor | Not recommended for this work |

---

## 7. The Three Most Important Actions After CRITICAL Fixes Land

Both reviewers independently agreed on these three as the highest-leverage post-CRITICAL work:

1. **Ship FIX 7 (Unified Multi-Distance Backbone).** This is the single experiment that turns "another transformer decoder" into a distance-agnostic neural decoder. Train one set of weights on interleaved $d \in \{3, 5, 7, 9\}$ batches, demonstrate zero-shot generalization to a held-out distance. Without it, the inductive-bias story is incremental; with it, the paper has a concrete, validated claim that AlphaQubit did not emphasize.

2. **Fix statistical rigor: shot counts + Wilson CIs + one circuit-level ablation.** Specifically: (a) report `N_shots` per $(d, p)$ cell; (b) report Wilson 95% CIs instead of point estimates; (c) treat zero-failure cells as upper bounds in the log-log fit; (d) verify the Y-errors claim (ERROR 1); (e) re-fit regression post-CRITICAL fixes and confirm $d=7$ slope within 1% of 4.0; (f) run at least one $d=5$ ablation under STIM circuit-level SI1000 to show the architecture transfers.

3. **Add FIX 6 (MWPM Distillation) and the interpretability figure.** MWPM-distilled training gives a causal decomposition of where the transformer improves — the delta becomes identifiable as "correlated-error structure MWPM misses," not "bigger is better." Pair with an attention-map visualization at $d=5$ showing the learned $\beta$ tracking $-\log p$. This converts the paper from "reasonable engineering" into "physics-informed contribution" and directly supports the strongest framing.

---

## 8. Honest Bottom Line

- **With CRITICAL fixes alone:** workshop-tier. Strong fit for NeurIPS ML4PS, QIP poster, ICLR Physics4ML. **Not** publishable in any Physical Review venue yet.
- **With CRITICAL + Tier A + Tier B + Tier C:** credible *Physical Review Research* submission (~3–4 months of work).
- **With all of the above + circuit-level ablation + latency benchmark:** *PRX Quantum* stretch target (~6+ months).
- *PRX*, *PRL*, and *Nature* are not on the trajectory for this specific scope; pursuing them would require hardware validation on an actual quantum processor (Sycamore, IBM, IonQ), which is a different project.
