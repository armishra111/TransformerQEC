# Research Landscape

TransformerQEC sits in a line of neural and classical decoders for surface-code quantum error correction. Its current role is not only to propose a Transformer decoder, but to package the workflow so claims can be rerun, compared, and accepted or rejected with checked-in evidence.

- AlphaQubit: recurrent and transformer decoding under circuit-level noise. It is the closest high-profile neural decoding reference point and motivates future circuit-level noise evaluation.
- Transformer-QEC repos: attention over syndrome strings with learned positional structure. These projects show that attention can model syndrome correlations, but reproducibility varies across data generation, training, and benchmark scripts.
- PyMatching and sparse blossom: strong classical graph-matching baselines. They remain the practical baseline family any learned decoder must compare against because they are fast, well understood, and broadly used.
- TransformerQEC differentiator: explicit `(2+1)D` RoPE with a checked-in regression harness and public baseline artifacts.

The engineering position of this repository is therefore conservative: every method improvement should be expressed as a reproducible package workflow. Architecture ideas, training changes, or benchmark updates should resolve to configs, saved artifacts, and a written conclusion against the blessed baseline.
