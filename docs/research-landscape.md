# Research Landscape

TransformerQEC sits in a line of neural and classical decoders for surface-code quantum error correction. Its current role is to package a Transformer-QEC workflow so claims can be rerun, compared, and accepted or rejected with checked-in evidence.

- AlphaQubit: recurrent and transformer decoding under circuit-level noise. It is a high-profile neural decoding reference point and a useful target for future circuit-level noise comparisons.
- PyMatching and Sparse Blossom: graph-matching decoders remain the practical classical baseline family for surface-code experiments. They are fast, well documented, and strong enough that any learned decoder needs explicit comparison against them.
- STIM: fast stabilizer circuit simulation and detector error model tooling make it practical to generate syndrome data and MWPM inputs from the same circuit definitions.
- RoPE: rotary position embeddings provide a reusable way to encode relative structure in attention. This repo adapts that idea to explicit detector coordinates with a `(2+1)D` spatial/temporal split.
- Transformer-QEC-style repositories: related projects apply attention to syndrome strings or detector events, often with learned positional structure. They motivate this package, but the reproducibility story depends on how each project records configs, artifacts, and baseline comparisons.
- TransformerQEC differentiator: checked-in configs, public baseline artifacts, docs contracts, and smoke/regression tests around the current Transformer-QEC workflow.

The engineering position of this repository is deliberately conservative. Architecture ideas, training changes, or benchmark updates should resolve to configs, saved artifacts, and a written conclusion against the blessed baseline before they are presented as package-facing research claims.
