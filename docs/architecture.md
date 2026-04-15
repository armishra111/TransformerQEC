# Architecture

TransformerQEC is organized as a package-first research codebase. The notebooks are retained as historical context, while package modules, CLI commands, checked-in configs, and recorded artifacts define the maintained workflow.

`transformerqec.codes` owns STIM circuit construction and detector coordinates. This is where rotated surface-code geometry and detector metadata enter the system.

`transformerqec.data` owns dataset generation. It turns circuit definitions, sampling parameters, and physical error rates into reusable syndrome examples for training and evaluation.

`transformerqec.models` owns the reusable Transformer and RoPE implementation. The model code is intentionally separate from training loops so architecture changes can be tested against fixed configs and artifact contracts.

`transformerqec.training` and `transformerqec.evaluation` own the executable training and benchmark workflows. These layers should bind configs, models, data, metrics, and artifact output together without hiding the evidence needed to reproduce a result.

`transformerqec.research` owns the registry for reproducible candidate methods and lightweight comparison helpers. Research claims should stay anchored in registered configs, saved artifacts, and docs.

The CLI is the package-facing entry point for these workflows. A baseline or research claim should be traceable from command, to config, to artifact set, to comparison against the blessed baseline.
