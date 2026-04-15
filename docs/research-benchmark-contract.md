# Research Benchmark Contract

Every candidate method must declare:

- the exact config it uses,
- the target metric it aims to improve,
- the baseline artifact set it compares against,
- the saved outputs it produces,
- the written conclusion that accepts or rejects the change.

The contract applies to architecture changes, training changes, data-generation changes, decoder comparisons, and metric updates. A result is not package-facing until another maintainer can rerun the declared command, inspect the artifact set, and compare it against the blessed baseline without reconstructing hidden notebook state.

When a method is rejected, keep the conclusion brief and specific. Negative results are useful when they identify the config, metric, artifact set, and reason the change did not improve the baseline.
