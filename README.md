# Static Analysis for Pyro Programs

### Overview

The static analyser implemented in `pyppai/` performs the following tasks:

- Verify the support correspondence for a given Pyro model-guide pair.
- Verify the absence of dimension-related errors from PyTorch's tensors and Pyro's plates.

To evaluate the static analyser, we applied it to the Pyro examples and Pyro test suite in `tests/`.
You can reproduce this experiment by running `make batch-suite` and `make batch-examples`.
Refer to [README.txt](README.txt) for the details of our implementation and experiments.

### Paper & Artifact

- Paper: [Towards Verified Stochastic Variational Inference for Probabilistic Programs](https://arxiv.org/abs/1907.08827),
  [POPL 2020](https://popl20.sigplan.org/).
- Artifact: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3545194.svg)](https://doi.org/10.5281/zenodo.3545194)
