# Reproducible release environment

Use the committed `uv.lock` for a repeatable package installation. The
installation commands in the README use `uv sync --locked`, which checks that
the lock agrees with `pyproject.toml` and installs the locked versions. Add the
same extras each time you sync; omitted extras are removed from the environment.
The documented commands were checked with uv 0.10.4 and Python 3.11.

For a minimal analyzer/evaluator environment, run `uv sync --locked`. For
training, analysis, and tests, run:

```bash
uv sync --locked --python 3.11 --extra train --extra analysis --extra dev
source .venv/bin/activate
uv pip check
python -m pytest -q
```

Using `pip install -e '.[train]'` instead resolves the version ranges in
`pyproject.toml`; it does not consume `uv.lock`. Use the locked workflow when
sharing an environment or comparing runs.

## Relationship to the reference checkpoints

PEFT 0.19.1 matches the version recorded in the reference adapters' configuration
files. This identifies the adapter serialization version; it does not establish
the package versions used at every training step. The accompanying base-model
configuration records Transformers 4.57.1. The release retains Transformers
4.57.1, Torch 2.7.1, torchvision 0.22.1, and qwen-vl-utils 0.0.14 as its checked
compatibility stack.

The original server package freeze and base-model commit have not been recovered.
The committed lock specifies the public release environment, rather than a
recovered snapshot of that server. CPU tests and small-model checks do not
establish full-model training or paper-result equivalence.

## GPU runtime

The preset attention backend is FlashAttention 2. Its original version, the
original CUDA runtime, and the GPU driver version are not known. FlashAttention
is therefore not assigned an invented historical pin or installed automatically.
Install a build compatible with your PyTorch, CUDA, and GPU, following the
[training setup](training.md). Selecting SDPA changes the attention backend and
requires its own numerical validation.

Complete the locked sync before installing FlashAttention. A subsequent exact
sync removes packages installed outside the lock, so reinstall the extension
afterward, or deliberately use `uv sync --locked --inexact` to retain it and
record its version separately.

Record the base-model commit, Python and package versions, FlashAttention version,
CUDA runtime and driver, GPU type, and distributed launch settings with each run.
The lock covers Python package resolution; it does not pin the host driver or
establish identical GPU kernels across machines.
