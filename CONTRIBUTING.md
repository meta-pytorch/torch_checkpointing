# Contributing

## Issues

We use GitHub issues to track public bugs and feature requests, and we'd love
to hear from you! Whether it's a bug report, feature request, or a question,
please don't hesitate to open an issue. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Development setup

`torch_checkpointing` requires Python >= 3.11 and torch >= 2.11. We use
[uv](https://docs.astral.sh/uv/) to manage the development environment, with
the dependency versions locked in `uv.lock`:

```bash
git clone https://github.com/meta-pytorch/torch_checkpointing
cd torch_checkpointing
uv sync               # create .venv: the package (editable) plus test and lint tools
uv run pytest         # run the tests/ suite
uv run ruff check .   # lint
```

`uv run pytest -n 8 --dist loadfile` runs the tests in parallel.

`uv sync` installs torch from PyPI, which on Linux is a CUDA build, so the GPU
tests run if your machine has GPUs; tests that need more GPUs than you have
skip themselves. To use the smaller CPU-only build instead, as CI does:

```bash
uv sync --no-group cuda --group cpu
uv run --no-sync pytest   # plain `uv run` would switch back to the CUDA build
```

If your NVIDIA driver is too old for PyPI's CUDA build, let uv pick a build that
matches it:

```bash
uv pip install --reinstall-package torch torch --torch-backend auto
uv run --no-sync pytest
```

If you change dependencies in `pyproject.toml`, run `uv lock` and commit the
updated `uv.lock`.

To check that the lower bounds in `pyproject.toml` still hold, run the tests
with the lowest versions they allow. This rewrites `uv.lock`; restore it
afterwards (`git checkout uv.lock`).

```bash
uv sync --resolution lowest-direct --no-group cuda --group cpu
uv run --no-sync pytest
```

## Pull Requests

We are not currently accepting pull requests, but we plan to enable them soon.
In the meantime, please file an issue and we'll work with you from there.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## License

By contributing to `torch_checkpointing`, you agree that your contributions will
be licensed under the LICENSE file in the root directory of this source tree.
`torch_checkpointing` is licensed under the BSD 3-Clause License, as found in the
[LICENSE](./LICENSE) file.
