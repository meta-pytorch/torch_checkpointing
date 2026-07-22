# Contributing

## Issues

We use GitHub issues to track public bugs and feature requests, and we'd love
to hear from you! Whether it's a bug report, feature request, or a question,
please don't hesitate to open an issue. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Development setup

`torch_checkpointing` requires Python >= 3.9 and torch >= 2.6. Clone the repo,
install it in editable mode with the dev extras, and run the tests:

```bash
git clone https://github.com/meta-pytorch/torch_checkpointing
cd torch_checkpointing
pip install -e ".[dev]"   # pytest, coverage, and formatting/type-check tools
pytest                     # runs the tests/ suite
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
