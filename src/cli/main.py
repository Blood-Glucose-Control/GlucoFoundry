# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""``glucofoundry`` command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.cli import new_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glucofoundry",
        description="GlucoFoundry toolkit command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    new_experiment.add_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
