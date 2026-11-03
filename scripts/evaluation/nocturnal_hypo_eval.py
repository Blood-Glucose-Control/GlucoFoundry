#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Canonical nocturnal hypoglycemia evaluation CLI entrypoint."""

from experiments.nocturnal.hypo_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
