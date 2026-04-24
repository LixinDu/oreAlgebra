#!/usr/bin/env python3
"""Backward-compatible CLI wrapper for ``core/ore_rag_assistant.py``."""


def main(argv=None) -> int:
    from core.ore_rag_assistant import main as _main

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

