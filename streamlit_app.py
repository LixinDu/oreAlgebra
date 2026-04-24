#!/usr/bin/env python3
"""Backward-compatible Streamlit entrypoint for ``ui/streamlit_app.py``."""


def main() -> None:
    from ui.streamlit_app import main as _main

    _main()


if __name__ == "__main__":
    main()

