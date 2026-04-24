#!/usr/bin/env python3
"""Backward-compatible Streamlit entrypoint for ``ui/streamlit_chat_app.py``."""


def main() -> None:
    from ui.streamlit_chat_app import main as _main

    _main()


if __name__ == "__main__":
    main()

