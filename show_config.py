#!/usr/bin/env python3
"""Thin shim — delegates to the package's show_professor_config().

Prefer:  python main.py --show-config
"""

from dotenv import load_dotenv

load_dotenv()

from src.runtime.info_commands import show_professor_config  # noqa: E402

if __name__ == "__main__":
    show_professor_config()
