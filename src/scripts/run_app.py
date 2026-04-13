#!/usr/bin/env python
"""Launch the PhishGuard Gradio web application."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phishguard.web.app import build_app


def main() -> None:
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
