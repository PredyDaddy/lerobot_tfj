#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().parents[1] / "tfj_envs" / "groot_rl" / "scripts" / "openclaw_groot_server.py"

runpy.run_path(str(TARGET), run_name="__main__")
