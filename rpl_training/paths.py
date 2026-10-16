"""Filesystem locations of the assets shipped with ``rpl_training``."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
"""FALCON ACCAD upper-body clips (``motions/``) and the MoE teacher checkpoints (``models/``)."""
