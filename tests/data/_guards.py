"""Shared skip guards for optional decode backends.

Test modules in this directory import this file directly (``from _guards
import ...``); pytest puts ``tests/data`` on ``sys.path`` when collecting
them, since the directory is not a package.
"""

from __future__ import annotations

import importlib

import pytest


def require_torchcodec(allow_module_level: bool = False) -> None:
    """Skip the current test (or module) unless torchcodec can decode video.

    torchcodec < 0.15 loads its FFmpeg shared libraries eagerly, so importing
    it raises RuntimeError (not ImportError) when the environment has no
    matching FFmpeg — common on CI runners. Newer versions import cleanly
    without FFmpeg and only raise once a decoder is constructed, so a
    successful import proves nothing; probe the deferred shared-library load
    explicitly as well.
    """
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f'torchcodec unavailable ({exc})',
            allow_module_level=allow_module_level,
        )

    for mod_name in (
        'torchcodec._core._decoder_utils',
        'torchcodec._internally_replaced_utils',
    ):
        try:
            load = importlib.import_module(mod_name).load_core_libraries
        except (ImportError, AttributeError):
            continue
        try:
            load()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(
                f'torchcodec FFmpeg shared libraries unavailable ({exc})',
                allow_module_level=allow_module_level,
            )
        return

    # No lazy-load hook found: this torchcodec is old enough to have loaded
    # FFmpeg during the import above, so decoding will work.
