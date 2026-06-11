"""Shared implementation package for av1q, av1q-essential, and av1q-crop.

The launcher scripts at the repo root remain the public entry points and
the stable import surface; the modules here hold the implementation.
Engine-specific behavior (mainline SVT-AV1 via ffmpeg vs the standalone
SVT-AV1-Essential binary) belongs under core/engines/ — everything else
is shared and must stay engine-agnostic.
"""
