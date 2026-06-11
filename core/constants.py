"""Shared domain constants: container whitelist, per-resolution VMAF
targets and bitrate floors, and the search acceptance-band overshoot."""

import math

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".mov", ".m4v", ".ts", ".avi", ".webm"}
INTRA_ONLY_CODECS = {"prores", "dnxhd", "mjpeg", "rawvideo", "ffv1", "jpeg2000", "cfhd"}

TARGET_VMAF_BY_RES = {0: 93.0, 720: 94.0, 2160: 90.0}

FALLBACK_MAXRATE = {
    0: 8_000_000, 720: 12_000_000, 1080: 25_000_000,
    1440: 35_000_000, 2160: 45_000_000, 4320: 60_000_000,
}

# Starvation backstops, not targets.
MIN_BITRATE_KBPS = {0: 0, 720: 1000, 1080: 1800, 1440: 2500, 2160: 5000, 4320: 8000}

# Full-encode VMAF this far above target is treated as wasted bitrate worth
# a re-encode at higher CQ (unless the bitrate floor is what's holding CQ
# down). Also caps the search loop's acceptance band and bounds the
# skip-existing acceptance band for outputs that predate the
# completed-search marker. 0.5 trades roughly one extra full encode per
# overshooting file for ~5-10% smaller output (1 CQ step ≈ 11% bitrate).
VMAF_OVERSHOOT = 0.5

# Cold-start bitrate-decay slope d(log kbps)/d(quantizer) for the floor
# model: ±6 quantizer steps ≈ 2× bitrate. Used until measured probes (or
# an engine cohort's learned decay — see core/calibrate.py) refine it.
DEFAULT_BITRATE_DECAY = math.log(2) / 6

# A re-encode predicted to trim less than this fraction of bitrate costs
# more than it buys — the full-file search endgame and the refine loop
# both accept the current point instead (sample probes are cheap and are
# never snapped: there the extra probe still shrinks the final encode).
ENDGAME_SNAP_GAIN = 0.03

# Scaled-down sampling plan for short files. Files at or under the
# configured plan's threshold used to fall straight to full-file search,
# where every probe is a full encode; mini-samples keep probes cheap for
# sources still long enough to amortize the final encode + verify that
# the sample path adds on top. MIN_RATIO is that amortization gate:
# below duration > count×duration×ratio, each probe nearly encodes the
# whole file anyway and full-file search is strictly cheaper.
MINI_SAMPLE_COUNT = 3
MINI_SAMPLE_DURATION = 2.0
MINI_SAMPLE_MIN_RATIO = 2.5
