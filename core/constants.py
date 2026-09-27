"""Shared domain constants and encode policy: container whitelist, the
run-wide numbers both launchers seed into cfg, per-resolution VMAF
targets and bitrate floors, and the search/refine economics."""

import math

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".mov", ".m4v", ".ts", ".avi", ".webm"}
INTRA_ONLY_CODECS = {"prores", "dnxhd", "mjpeg", "rawvideo", "ffv1", "jpeg2000", "cfhd"}

# Run-wide policy the launchers seed into cfg. One source for both
# pipelines: everything that shapes a search must stay in sync between
# the two scripts, and a literal repeated in each is how they drift.

# Output container. Matroska carries whatever audio and subtitle streams
# the source has (the mux ladder steps the few it rejects down to SRT).
OUTPUT_CONTAINER = ".mkv"

# VMAF noise epsilon: two scores closer than this are the same score. It
# is the acceptance band's lower edge (target - tol), the refine loop's
# hysteresis above the band top, and the pass/fail color's threshold —
# nothing anywhere acts on a difference smaller than this.
VMAF_TOLERANCE = 0.1

# Cold-start sample→full bitrate margin for complexity-selected samples:
# the hardest scenes are modeled as encoding ~20% hotter than the whole
# file, so a sample must clear margin × floor for the video to clear the
# floor. Superseded per file by the measured complexity spread (bounded
# to only ever tighten it) and by the cohort ratio once one exists;
# evenly-spaced samples use EVEN_SAMPLE_MARGIN instead (below).
BITRATE_MARGIN = 1.20

# Sample clip length. 6s spans several GOPs at any common keyint, enough
# for one probe's VMAF to average over real coding decisions; variety
# comes from sampling more scenes (SAMPLE_SCALE_*), never longer ones.
SAMPLE_DURATION = 6.0

# Scenes shorter than this are never sampled: a clip is cut to
# min(scene, SAMPLE_DURATION), and under 2s it is too few frames for a
# VMAF mean to say anything about the scene.
MIN_SCENE_DURATION = 2.0

# Sources under this are too short to sample by scenes: the sampling
# plan drops to the mini plan or full-file search (its own amortization
# gate, 1.25× the extracted total, is higher at the default --samples),
# and the crop scan uses evenly spaced windows instead of scene picks.
SHORT_THRESHOLD = 48

# scdet cut threshold, well under the filter's stock 10: soft cuts and
# dissolves register too. A missed cut means a complex scene never
# reaches the candidate list; a false cut only splits one scene in two.
SCENE_THRESHOLD = 3

# Packet-stat complexity is measured per window of this many seconds:
# fine enough to tell one scene from the next, coarse enough that a
# window's mean packet size averages over a whole GOP's worth of frames
# instead of reading one keyframe. The scene list and the sample picks
# map onto these windows (analyze.window_of), so every consumer must
# bucket time the same way.
COMPLEXITY_WINDOW = 5.0

# Wall-clock budget for one whole-file pass (the scdet decode, a packet
# demux, essential's VFR pts scan): 1× the source's runtime, floored so
# short files get a real allowance and ceiled so a genuinely hung
# process can't block a batch forever. Decode at 640px runs many times
# realtime and a demux at I/O speed, so the budget only trips on a
# stall; a fixed short cap used to kill the scan on exactly the long
# films that most need complexity-biased selection.
SCAN_TIMEOUT_MIN = 300
SCAN_TIMEOUT_MAX = 3600

TARGET_VMAF_BY_RES = {0: 93.0, 720: 94.0, 2160: 90.0}

FALLBACK_MAXRATE = {
    0: 8_000_000, 720: 12_000_000, 1080: 25_000_000,
    1440: 35_000_000, 2160: 45_000_000, 4320: 60_000_000,
}

# Starvation backstops, not targets.
MIN_BITRATE_KBPS = {0: 0, 720: 1000, 1080: 1800, 1440: 2500, 2160: 4500, 4320: 8000}

# Bitrate acceptance band: a video anywhere in [floor, floor × BITRATE_BAND]
# has hit the floor closely enough. The refine loop accepts the whole band,
# so it never spends an extra full encode shaving the last few percent off
# a video that's already there (e.g. trimming 5330kbps toward 5000 when the
# floor is 5000), and its bitrate jumps aim at the band's center. 1.1 keeps
# the overshoot under ~one CQ grid step.
BITRATE_BAND = 1.1

# Sample→full bitrate margin for evenly-spaced sampling. The normal margin
# (BITRATE_MARGIN, carried as cfg["bitrate_margin"]) models complexity-
# selection bias: samples cut from the hardest scenes encode hotter than
# the full video, so the sample must clear margin × floor for the video
# to clear the floor. Evenly spaced samples (intra-only sources, or any
# file with no detected scenes) carry no such bias — the sample is
# representative, so the ratio is ~1.0 and the big margin would over-cap
# the search and force a wasted refine re-encode. A small margin keeps
# the video centered in the band above the floor while leaving room for
# ratio noise.
#
# This is a COLD-START cushion, not a belief about the ratio: evenly-spaced
# files roll their measured ratios into their own cohort (see cohort_keys
# in core/calibrate.py), and once that cohort exists it supersedes this
# margin and shrinks toward the structural 1.0 instead. Leaving the cushion
# in the shrink target held a real file's floor threshold 9% above the
# truth, which capped its search a step early and bought a second full
# encode.
EVEN_SAMPLE_MARGIN = 1.05

# Lower bound for the complexity-derived sample→full margin (see
# complexity_bias_margin in core/sampling.py). For scene-selected samples
# the bias the floor search needs — how much hotter the sampled (hottest)
# scenes encode than the whole file — is estimated per file from the
# packet-stat complexity spread instead of the fixed cold-start margin.
# This floor caps how far that estimate may tighten the margin, bounding
# the downside when the source-codec complexity proxy under-reads the true
# AV1 bitrate bias (the two-sided refine loop still backstops any floor
# miss). It sits above EVEN_SAMPLE_MARGIN: a scene-selected sample is
# never as representative as an evenly-spaced one.
COMPLEXITY_MARGIN_FLOOR = 1.10

# Full-encode VMAF this far above target is treated as wasted bitrate worth
# a re-encode at higher CQ (unless the bitrate floor is what's holding CQ
# down). Also caps the search loop's acceptance band and bounds the
# skip-existing acceptance band for outputs that predate the
# completed-search marker. 0.5 trades roughly one extra full encode per
# overshooting file for ~5-10% smaller output (1 CQ step ≈ 11% bitrate).
VMAF_OVERSHOOT = 0.5

# Structural cold-start sample→full VMAF offset for a FULLY
# complexity-biased sample. The sample is cut from the file's hardest
# scenes, so at equal quantizer it scores systematically LOW against the
# full encode; aiming the sample search with a zero offset lands the
# first full encode high and the refine loop pays a whole re-encode to
# walk it back down. The cohort average supersedes this as evidence
# accumulates (calibration_offset blends the cohort toward this center,
# not toward 0). Field-measured true offsets so far: -0.58, -0.79, -1.10
# (mean -0.82) — a -0.5 prior left two of those three cold files landing
# above the acceptance band and re-encoding. -0.75 centers the observed
# range while staying a notch shallow of the mean.
#
# How biased a file's selection actually came out varies, so this is the
# value at REFERENCE bias, scaled down per file by
# calibrate.scene_offset_center: a source whose hottest scenes barely
# clear its own average is evenly sampled in all but name and gets an
# offset near 0. Charging one such file the full -0.75 aimed its search
# a step low and cost a 41-minute re-encode (its measured bias was 1.03
# against a true offset of +0.04). Evenly-spaced samples are
# representative by construction and keep a center of 0.
SCENE_OFFSET_PRIOR = -0.75

# Generic cold-start bitrate-decay slope d(log kbps)/d(quantizer) for the
# floor model: ±6 quantizer steps ≈ 2× bitrate. Used until measured
# probes (or an engine cohort's learned decay — see core/calibrate.py)
# refine it.
#
# How a nominal quantizer maps to bitrate is encoder physics, so each
# engine states its own (Engine.default_decay); this is the fallback and
# the value av1q's integer CQ grid uses. Nothing shared may assume it —
# blending Essential's cohort toward it taxed that engine's early files.
DEFAULT_BITRATE_DECAY = math.log(2) / 6

# Cold-start VMAF points per quantizer step, before two probes have
# measured the local slope. It only sizes the first jump: the search
# refits from every probe pair, and a bound landing sized by this guess
# is never taken as proof (the min_q short-circuit in core/search.py
# waits for a measured slope). The refine loop starts from it too when
# the search measured none.
DEFAULT_VMAF_SLOPE = 0.5

# Floor-bound mode: a seed that clears the target by at least
# FLOOR_BOUND_VMAF_MARGIN while its bitrate sits under
# FLOOR_BOUND_KBPS_RATIO × the floor threshold is a bitrate problem, not
# a quality one, so the search stops measuring VMAF on the probes that
# walk down to the floor. The margin is what makes the skip safe: every
# later probe sits at a lower quantizer and scores higher still, and the
# chosen one is measured once at the end.
FLOOR_BOUND_VMAF_MARGIN = 2.0
FLOOR_BOUND_KBPS_RATIO = 0.80

# A re-encode predicted to trim less than this fraction of bitrate costs
# more than it buys — the full-file search endgame and the refine loop
# both accept the current point instead. The same economics waive a
# shortfall this close under the floor, in the full-file search, the
# refine loop and final selection alike: lifting bitrate by less buys
# nothing either. Sample probes are cheap and are never snapped or
# waived: there the extra probe still shrinks the final encode.
ENDGAME_SNAP_GAIN = 0.03

# Resumable segmented encodes. Full encodes of sources at least this long
# are written as keyframe-aligned segment files that the muxer finalizes
# as they complete, so an interrupted encode resumes at the last segment
# boundary instead of restarting from frame 0. Below the gate the
# segment/concat/remux overhead isn't worth the ~minutes it could save;
# above it, a kill costs at most ~2 segments of work (the in-flight one
# plus the boundary segment re-encoded for an exact seam). Segment length
# is a policy constant, not a knob: 60s keeps the worst-case loss around
# one percent of a feature while keeping the segment count (and concat
# list) small. Cuts land on the first keyframe at/after each multiple, so
# real segments run a few seconds over (SVT's default keyint is ~5-7s).
RESUMABLE_MIN_DURATION = 900.0
SEGMENT_TIME = 60

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

# Duration-aware scaling for the standard sampling plan. A fixed sample
# count captures a fixed slice of runtime, so an hours-long feature is
# sampled at a small fraction of a short clip's coverage — too few distinct
# scenes to represent the film's full complexity range, which pushes the
# search onto its expensive full-file refine backstop (a whole re-encode)
# whenever the sample estimate misses. The sampled scene COUNT therefore
# grows with duration; clip length stays fixed (6s already spans multiple
# GOPs — variety comes from more scenes, not longer ones). Growth is
# logarithmic off SAMPLE_SCALE_REF, the reference runtime the base count
# (cfg["sample_count"], i.e. --samples) is tuned for: the base band is
# unchanged at/below the reference, then +SAMPLE_SCALE_K scenes per
# doubling of duration, capped at SAMPLE_COUNT_MAX so search cost stays a
# small fraction of a long encode. --samples is the base the curve scales
# up FROM, so raising it shifts the whole curve; a base at or above the cap
# pins the count flat (clamp with lo >= hi returns lo).
SAMPLE_SCALE_REF = 600.0
SAMPLE_SCALE_K = 4.0
SAMPLE_COUNT_MAX = 24
