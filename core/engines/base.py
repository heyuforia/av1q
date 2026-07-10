"""Engine interface: every behavior that legitimately differs between
the encode pipelines, behind one explicit surface.

The shared brain (search, refine, pipeline) must consume ONLY this
interface. Anything engine-specific that leaks outside core/engines/ is
a bug; a new encoder back-end is a new module here — never a fork of
the brain. Each engine's cache `sig` and key formats are frozen: caches
written before and after the package split must stay interchangeable.
"""


class Grid:
    """Quantizer-grid arithmetic.

    av1q searches integer CQs; av1q-essential searches quarter-step
    CRFs. All search/refine stepping goes through a Grid so the brain
    stays grid-agnostic. Bounds (min/max, floors, caps) are themselves
    always on-grid, which makes quantize-then-clamp equivalent to
    clamp-then-quantize — the brain relies on that.
    """

    step = None  # distance between adjacent grid points

    def quantize(self, v):
        """Snap v to the nearest grid point (the grid's native type)."""
        raise NotImplementedError

    def fmt(self, q):
        """Stable string for filenames and cache keys ('30', '23.25')."""
        raise NotImplementedError

    def fmt_delta(self, d):
        """Signed string for a quantizer jump ('+3', '-0.75')."""
        raise NotImplementedError

    def floor(self, v):
        """Largest grid point <= v."""
        raise NotImplementedError

    def ceil(self, v):
        """Smallest grid point >= v."""
        raise NotImplementedError

    def span(self, lo, hi):
        """All grid points from lo to hi inclusive."""
        raise NotImplementedError


class Engine:
    """One encoder back-end.

    Attributes and methods are consumed by the shared search/refine/
    pipeline brain. Implementations delegate to plain functions in their
    module so those functions stay directly importable (the launchers
    re-export them as public API).
    """

    sig = None              # per-file cache signature — FROZEN forever
    qname = None            # quantizer label in output: "CQ" / "CRF"
    banner = None           # startup banner name
    banner_extra = ""       # dim suffix after the banner name
    grid = None
    vmaf_key_base = None    # full-encode VMAF cache key: "full" / "vmaf"
    sample_ext = None       # container for sample probe encodes
    tmp_patterns = ()       # leftover temp outputs swept at startup
    rec_q_key = None        # quantizer field in the `recommended` block
    rec_bound_keys = ()     # (min, max) field names in `recommended`
    rec_extra_keys = ()     # extra cfg keys the `recommended` block covers
    seed_key = None         # cfg key holding the user seed quantizer
    seed_prompt_hint = None  # dim hint in the interactive seed prompt
    cal_q_key = None        # quantizer field in the calibration block
    needs_expected_frames = False  # engine's progress bar needs a frame count

    def cache_root(self, cfg):
        """Per-pipeline cache directory (never shared between engines)."""
        raise NotImplementedError

    def q_bounds(self, cfg):
        """(min, max) quantizer bounds from cfg, in grid-native type."""
        raise NotImplementedError

    def seed_override(self, cfg):
        """User-provided seed quantizer, or None for automatic."""
        raise NotImplementedError

    def parse_user_q(self, raw):
        """Parse interactive seed input; raises ValueError when invalid."""
        raise NotImplementedError

    def signature(self, cfg, crop=None):
        """Tag covering everything that changes encoder output at one q."""
        raise NotImplementedError

    def setup(self, cfg):
        """Discover required/optional tool binaries before processing.
        May raise FileNotFoundError when a required binary is missing."""
        raise NotImplementedError

    def make_dirs(self, cfg):
        """Create engine-specific cache directories (input/output dirs
        are the pipeline's job). Default: nothing extra."""

    def full_ref_index(self, cfg, source, file_hash, size):
        """Persistent FFMS2 reference-index path for a source file (the
        SSIMU2 info column re-measures it at verify/refine/final)."""
        raise NotImplementedError

    def sample_ref_index(self, cfg, sample_src):
        """Persistent FFMS2 reference-index path for the sample concat
        (reused across all probes within one search), or None."""
        raise NotImplementedError

    def gate(self, source, meta):
        """Reason string when this engine cannot process the source
        (e.g. VFR for the Y4M pipe), else None."""
        return None

    def prepare_meta(self, source, meta, cfg):
        """Engine-specific metadata enrichment before encoding starts.
        Returns True when something user-visible was carried over."""
        return False

    def prep_sample(self, concat, meta, cfg):
        """Turn the raw sample concat into this engine's search source
        (or None on failure). Default: use the concat as-is."""
        return concat

    def encode(self, source, dest, meta, q, cfg,
               show_progress=False, expected_frames=0, resumable=False):
        """Encode source to dest at quantizer q.

        resumable=True marks a full-file output encode the engine MAY
        route through an interrupted-encode resume path (sample probes
        never set it); engines without one simply ignore it."""
        raise NotImplementedError

    def ssimu2_info(self, ref, dist, meta, cfg, ref_index=None):
        """Display-only SSIMULACRA2 measurement ({'mean','p5'} or None)."""
        raise NotImplementedError

    def dst_name(self, stem, q, token, ext):
        """Output filename for a full encode (carries the crop token)."""
        raise NotImplementedError
