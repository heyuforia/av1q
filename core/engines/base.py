"""Engine interface: every behavior that legitimately differs between
the encode pipelines, behind one explicit surface.

The shared brain (search, refine, pipeline) must consume ONLY this
interface. Anything engine-specific that leaks outside core/engines/ is
a bug; a new encoder back-end is a new module here — never a fork of
the brain. Each engine's cache `sig` and key formats are frozen: caches
written before and after the package split must stay interchangeable.
"""

from ..constants import DEFAULT_BITRATE_DECAY
from ..probe import probe_hdr_metadata
from ..segments import mux_states_hdr10


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
    # Leftover temp names swept at startup (output dir + shared cache
    # root). Only names THIS engine writes — dest-derived or hash-named —
    # never a bare "*.tmp.mkv": both pipelines may share an output
    # folder, and a loose glob would delete the other pipeline's
    # in-flight temp mid-encode.
    tmp_patterns = ()
    # ffmpeg components the run cannot do without, checked once at
    # launch against the resolved build. Without the check a build
    # lacking one fails every file only after its scene scan and sample
    # extraction were already paid for.
    ffmpeg_encoders = ()
    ffmpeg_filters = ()
    rec_q_key = None        # quantizer field in the `recommended` block
    rec_bound_keys = ()     # (min, max) field names in `recommended`
    rec_extra_keys = ()     # extra cfg keys the `recommended` block covers
    seed_key = None         # cfg key holding the user seed quantizer
    seed_prompt_hint = None  # dim hint in the interactive seed prompt
    cal_q_key = None        # quantizer field in the calibration block
    needs_expected_frames = False  # engine's progress bar needs a frame count
    # True when ffmpeg hands the encoder the HDR10 static metadata of
    # the first frame it decodes, so a failed read still leaves the
    # encode that copy. Without it a failed read would ship the encode
    # with none.
    hdr10_passthrough = False

    # Cold-start d(log kbps)/d(quantizer) for the floor model, before this
    # file's probes or the engine cohort have measured it. Encoder physics,
    # not policy: an engine whose quantizer scale is flatter than the
    # generic ln2/6 must say so, or every file of a young cohort is aimed
    # with a curve twice as steep as its encoder's and pays extra probes.
    default_decay = DEFAULT_BITRATE_DECAY

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
        Raises OSError (FileNotFoundError for a missing required binary)
        when the engine cannot run here; the message is printed as is."""
        raise NotImplementedError

    def launch_notes(self, cfg):
        """Dim lines printed under the banner, after setup: what this run
        found that the banner cannot name, such as the binary it runs.
        Default: none."""
        return ()

    def make_dirs(self, cfg):
        """Create engine-specific cache directories (input/output dirs
        are the pipeline's job). Default: nothing extra."""

    # FFMS2 reference indexes for the SSIMU2 info column, under this
    # engine's cache root. FFVship reads index paths as ANSI argv, so
    # names are hash-based and never carry a source's (possibly
    # non-Latin) stem. It also reuses an index without checking it
    # against the file, so each name changes when its file does.

    def full_ref_index(self, cfg, file_hash):
        """Persistent index for a source file (the SSIMU2 column
        re-measures it at verify, refine and final)."""
        return self.cache_root(cfg) / "_ffindex" / f"{file_hash}.ffindex"

    def sample_ref_index(self, cfg, sample_src):
        """Index for the search source, reused across one search's
        probes, or None when the file can't be read. It lives exactly
        as long as the file it indexes: the pipeline deletes the two
        together."""
        try:
            size = sample_src.stat().st_size
        except OSError:
            return None
        return (
            self.cache_root(cfg) / "_ffindex"
            / f"{sample_src.stem}_{size}.ffindex"
        )

    def gate(self, source, meta):
        """Reason string when this engine cannot process the source
        (e.g. VFR for the Y4M pipe), else None."""
        return None

    def prepare_meta(self, source, meta, cfg):
        """Source facts read once per file, before any encode of it.
        Returns the note the file's hdr line prints, or None for none.

        Default: the HDR10 static metadata (meta["mastering"] and
        meta["cll"], None when absent, in probe_hdr_metadata's form),
        which every engine states to its encoder itself, and the final
        mux to the container. No encoder finds it alone: the Y4M pipe
        carries none, and ffmpeg hands libsvtav1 only what its first
        decoded frame carries. A failed read raises, stopping the file,
        unless the engine has hdr10_passthrough to fall back on. An
        engine that reads more calls this too."""
        meta["mastering"] = meta["cll"] = None
        if not meta["hdr"]:
            return None
        read = probe_hdr_metadata(source)
        if read is None:
            if not self.hdr10_passthrough:
                raise RuntimeError(
                    "HDR10 metadata could not be read, the file is"
                    " stopped; the next run tries again"
                )
            return (
                "static metadata could not be read,"
                " ffmpeg's own copy is used"
            )
        meta["mastering"], meta["cll"] = read
        if not (meta["mastering"] or meta["cll"]):
            return None
        if mux_states_hdr10():
            return "static metadata carried over"
        return (
            "static metadata carried over"
            " (ffmpeg 9.0+ also writes it to the MKV header)"
        )

    def prep_sample(self, concat, meta, cfg):
        """Turn the raw sample concat into this engine's search source
        (or None on failure). Default: use the concat as-is."""
        return concat

    def encode(self, source, dest, meta, q, cfg,
               show_progress=False, expected_frames=0, resumable=False):
        """Encode source to dest at quantizer q.

        resumable=True marks a full-file output encode (sample probes
        never set it): dest gets the source's audio, subtitles, fonts and
        chapters beside the picture, and the engine MAY route the encode
        through an interrupted-encode resume path; engines without one
        ignore that part."""
        raise NotImplementedError

    def ssimu2_info(self, ref, dist, meta, cfg, ref_index=None):
        """Display-only SSIMULACRA2 measurement ({'mean','p5'} or None)."""
        raise NotImplementedError

    def dst_name(self, stem, q, token, ext):
        """Output filename for a full encode (carries the crop token)."""
        raise NotImplementedError
