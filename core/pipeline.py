"""The per-file processing pipeline shared by av1q and av1q-essential.

process_videos drives: discovery → skip-existing → probe → crop →
engine gate/meta prep → scene analysis → sampling (engine sample prep)
→ search (core.search) → final encode + verify → calibration
persistence → the consolidated refine loop → final selection and
cleanup. Everything engine-specific goes through the Engine interface,
including the printed output's quantizer labels and grid formatting.

Two cache scopes, deliberately distinct, and the calibration beside them:
  cfg["cache_dir"]        shared between pipelines — scene analysis,
                          sample extraction, crop sidecars and temp
                          logs (source facts; the formats are
                          identical).
  engine.cache_root(cfg)  per-pipeline — result caches, per-file
                          calibration, sample encodes, FFMS2 indexes.
                          Different encoders must never share these.
  engine.calibration_root(cfg)
                          per-pipeline cross-file cohort, outside the
                          cache so deleting the cache keeps it.
"""

import math
import os
import sys
import time

from . import search as core_search
from . import segments as core_segments
from . import vmaf as core_vmaf
from .analyze import scene_analysis
from .bitrate import calc_kbps, video_kbps
from .cache import load_cache, recommended_matches
from .calibrate import (
    DECAY_MAX, DECAY_MIN, OFFSET_MAX, RATIO_MAX, RATIO_MIN,
    calibration_offset, decay_prior, file_calibration, ratio_prior,
    load_global_calibration, scene_offset_center, update_global_calibration,
    vmaf_slope_prior,
)
from .constants import (
    BITRATE_BAND, COMPLEXITY_MARGIN_FLOOR, DEFAULT_VMAF_SLOPE,
    ENDGAME_SNAP_GAIN, EVEN_SAMPLE_MARGIN, INTRA_ONLY_CODECS, MIN_BITRATE_KBPS,
    MINI_SAMPLE_COUNT, MINI_SAMPLE_DURATION, MINI_SAMPLE_MIN_RATIO,
    TARGET_VMAF_BY_RES, VIDEO_EXTENSIONS, VMAF_OVERSHOOT, VMAF_SLOPE_MAX,
    VMAF_SLOPE_MIN,
)
from .crop import (
    crop_scan_cfg, crop_token, detect_crop_for_file, load_crop_sidecar,
    read_crop_sidecar, sidecar_crop, sidecar_path,
)
from .probe import parse_rate, probe_video, res_tier
from .sampling import (
    complexity_bias, complexity_bias_margin, extract_samples, sampling_plan,
    select_samples,
)
from .tools import have_ffmpeg, local_ffmpeg_dir, missing_ffmpeg_components
from .ui import (
    BOLD, CHECK, CROSS, DIM, GREEN, LABEL_W, MIDDOT, ORANGE, PURPLE, RED,
    RESET, SEP, fmt_s2, fmt_size, fmt_time, fmt_vmaf, label,
)
from .util import atomic_write_json, clamp, cleanup_temp, partial_hash


def result_kbps(video, size_bytes, duration):
    """Bitrate for a finished encode's result line: `video`, the encode's
    video-only rate, or None when it could not be read.

    Every bitrate this tool decides on is video-only — the floor, the
    sample threshold, the refine gates — because samples are cut -an
    while full outputs carry audio and subs. Printing the muxed rate here
    instead put a different, larger number under the same word on the one
    line the eye lands on, so a file accepted at 1954kbps against an 1800
    floor read as 2419. The size line below already reports the whole
    file. Falls back to the muxed rate only when the video stream can't
    be measured (unreadable, or too short to divide by). It takes the rate
    the caller already read: reading again after a failure only prints
    the same failure twice.
    """
    return video or calc_kbps(size_bytes, duration)


def process_videos(cfg, engine):
    grid = engine.grid
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    cache_dir = cfg["cache_dir"]
    ext = cfg["container"]
    vmaf_threads = os.cpu_count() or 4

    if not have_ffmpeg():
        print(f"{CROSS} ffmpeg/ffprobe not found in PATH or the av1q folder")
        return 1
    # The build must carry what this engine encodes and measures with.
    # Found out here, once, rather than per file after its scene scan
    # and sample extraction have run — and named, because a dropped-in
    # build missing libvmaf otherwise reads as a broken install.
    try:
        missing = missing_ffmpeg_components(
            engine.ffmpeg_encoders, engine.ffmpeg_filters
        )
    except (OSError, RuntimeError) as e:
        print(f"{CROSS} ffmpeg failed to run: {e}")
        return 1
    if missing:
        where = local_ffmpeg_dir() or "PATH"
        print(
            f"{CROSS} ffmpeg ({where}) was built without"
            f" {', '.join(missing)}"
        )
        return 1
    try:
        engine.setup(cfg)
    except OSError as e:
        print(f"{CROSS} {e}")
        return 1

    root_cache = engine.cache_root(cfg)
    cal_root = engine.calibration_root(cfg)
    min_q, max_q = engine.q_bounds(cfg)
    # How this encoder's quantizer maps to bitrate before anything is
    # measured. Per engine, never a shared constant (Engine.default_decay).
    engine_decay = engine.default_decay
    # Forced-quantizer mode: the user picked the value, so the whole
    # estimation apparatus (sampling, search, VMAF, refine) has nothing
    # to decide. Grid-native, set by the launchers (--force-cq /
    # --force-crf); deliberately NOT clamped to the search bounds —
    # they bound the search, and there is no search.
    force_q = cfg.get("force_q")

    print(f"{PURPLE}{BOLD}{engine.banner}{RESET}{engine.banner_extra}\n{SEP}")

    # A build dropped into the av1q folder silently outranks whatever
    # is on PATH, so name the folder that won: a local build missing
    # libsvtav1 or libvmaf otherwise reads as a broken install. The
    # engine names its own binaries the same way.
    local_ff = local_ffmpeg_dir()
    notes = [f"ffmpeg: {local_ff}"] if local_ff else []
    notes += engine.launch_notes(cfg)
    if notes:
        for note in notes:
            print(f"{DIM}{note}{RESET}")
        print(SEP)

    # Interactive seed prompt: lets a batch of similar files start the
    # search at a known-good quantizer instead of the automatic seed
    # (which falls back to 30 for intra-only sources like ProRes). Enter
    # keeps auto behavior. Only when stdin is a terminal — piped/scripted
    # runs must not block.
    if (force_q is None and engine.seed_override(cfg) is None
            and sys.stdin.isatty()):
        while True:
            try:
                raw = input(
                    f"{PURPLE}{BOLD}Seed {engine.qname}"
                    f" {grid.fmt(min_q)}–{grid.fmt(max_q)}{RESET}"
                    f" {DIM}{engine.seed_prompt_hint}{RESET}: "
                ).strip()
            except EOFError:
                break
            if not raw:
                break
            try:
                val = engine.parse_user_q(raw)
            except ValueError:
                print(f"  {RED}Invalid input.{RESET}")
                continue
            if not min_q <= val <= max_q:
                print(
                    f"  {RED}Invalid: {engine.qname} must be "
                    f"{grid.fmt(min_q)}–{grid.fmt(max_q)}{RESET}"
                )
                continue
            cfg[engine.seed_key] = val
            break
        print(SEP)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.make_dirs(cfg)

    # Leftover encode temps live next to the output (dest-derived names)
    # and under the shared cache root: an engine may redirect the
    # encoder's scratch file there (essential's Y4M path), and the
    # clean-sample pass writes beside the shared concats. Both are swept
    # so a hard kill can't strand either; each engine's patterns name
    # only its own temps (see Engine.tmp_patterns).
    for pat in engine.tmp_patterns:
        for base in (output_dir, cache_dir):
            for p in base.rglob(pat):
                try:
                    p.unlink()
                except OSError:
                    pass
    # Unlike the leftover .tmp outputs above, segment dirs with a valid
    # manifest are resume state and survive; only torn ones are junk.
    core_segments.sweep_orphan_segments(root_cache)

    pattern = "**/*" if cfg["recurse"] else "*"
    # Sorted for a deterministic batch order (glob order is filesystem-
    # dependent), matching av1q-crop's listing.
    files = sorted(
        f for f in input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )
    total = len(files)
    if not files:
        print(f"{CROSS} No videos found in {input_dir}")
        return 1

    # A user seed only seeds NEW searches: files with a completed search
    # resume past it (and verified outputs skip entirely), which reads as
    # the seed being silently ignored. Offer the choice up front, once
    # for the whole batch — a per-file prompt would stall unattended
    # runs partway through. Yes clears those files' caches so they get a
    # fresh search from the seed; the default keeps today's behavior.
    user_seed = engine.seed_override(cfg)
    seeded_redo = set()
    if force_q is None and user_seed is not None and files and sys.stdin.isatty():
        prior = []
        for f in files:
            try:
                fh = partial_hash(f)
            except OSError:
                continue
            c, _ = load_cache(root_cache, fh, engine.sig)
            if recommended_matches(
                    c.get("recommended"), engine, cfg,
                    load_crop_sidecar(cache_dir, f, fh)
                    if cfg["use_crops"] else None,
                    cfg["target_vmaf"]):
                prior.append((f, fh))
        if prior:
            print(
                f"{ORANGE}{BOLD}{len(prior)}{RESET}{ORANGE} of {total}"
                f" video(s) already encoded in a previous run:{RESET}"
            )
            for f, _ in prior[:5]:
                print(f"   {DIM}{f.name}{RESET}")
            if len(prior) > 5:
                print(f"   {DIM}... and {len(prior) - 5} more{RESET}")
            try:
                raw = input(
                    f"{PURPLE}{BOLD}Re-encode them with a fresh search"
                    f" from your seed {engine.qname}"
                    f" {grid.fmt(grid.quantize(user_seed))}?{RESET}"
                    f" {DIM}(y/N){RESET}: "
                ).strip().lower()
            except EOFError:
                raw = ""
            if raw in ("y", "yes"):
                seeded_redo = {fh for _, fh in prior}
            print(SEP)

    # `failed` counts files that stopped on an error, not the ones skipped
    # or deleted by design; any makes the run's exit code 1.
    stats = {
        "proc": 0, "vmaf_sum": 0.0, "vmaf_n": 0,
        "saved": 0, "orig": 0, "deleted": 0, "failed": 0,
    }
    t_start = time.time()
    global_cal = load_global_calibration(cal_root)

    # Whether (and how) a file gets sampled is sampling_plan's call:
    # the configured plan for long sources, a scaled-down mini plan for
    # short ones, full-file search only below the mini amortization gate.
    mini_min = MINI_SAMPLE_COUNT * MINI_SAMPLE_DURATION * MINI_SAMPLE_MIN_RATIO

    all_qs = grid.span(min_q, max_q)

    for idx, filepath in enumerate(files, 1):
        sample_src = sample_concat = sample_idx = None
        _file_error = False
        search_state = None
        try:
            rel = filepath.parent.relative_to(input_dir)
            out_dir = output_dir / rel
            out_dir.mkdir(parents=True, exist_ok=True)

            file_hash = partial_hash(filepath)
            cache, cp = load_cache(root_cache, file_hash, engine.sig)

            if file_hash in seeded_redo:
                # User chose a fresh seeded search over the previous
                # result: drop every search product (entries,
                # calibration, recommended) so nothing resumes or skips
                # below. The forced block and the outputs record are
                # facts about files on disk, not search products, and
                # survive; scene analysis lives in the shared store and
                # is untouched.
                cache = {"sig": engine.sig, "entries": {}, **{
                    k: cache[k] for k in ("forced", "outputs") if k in cache
                }}
                atomic_write_json(cp, cache)

            # Output names carry the crop token so cropped and uncropped
            # encodes of the same source never collide (flipping
            # --no-crops between runs used to confuse the skip-existing
            # check). Before probing, the sidecar is the best crop
            # expectation; rebound after crop resolution (--auto-crop may
            # just have written one).
            def make_dst_path(crop):
                token = crop_token(crop)
                return lambda q: out_dir / engine.dst_name(
                    filepath.stem, q, token, ext
                )

            expected_crop = (
                load_crop_sidecar(cache_dir, filepath, file_hash)
                if cfg["use_crops"] else None
            )
            dst_path = make_dst_path(expected_crop)

            # Cached-VMAF measurement bound to this file's cache. Search,
            # verify, and refine all route through here so every score
            # lands in (and reuses) the same frozen cache layout.
            def measure(ref, dist, q, tag=None):
                return core_vmaf.vmaf_cached(
                    ref, dist, meta, q, cache, cp, tag=tag,
                    threads=vmaf_threads, log_dir=root_cache,
                    key_base=engine.vmaf_key_base, q_key=grid.fmt(q),
                )

            if cfg["skip_existing"]:
                skip_note = None
                if force_q is not None:
                    # Forced mode has its own done contract: the `forced`
                    # cache block records, per quantizer, the settings a
                    # forced encode was made with plus its output size. A
                    # bare file at the right name proves nothing (it could
                    # be a stale-settings encode or a searched-run
                    # leftover), and the search's `recommended` block must
                    # play no part in either direction — a forced encode
                    # is not a finished search, and a finished search must
                    # never skip a forced encode.
                    fb = cache.get("forced")
                    fe = (
                        fb.get(grid.fmt(force_q))
                        if isinstance(fb, dict) else None
                    )
                    d = dst_path(force_q)
                    if (isinstance(fe, dict)
                            and all(
                                fe.get(k) == cfg[k]
                                for k in ("preset", "film_grain",
                                          *engine.rec_extra_keys)
                            )
                            and fe.get("crop") == expected_crop
                            and d.exists()
                            and fe.get("size") == d.stat().st_size):
                        skip_note = f"{CHECK} exists"
                else:
                    # A bare output file proves nothing (an interrupted
                    # search leaves probe encodes behind), and neither
                    # does a completed search: the refine loop can still
                    # move the quantizer or reject the whole encode. The
                    # cache's `recommended` block, written when a search
                    # completes and matched here against the current
                    # settings, therefore records how the file ended:
                    #   pending  search done, file not (a run stopped in
                    #            verify or refine): never skipped, the
                    #            run resumes where it stopped
                    #   kept     skipped on the recommended quantizer's
                    #            output, its full VMAF cached for that
                    #            exact file
                    #   larger   the encode came out larger than the
                    #            source and was deleted: skipped with no
                    #            output, since the same settings would
                    #            only encode it again to delete it again
                    # A block from before outcomes were recorded keeps its
                    # own rule: the recommended quantizer's output, or any
                    # output whose verified VMAF sits inside the band.
                    rec = cache.get("recommended")
                    # Auto targets vary by resolution and the file hasn't
                    # been probed yet, so only an explicit --vmaf can be
                    # checked here.
                    if recommended_matches(rec, engine, cfg, expected_crop,
                                           cfg["target_vmaf"]):
                        outcome = rec.get("outcome")
                        rec_target = rec.get("target")
                        if outcome == "larger":
                            skip_note = (
                                f"{CHECK} source kept"
                                f" {DIM}(its AV1 encode came out larger){RESET}"
                            )
                        for c in all_qs if outcome in ("kept", None) else ():
                            d = dst_path(c)
                            if not d.exists():
                                continue
                            score = core_vmaf.stored_vmaf(
                                cache["entries"].get(grid.fmt(c)),
                                engine.vmaf_key_base, d.stat().st_size,
                            )
                            if score is None:
                                continue
                            in_band = (
                                outcome is None
                                and isinstance(rec_target, (int, float))
                                and rec_target - cfg["vmaf_tolerance"]
                                <= score["mean"]
                                <= rec_target + VMAF_OVERSHOOT
                            )
                            if c == rec.get(engine.rec_q_key) or in_band:
                                skip_note = f"{CHECK} exists"
                                break
                if skip_note:
                    # A skipped file never reaches the post-encode
                    # cleanup below, so reclaim any segment dirs an
                    # interrupted encode left behind here — they can
                    # hold most of a movie's video stream.
                    core_segments.cleanup_file_segments(root_cache, file_hash)
                    print(f" {PURPLE}{filepath.name:<30}{RESET} {skip_note}")
                    continue

            if idx > 1:
                print(SEP)
            print(f"{PURPLE}{BOLD}[{idx}/{total}]{RESET} {PURPLE}{filepath.name}{RESET}")
            if file_hash in seeded_redo:
                print(
                    f"{label('redo')}{DIM}previous results cleared, searching"
                    f" from seed {engine.qname}"
                    f" {grid.fmt(grid.quantize(user_seed))}{RESET}"
                )

            meta = probe_video(filepath)

            # File info line
            in_sz = filepath.stat().st_size
            res_str = f"{meta['w']}x{meta['h']}" if meta["w"] and meta["h"] else "?"
            codec_str = (meta["codec"] or "?").upper()
            sz_str = fmt_size(in_sz)
            src_kbps = f"{meta['bitrate'] // 1000}kbps" if meta.get("bitrate") else ""
            dur_str = fmt_time(meta["duration"]) if meta["duration"] > 0 else ""
            hdr_str = "HDR" if meta["hdr"] else ""
            info_parts = [p for p in [res_str, codec_str, sz_str, src_kbps, dur_str, hdr_str] if p]
            sep = f" {MIDDOT} "
            print(f"      {DIM}{sep.join(info_parts)}{RESET}")

            # An audio-only container (a podcast .mp4, a music .webm)
            # probes with no picture at all; every stage below maps
            # 0:v:0 and would fail one by one.
            if not meta["w"] or not meta["h"]:
                print(f" {CROSS} No video stream, skipping")
                continue

            if meta["codec"] == "av1":
                print(f" {CHECK} Already AV1, skipping")
                continue

            # Engine gate: sources this engine cannot process (e.g. VFR
            # for the CFR-only Y4M pipe).
            gate_reason = engine.gate(filepath, meta)
            if gate_reason:
                print(f" {CROSS} {gate_reason}")
                continue

            # Crop resolution. A fresh --auto-crop scan applies its own
            # result directly (its verdict line was just printed) and
            # writes the sidecar for next time; a sidecar write that
            # fails costs only that re-scan. A sidecar read back from
            # disk says so, and one that is present but not applied
            # says why — a scanned file about to be encoded uncropped
            # is worth a line.
            meta["crop"] = None
            sidecar = sidecar_path(cache_dir, filepath, file_hash)
            if cfg["auto_crop"] and not sidecar.exists():
                try:
                    data = detect_crop_for_file(
                        filepath, meta, crop_scan_cfg(cfg), file_hash,
                    )
                except Exception as e:
                    print(f"{label('crop err')}{e}")
                else:
                    try:
                        sidecar.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write_json(sidecar, data, indent=2)
                    except OSError as e:
                        print(
                            f"{label('crop')}{DIM}sidecar not written"
                            f" ({e}); the next run scans again{RESET}"
                        )
                    if cfg["use_crops"]:
                        meta["crop"] = sidecar_crop(data, file_hash)[0]
            elif cfg["use_crops"]:
                meta["crop"], note = read_crop_sidecar(
                    cache_dir, filepath, file_hash
                )
                if meta["crop"]:
                    print(f"{label('crop')}{BOLD}{meta['crop']}{RESET}")
                elif note:
                    print(f"{label('crop')}{DIM}{note}{RESET}")
            dst_path = make_dst_path(meta["crop"])

            # Source facts the encodes state themselves (HDR10 static
            # metadata, plus whatever the engine adds).
            hdr_note = engine.prepare_meta(filepath, meta, cfg)
            if hdr_note:
                print(f"{label('hdr')}{DIM}{hdr_note}{RESET}")

            expected_frames = 0
            if engine.needs_expected_frames:
                fps_f = parse_rate(meta.get("fps"))
                if fps_f and meta["duration"] > 0:
                    expected_frames = int(meta["duration"] * fps_f)

            # Which settings made the file at each output name. A name
            # carries only the quantizer and the crop, so a file another
            # run left under other settings (film grain, preset, tune)
            # sits at the exact name this run would write. It is reused
            # only when this record says these settings made it at its
            # current size; anything else is encoded again.
            enc_tag = engine.signature(cfg, meta.get("crop"))
            outputs = cache.get("outputs")
            if not isinstance(outputs, dict):
                outputs = cache["outputs"] = {}

            def record_output(q):
                outputs[grid.fmt(q)] = {
                    "enc_tag": enc_tag, "size": dst_path(q).stat().st_size,
                }

            t_enc = t_vmaf = 0.0

            def full_encode(q):
                """The full encode at q (on-grid; the search, refine and
                the launchers' CLI all hand one over) for forced mode,
                the full-file search, the verify and the refine loop
                alike: the file already at its output name when these
                settings made it (see record_output), otherwise a fresh
                encode, recorded as soon as it lands."""
                nonlocal t_enc
                d = dst_path(q)
                made = outputs.get(grid.fmt(q))
                if (d.exists() and isinstance(made, dict)
                        and made.get("enc_tag") == enc_tag
                        and made.get("size") == d.stat().st_size):
                    print(
                        f"{label('reuse')}{engine.qname}"
                        f" {BOLD}{grid.fmt(q)}{RESET} encode exists"
                    )
                    return d
                t0 = time.time()
                engine.encode(
                    filepath, d, meta, q, cfg,
                    show_progress=True, expected_frames=expected_frames,
                    resumable=True,
                )
                t_enc += time.time() - t0
                record_output(q)
                atomic_write_json(cp, cache)
                return d

            # Forced mode: one full encode at the user's quantizer and
            # done. Probe/crop/gate/meta prep above still apply (they are
            # source facts, not search machinery); segments still resume;
            # nothing downstream runs — no sampling, search, VMAF, SSIMU2,
            # calibration, or refine. Deliberate differences from the
            # searched path: `recommended` is neither read nor written
            # (it means "a search finished", which never happened),
            # sibling-quantizer outputs are left alone (forcing a ladder
            # of values for an A/B is the point of the mode), and a
            # larger-than-source result is kept — the user chose the
            # quantizer, so the file is the deliverable, not a failed
            # compression bet.
            if force_q is not None:
                print(
                    f"{label('forced')}{engine.qname}"
                    f" {BOLD}{grid.fmt(force_q)}{RESET}"
                    f" {DIM}(skipping search){RESET}"
                )
                # A file these settings already made at this quantizer,
                # by any earlier run, is that encode and is reused.
                final = full_encode(force_q)
                if not final.exists():
                    print(f" {CROSS} Final encode missing")
                    _file_error = True
                    continue
                core_segments.cleanup_file_segments(root_cache, file_hash)
                out_sz = final.stat().st_size

                # The forced done-marker (see the skip-existing check):
                # per-quantizer so a ladder of forced values each skip
                # independently on re-runs.
                forced = cache.get("forced")
                forced = dict(forced) if isinstance(forced, dict) else {}
                forced[grid.fmt(force_q)] = {
                    "preset": cfg["preset"],
                    "film_grain": cfg["film_grain"],
                    **{k: cfg[k] for k in engine.rec_extra_keys},
                    "crop": meta["crop"], "size": out_sz,
                }
                cache["forced"] = forced
                atomic_write_json(cp, cache)

                saved = (1.0 - out_sz / in_sz) * 100
                out_kbps = result_kbps(
                    video_kbps(final, meta["duration"]), out_sz,
                    meta["duration"],
                )
                kbps_final = (
                    f"  {DIM}{MIDDOT}{RESET}  {BOLD}{out_kbps}kbps{RESET}"
                    if out_kbps else ""
                )
                print(SEP)
                print(
                    f" {CHECK} {engine.qname}"
                    f" {BOLD}{grid.fmt(force_q)}{RESET}{kbps_final}"
                )
                sv_color = GREEN if out_sz < in_sz else RED
                print(
                    f" {CHECK} {fmt_size(in_sz)} ->"
                    f" {BOLD}{fmt_size(out_sz)}{RESET}"
                    f" saved {sv_color}{BOLD}{saved:.1f}%{RESET}"
                )
                if out_sz >= in_sz:
                    print(
                        f" {ORANGE}Larger than the source — kept"
                        f" (forced {engine.qname}){RESET}"
                    )
                print(f"   {DIM}Enc {fmt_time(t_enc)}{RESET}")

                stats["proc"] += 1
                stats["saved"] += in_sz - out_sz
                stats["orig"] += in_sz
                continue

            tier = max(k for k in TARGET_VMAF_BY_RES if min(meta["w"], meta["h"]) >= k)
            target = cfg.get("target_vmaf") or TARGET_VMAF_BY_RES[tier]

            # Persistent FFMS2 reference index for this source (SSIMU2
            # info column only — display, never gating).
            full_idx = engine.full_ref_index(cfg, file_hash)

            # Resume only from the cache's `recommended` block, written
            # when a search completes. A bare output file at some
            # quantizer is NOT evidence of a finished search: the
            # full-file search path writes its probe encodes straight to
            # the output dir, so an interrupted run leaves the seed
            # encode behind — trusting it shipped seed-quality files at
            # several times the intended bitrate. Leftover probes are
            # still reused (full_encode keeps a file these settings
            # made, VMAF is cached by size), so re-running the search
            # after an interruption stays cheap.
            existing_q = None
            rec = cache.get("recommended")
            if recommended_matches(rec, engine, cfg, meta["crop"], target):
                existing_q = grid.quantize(rec[engine.rec_q_key])
                # The file is being worked on again, so it is pending
                # until it finishes, whatever an earlier run recorded: a
                # stop from here on must resume, never skip. A dry run
                # touches no encode and leaves the outcome alone.
                if not cfg["dry_run"] and rec.get("outcome") != "pending":
                    rec["outcome"] = "pending"
                    atomic_write_json(cp, cache)
                seed_note = ""
                if user_seed is not None:
                    # The seed only starts a NEW search; saying so here
                    # beats looking like silently ignored input.
                    seed_note = (
                        f" {DIM}(seed {engine.qname}"
                        f" {grid.fmt(grid.quantize(user_seed))} not used:"
                        f" search already done){RESET}"
                    )
                print(
                    f"{label('resume')}{engine.qname} {BOLD}{grid.fmt(existing_q)}{RESET}"
                    f" from previous search{seed_note}"
                )

            sample_scenes = sample_src = sample_at_best = None
            even_sampling = False
            complexity = []  # per-window complexity; used for the margin estimate
            plan = sampling_plan(meta["duration"], cfg)
            # Mini-plan runs keep their own cohort (see cohort_keys).
            mini_sampling = bool(plan) and plan[2] == "mini"

            if existing_q is None and plan:
                n_samples, s_dur, _ = plan
                if mini_sampling:
                    print(
                        f"{label('short')}{meta['duration']:.0f}s source →"
                        f" mini-samples ({n_samples}×{s_dur:.0f}s)"
                    )
                if meta["codec"] in INTRA_ONLY_CODECS:
                    print(f"{label('skip')}Intra-only codec ({meta['codec']}), using even samples")
                # Stored per source in the shared cache root, so the
                # inline crop scan, av1q-crop, and the other pipeline
                # all read the one scan.
                scenes, complexity, keyframes = scene_analysis(
                    filepath, meta, cfg, file_hash
                )

                # The plan already decided sampling applies, so disarm
                # select_samples' own short-file bail-out and use the
                # plan's clip length (mini plans cut shorter clips).
                select_cfg = {
                    **cfg, "sample_duration": s_dur, "short_threshold": 0,
                }
                sample_scenes = select_samples(
                    scenes, complexity, meta["duration"], n_samples,
                    keyframes, select_cfg,
                )
                # No detected scenes (intra-only sources, or scdet found
                # none) means the samples are evenly spaced and therefore
                # representative, not complexity-biased — the sample→full
                # bitrate ratio is ~1.0, so the floor search uses a small
                # margin instead of the complexity-bias one.
                even_sampling = not scenes
                # A degenerate scene list can't fill the plan:
                # select_samples picks each distinct scene at most once,
                # so a source with a lone detected cut yields a single
                # clip — and betting the whole search on it is how one
                # near-static scene misreads a high-bitrate source as
                # floor-bound. Too few scene samples → re-select evenly
                # spaced, which is representative by construction
                # (mirrors the count//2 guard on select_samples' keyframe
                # path; the max(2, ·) stops mini plans from riding on a
                # single clip, min(count, ·) keeps 1-sample plans valid).
                min_scene_samples = min(n_samples, max(2, n_samples // 2))
                if (not even_sampling and sample_scenes
                        and len(sample_scenes) < min_scene_samples):
                    print(
                        f"{label('fallback')}scenes fill only"
                        f" {len(sample_scenes)} of {n_samples} samples,"
                        f" switching to evenly-spaced"
                    )
                    sample_scenes = select_samples(
                        [], complexity, meta["duration"], n_samples,
                        keyframes, select_cfg,
                    )
                    even_sampling = True
                if sample_scenes:
                    info = (
                        f"samples from {BOLD}{len(scenes)}{RESET} scenes"
                        if not even_sampling else "evenly-spaced samples"
                    )
                    print(f"{label('scenes')}{BOLD}{len(sample_scenes)}{RESET} {info}")
                    print(f"{label('extract')}Extracting samples...")
                    sample_concat = extract_samples(
                        filepath, sample_scenes, keyframes, cfg,
                        file_hash=file_hash,
                    )
                    # Engines may turn the raw concat into their own
                    # search source (av1q uses it as-is; essential runs a
                    # lossless clean re-encode — see clean_sample_source).
                    sample_src = (
                        engine.prep_sample(sample_concat, meta, cfg)
                        if sample_concat else None
                    )
                    if not sample_src:
                        print(f"{label('fallback')}Extraction failed, using full encode")
                        sample_scenes = None
                    else:
                        # Named while the file exists (the name carries
                        # its size); deleted with it below.
                        sample_idx = engine.sample_ref_index(cfg, sample_src)
                else:
                    print(f"{label('scenes')}Using full VMAF")
            elif existing_q is None:
                print(f"{label('short')}≤{mini_min:.0f}s, full VMAF")

            sample_enc_dir = root_cache / "_sample_enc"
            sample_enc_dir.mkdir(parents=True, exist_ok=True)
            sample_enc_cache = {}

            def do_enc_sample(q):
                nonlocal t_enc
                q = grid.quantize(clamp(q, min_q, max_q))
                if q in sample_enc_cache:
                    return sample_enc_cache[q]
                if not sample_src or not sample_src.exists():
                    raise RuntimeError("Sample source missing")
                d = sample_enc_dir / (
                    f"sample_enc_{file_hash[:8]}_{enc_tag}_{grid.fmt(q)}"
                    f"{engine.sample_ext}"
                )
                if d.exists() and d.stat().st_size > 0:
                    sample_enc_cache[q] = d
                    return d
                t0 = time.time()
                engine.encode(sample_src, d, meta, q, cfg)
                t_enc += time.time() - t0
                if not d.exists():
                    raise RuntimeError("Encoding failed")
                sample_enc_cache[q] = d
                return d

            min_kbps = MIN_BITRATE_KBPS.get(res_tier(meta["w"], meta["h"]), 0)
            floor_str = (
                f" {DIM}{MIDDOT}{RESET} floor {BOLD}{min_kbps}kbps{RESET}"
                if min_kbps else ""
            )
            print(f"{label('target')}VMAF {BOLD}{target:.1f}{RESET}{floor_str}")

            # What this file measured before, under these settings only
            # (see file_calibration). Every per-file prior reads it, and
            # the calibration write after the verify extends it.
            file_cal = file_calibration(cache, enc_tag)

            # Engine-cohort bitrate-decay prior for the search's floor
            # model: how fast THIS encoder's bitrate falls per quantizer
            # step (Essential's CRF encodes richer than mainline's CQ at
            # equal numbers). Sizes the first jump toward the floor for
            # the engine instead of the generic ±6 ≈ 2× cold-start, which
            # cost essential 1-2 extra probes per file.
            dec_prior, dec_src = decay_prior(
                file_cal, global_cal, default=engine_decay
            )
            if (min_kbps and dec_prior is not None
                    and abs(dec_prior - engine_decay)
                    >= 0.15 * engine_decay):
                print(
                    f"{label('calibr')}bitrate decay {BOLD}{dec_prior:.3f}{RESET}"
                    f"/{engine.qname} {DIM}({dec_src}){RESET}"
                )

            if existing_q is not None:
                best_q = existing_q
            elif sample_src:
                # How complexity-biased this file's selection actually
                # came out — the same measurement the bitrate margin below
                # is built from (complexity_bias_margin), so the two halves
                # of the sample→full prediction can't disagree about how
                # biased the sample is.
                bias = complexity_bias(complexity, sample_scenes)
                # Apply learned VMAF offset (sample over/under-predicts
                # full VMAF) so the sample search aims at the quantizer
                # that will hit `target` on the full video. Per-file
                # calibration takes precedence; on first encounter we fall
                # back to the cohort average, blended toward this file's
                # structural center — for complexity-selected samples that
                # is SCENE_OFFSET_PRIOR scaled by the bias actually
                # measured above (scene_offset_center), for evenly-spaced
                # ones it is 0. Each mode reads its OWN cohort: the scene
                # cohort's whole content is scene-selection bias, which
                # doesn't apply to an evenly-spaced sample.
                sample_target = target
                off, off_src = calibration_offset(
                    file_cal, global_cal,
                    prior_center=(
                        0.0 if even_sampling
                        else scene_offset_center(bias, cfg["bitrate_margin"])
                    ),
                    even=even_sampling, mini=mini_sampling,
                )
                if off is not None and abs(off) >= cfg["vmaf_tolerance"]:
                    sample_target = clamp(target + off, 0.0, 100.0)
                    print(
                        f"{label('calibr')}sample target"
                        f" {BOLD}{sample_target:.2f}{RESET}"
                        f" {DIM}(offset {off:+.2f} {off_src}){RESET}"
                    )
                # Sample→full bitrate margin for the floor search — how much
                # hotter the sampled scenes encode than the whole video.
                # Evenly-spaced samples are representative (ratio ~1.0), so
                # shrink toward the small EVEN_SAMPLE_MARGIN; otherwise the
                # search over-predicts the video bitrate, caps the quantizer
                # too low, and ships a video well over the floor that refine
                # then climbs back down a full encode at a time. Scene-
                # selected samples ARE complexity-biased, but how much is
                # estimated per file from its own complexity spread instead
                # of a fixed guess — bounded so it can only tighten the
                # cold-start margin. Either way the cohort ratio prior
                # (below) supersedes this margin once a few files of the
                # same sampling mode have been measured; for scene-selected
                # samples the margin still seeds that prior's shrink target.
                search_margin = cfg["bitrate_margin"]
                if even_sampling:
                    search_margin = min(search_margin, EVEN_SAMPLE_MARGIN)
                elif sample_scenes:
                    search_margin = complexity_bias_margin(
                        complexity, sample_scenes, search_margin,
                        COMPLEXITY_MARGIN_FLOOR,
                    )
                search_cfg = cfg
                if search_margin != cfg["bitrate_margin"]:
                    search_cfg = {**cfg, "bitrate_margin": search_margin}
                    if min_kbps and not even_sampling:
                        print(
                            f"{label('calibr')}sample margin"
                            f" {BOLD}{search_margin:.2f}{RESET}"
                            f" {DIM}(complexity spread){RESET}"
                        )

                # Cohort sample→full ratio prior: the cross-file other half
                # of the floor calibration. Per-file ratio (in calibration)
                # still takes precedence inside effective_sample_floor; this
                # only kicks in for files that haven't been encoded yet, so a
                # fresh file aims at the learned floor instead of paying the
                # conservative-margin tax. Each sampling mode reads its own
                # cohort: evenly-spaced files used to be denied a cohort
                # entirely and stayed pinned to EVEN_SAMPLE_MARGIN's implied
                # ratio however many of them had been measured — a fixed 5%
                # cushion that on a real file sat 9% off the truth, capped
                # the search a step early and cost a second full encode.
                rat_prior, rat_src = ratio_prior(
                    file_cal, global_cal, search_margin,
                    even=even_sampling, mini=mini_sampling,
                )
                if (min_kbps and rat_prior is not None and rat_src != "per-file"
                        and abs(rat_prior - 1.0 / search_margin) >= 0.01):
                    print(
                        f"{label('calibr')}bitrate ratio {BOLD}{rat_prior:.2f}{RESET}"
                        f" {DIM}({rat_src}){RESET}"
                    )

                best_q, sample_at_best, _, vt, search_state = core_search.search(
                    sample_src, meta, sample_target, cache, cp,
                    do_enc_sample, search_cfg, engine, tag="sample",
                    decay_prior=dec_prior, ratio_prior=rat_prior,
                    measure_fn=lambda ref, dist, q: measure(
                        ref, dist, q, tag="sample"),
                    probe_fn=probe_video,
                    s2_fn=lambda ref, dist, m, ri: engine.ssimu2_info(
                        ref, dist, m, cfg, ref_index=ri),
                    s2_ref_index=sample_idx,
                )
                t_vmaf += vt
                for p in sample_enc_cache.values():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            else:
                best_q, best_vmaf, _, vt, search_state = core_search.search(
                    filepath, meta, target, cache, cp,
                    full_encode, cfg, engine, decay_prior=dec_prior,
                    measure_fn=lambda ref, dist, q: measure(ref, dist, q),
                    probe_fn=probe_video,
                    s2_fn=lambda ref, dist, m, ri: engine.ssimu2_info(
                        ref, dist, m, cfg, ref_index=ri),
                    s2_ref_index=full_idx,
                )
                t_vmaf += vt

            # The search returns nothing only when its first probe could
            # not be measured.
            if best_q is None:
                print(f" {CROSS} Search stopped: the first probe has no VMAF")
                _file_error = True
                continue

            # Mark the search as completed for BOTH search paths, with the
            # file still pending (see the skip-existing check). Written
            # before the final encode so an interruption resumes here; the
            # finished file's quantizer and outcome replace it at the end.
            if existing_q is None:
                cache["recommended"] = {
                    engine.rec_q_key: best_q, "target": target,
                    engine.rec_bound_keys[0]: cfg[engine.rec_bound_keys[0]],
                    engine.rec_bound_keys[1]: cfg[engine.rec_bound_keys[1]],
                    "preset": cfg["preset"], "film_grain": cfg["film_grain"],
                    **{k: cfg[k] for k in engine.rec_extra_keys},
                    "crop": meta["crop"], "outcome": "pending",
                }
                # The sample half of the calibration pair, which the
                # verify completes: what the search measured at its
                # answer, the quantizer it measured it at, and the cohort
                # the pair rolls into, since a resumed run skips the
                # sampling that decides it. Kept here, not read back from
                # the entries, where a later search can measure the same
                # quantizer under other settings or another sample.
                # Absent when there is no pair: a full-file search, or a
                # sample search whose answer was chosen without a probe.
                if (sample_at_best
                        and math.isfinite(sample_at_best.get("mean", float("nan")))):
                    pair = {
                        "q": grid.fmt(best_q), "vmaf": sample_at_best["mean"],
                        "even": even_sampling, "mini": mini_sampling,
                    }
                    kbps = search_state.get("kbps", {}).get(best_q)
                    if kbps:
                        pair["kbps"] = kbps
                    cache["recommended"]["sample_pair"] = pair
                atomic_write_json(cp, cache)

            if cfg["dry_run"]:
                entry = cache["entries"].get(grid.fmt(best_q), {})
                sv = (entry.get(f"sample_{engine.vmaf_key_base}")
                      or entry.get(engine.vmaf_key_base))
                vmaf_str = (
                    f" VMAF {BOLD}{sv:.2f}{RESET}"
                    if isinstance(sv, (int, float)) else ""
                )
                print(
                    f" {CHECK} Recommended {engine.qname}"
                    f" {BOLD}{grid.fmt(best_q)}{RESET}{vmaf_str}"
                )
                print("   Run without --dry-run to encode")
                continue

            # The full-file search's probes ARE full encodes: what it
            # measured of them is reused below, never measured twice.
            full_search = not sample_src and existing_q is None
            full_probes = search_state if full_search else {}

            # SSIMU2 info per full-encode quantizer (display only).
            s2_seen = dict(full_probes.get("ssimu2", {}))

            def size_kbps_suffix(path, kbps):
                """Trailing 'size  video-kbps' field for a full-encode result
                line, mirroring the sample probe lines (file size then
                video-only bitrate). The size is the muxed output on disk
                (audio/subs included); the kbps stays video-only for floor
                parity, so the two can legitimately differ."""
                parts = [fmt_size(path.stat().st_size)]
                if kbps:
                    parts.append(f"{kbps}kbps")
                return f"  {DIM}{' '.join(parts)}{RESET}"

            # Bitrate of the best_q encode: read once, by the verify below
            # or already by the full-file search, and reused by the
            # calibration block, the refine loop and the result line.
            actual_kbps_now = full_probes.get("kbps", {}).get(best_q)

            # Final full encode at the candidate quantizer + VMAF verify
            if sample_src or existing_q is not None:
                full_encode(best_q)
                print(f"{label('verify')}Full VMAF...")
                t0 = time.time()
                best_vmaf = measure(filepath, dst_path(best_q), best_q)
                s2_seen[best_q] = engine.ssimu2_info(
                    filepath, dst_path(best_q), meta, cfg, ref_index=full_idx,
                )
                t_vmaf += time.time() - t0
                actual_kbps_now = video_kbps(dst_path(best_q), meta["duration"])
                print(
                    f"{'':>{LABEL_W + 1}}"
                    f"{fmt_vmaf(best_vmaf, target, cfg['vmaf_tolerance'])}"
                    f"{fmt_s2(s2_seen[best_q])}"
                    f"{size_kbps_suffix(dst_path(best_q), actual_kbps_now)}"
                )

            # Persist this pass's measurements BEFORE the refine loop, so
            # a resumed or repeated run of this file starts from them. The
            # block extends file_cal, the one written under these
            # settings; a block from other settings is replaced whole. A
            # resumed file runs no search, so refine starts from the slope
            # the file's own search stored.
            stored_slope = None if search_state else vmaf_slope_prior(file_cal)
            cal_now = dict(file_cal or {})
            # Measurements the block did not hold yet. Only these reach
            # the cohort, so each file counts once per settings in it,
            # however often it is resumed or searched again.
            new = {}

            if search_state:
                sl = search_state.get("vmaf_slope")
                if sl and VMAF_SLOPE_MIN <= sl <= VMAF_SLOPE_MAX:
                    cal_now["vmaf_slope"] = sl
                # Bitrate decay actually measured by this search's probes
                # (never the cold-start default: search reports those as
                # None).
                md = search_state.get("measured_decay")
                if isinstance(md, (int, float)) and DECAY_MIN <= md <= DECAY_MAX:
                    if "decay" not in cal_now:
                        new["decay"] = md
                    cal_now["decay"] = md

            # The sample→full pair, at the search's answer: the sample
            # half the search recorded there (sample_pair), the full half
            # this verify, at the same quantizer only. Each half is
            # measured once per block; a later pass that verifies again
            # (a resume, an --overwrite rerun) already holds it, and after
            # refine moved the file its quantizer is not the pair's any
            # more, so a half that failed to measure the first time is
            # never taken from another quantizer. The isinstance guards:
            # these come straight from the JSON cache, and a corrupt
            # value must be ignored like every other calibration read,
            # not crash the file on the arithmetic.
            pair = cache["recommended"].get("sample_pair")
            if not (isinstance(pair, dict) and pair.get("q") == grid.fmt(best_q)):
                pair = None
            if pair:
                sample_kbps = pair.get("kbps")
                sample_vmaf = pair.get("vmaf")
                if ("ratio" not in cal_now and actual_kbps_now
                        and isinstance(sample_kbps, (int, float))
                        and sample_kbps > 0):
                    ratio = actual_kbps_now / sample_kbps
                    if RATIO_MIN <= ratio <= RATIO_MAX:
                        new["ratio"] = ratio
                        print(
                            f"{label('calibr')}sample {sample_kbps}kbps ->"
                            f" video {actual_kbps_now}kbps (ratio {ratio:.2f})"
                        )
                if ("vmaf_offset" not in cal_now and best_vmaf
                        and isinstance(sample_vmaf, (int, float))
                        and math.isfinite(sample_vmaf)
                        and math.isfinite(best_vmaf.get("mean", float("nan")))):
                    offset = sample_vmaf - best_vmaf["mean"]
                    if -OFFSET_MAX <= offset <= OFFSET_MAX:
                        new["vmaf_offset"] = offset
                if "ratio" in new or "vmaf_offset" in new:
                    cal_now[engine.cal_q_key] = best_q
            cal_now.update(new)

            if cal_now != (file_cal or {}):
                cal_now["enc_tag"] = enc_tag
                cal_now["t"] = time.time()
                cache["calibration"] = cal_now
                atomic_write_json(cp, cache)

            # Roll the new measurements into the cohort so later files
            # start from an informed prior. (Each engine has its own
            # cohort file; different encoders must never share
            # calibration.) The offset and ratio go into the cohort of
            # the sampling mode and plan the pair was measured under:
            # mixing representative (evenly-spaced) measurements into the
            # scene cohort would dilute the selection bias it exists to
            # measure and mis-aim every scene-sampled file after them.
            # Decay is engine physics, not selection bias: every cohort
            # shares one average.
            if new:
                update_global_calibration(
                    cal_root,
                    vmaf_offset=new.get("vmaf_offset"),
                    ratio=new.get("ratio"),
                    decay=new.get("decay"),
                    even=bool(pair and pair.get("even")),
                    mini=bool(pair and pair.get("mini")),
                )
                global_cal = load_global_calibration(cal_root)

            # Consolidated refine for quality/bitrate misses in BOTH
            # directions. Deficits (VMAF below target, bitrate below the
            # floor) step the quantizer down; overshoot (VMAF more than
            # VMAF_OVERSHOOT above target with bitrate headroom over the
            # floor) steps it up — without this, any candidate that
            # arrives here too low (e.g. a sample search whose offset
            # under-corrected) ships an oversized file. Each move is a
            # slope-sized jump, not a step-by-1, so it converges in 1-2
            # encodes. Both models start from what this file measured
            # (the search's fits, or on a resume the stored slope and
            # the per-file or cohort decay), and fall back to the cold
            # start only when nothing was. The clamps are the shared
            # sanity ranges: the decay's is engine-neutral, since each
            # engine's curve sits in a different place inside it.
            slope_v = clamp(
                (search_state.get("vmaf_slope") if search_state else None)
                or stored_slope or DEFAULT_VMAF_SLOPE,
                VMAF_SLOPE_MIN, VMAF_SLOPE_MAX,
            )
            decay_b = clamp(
                (search_state.get("bitrate_decay") if search_state else None)
                or dec_prior or engine_decay,
                DECAY_MIN, DECAY_MAX,
            )
            # VMAF jumps aim at the CENTER of the acceptance band
            # [target - tol, target + VMAF_OVERSHOOT] and round to the
            # grid, so slope error spreads symmetrically inside the band.
            # A floored step against a higher aim stacks every landing in
            # the band's top quarter, where one slope misread walks back
            # out and costs another full encode.
            refine_aim = target + (VMAF_OVERSHOOT - cfg["vmaf_tolerance"]) / 2
            full_points = {}

            if best_vmaf and math.isfinite(best_vmaf.get("mean", float("nan"))):
                full_points[best_q] = {
                    "vmaf": best_vmaf, "kbps": actual_kbps_now,
                }

            for _ in range(4):
                if not (best_vmaf
                        and math.isfinite(best_vmaf.get("mean", float("nan")))):
                    break
                vm_mean = best_vmaf["mean"]
                cur_kbps = full_points.get(best_q, {}).get("kbps")

                deficits = []
                d = target - vm_mean
                if d > cfg["vmaf_tolerance"]:
                    deficits.append(("VMAF", max(
                        grid.step,
                        grid.quantize((refine_aim - vm_mean) / slope_v),
                    )))
                if min_kbps and cur_kbps and cur_kbps < min_kbps:
                    if cur_kbps >= min_kbps * (1 - ENDGAME_SNAP_GAIN):
                        # Endgame-snap economics, deficit side: a full
                        # re-encode lifting bitrate by under
                        # ENDGAME_SNAP_GAIN buys nothing real — the floor
                        # is a starvation backstop, not a target (a real
                        # file re-encoded over a 4kbps shortfall).
                        # point_ok below waives the same sliver so final
                        # selection keeps this point.
                        if not deficits:
                            print(
                                f"{label('refine')}{cur_kbps}kbps is within"
                                f" {ENDGAME_SNAP_GAIN:.0%} of the"
                                f" {min_kbps}kbps floor — accepting"
                            )
                    else:
                        # Aim at the CENTER of the bitrate band [floor,
                        # floor × BITRATE_BAND], not at the floor edge:
                        # the jump is a model prediction, and against an
                        # edge aim any under-prediction lands short and
                        # costs a whole extra encode (a real file jumped
                        # to the edge and landed 1796kbps against an 1800
                        # floor). grid.ceil keeps the rounding bias on
                        # the safe (above-floor) side — overshooting the
                        # band top re-encodes nothing, undershooting the
                        # floor does.
                        step_b = max(
                            grid.step,
                            grid.ceil(
                                math.log(
                                    min_kbps * math.sqrt(BITRATE_BAND)
                                    / cur_kbps
                                ) / decay_b
                            ),
                        )
                        deficits.append(("bitrate", step_b))

                if deficits:
                    if best_q <= min_q:
                        short_names = ", ".join(n for n, _ in deficits)
                        print(
                            f"{label('refine')}at min {engine.qname}"
                            f" {BOLD}{grid.fmt(min_q)}{RESET},"
                            f" accepting ({short_names} short)"
                        )
                        break
                    step = max(s for _, s in deficits)
                    try_q = grid.quantize(max(min_q, best_q - step))
                    desc = ", ".join(n for n, _ in deficits) + " short"
                else:
                    overshoot = vm_mean - target
                    # Already inside the bitrate band [floor, floor × BAND]:
                    # the video has hit the floor closely enough, so accept
                    # it instead of spending another full encode trimming a
                    # few percent of bitrate the band already allows. This
                    # is what stops the floor-bound crawl (e.g. 5330kbps
                    # over a 5000 floor is done, not a step toward 5000).
                    in_band = (
                        min_kbps and cur_kbps
                        and cur_kbps <= min_kbps * BITRATE_BAND
                    )
                    # Bitrate headroom caps how far the quantizer can rise
                    # (log-linear model, same as the search). It aims at
                    # the CENTER of the bitrate band, like the deficit
                    # jump: aimed at the floor edge, any over-read of the
                    # decay lands under the floor (a real file came out at
                    # 1760kbps against an 1800 floor), and a landing past
                    # the waiver buys a third encode. Rounded to the
                    # nearest step, but never past the floor edge itself.
                    # A file that's over target because the floor pinned
                    # its quantizer gets ceiling == best_q and is accepted
                    # as-is.
                    ceiling = max_q
                    if min_kbps and cur_kbps:
                        headroom = min(
                            grid.quantize(math.log(
                                cur_kbps
                                / (min_kbps * math.sqrt(BITRATE_BAND))
                            ) / decay_b),
                            grid.floor(math.log(cur_kbps / min_kbps) / decay_b),
                        )
                        ceiling = min(
                            ceiling, grid.quantize(best_q + max(0, headroom))
                        )
                    # tol past the band top is measurement-noise
                    # hysteresis: tol is the declared VMAF noise epsilon
                    # (differences under it are noise everywhere else in
                    # the search), so a re-encode triggered by a
                    # sub-noise excess — 94.51 against a 94.50 band top
                    # on a real file — is spurious precision. Only the
                    # is-it-worth-redoing edge widens; the aim below
                    # stays at the band center.
                    if (overshoot <= VMAF_OVERSHOOT + cfg["vmaf_tolerance"]
                            or best_q >= ceiling or in_band):
                        break
                    step = max(
                        grid.step,
                        grid.quantize((vm_mean - refine_aim) / slope_v),
                    )
                    try_q = grid.quantize(min(ceiling, best_q + step))
                    # Same economics as the search's endgame snap: a full
                    # re-encode predicted to trim under ENDGAME_SNAP_GAIN
                    # of bitrate costs more than it buys — without this,
                    # refine would spend the encode the search just saved.
                    if ((try_q - best_q) * decay_b
                            < -math.log1p(-ENDGAME_SNAP_GAIN)):
                        print(
                            f"{label('refine')}{engine.qname}"
                            f" {grid.fmt(try_q)} would trim under"
                            f" {ENDGAME_SNAP_GAIN:.0%} bitrate — keeping"
                            f" {engine.qname} {BOLD}{grid.fmt(best_q)}{RESET}"
                        )
                        break
                    desc = f"VMAF {overshoot:.1f} over target"

                if try_q == best_q or try_q in full_points:
                    break

                print(
                    f"{label('refine')}{desc} -> {engine.qname}"
                    f" {BOLD}{grid.fmt(try_q)}{RESET}"
                    f" {DIM}(jump {grid.fmt_delta(try_q - best_q)}){RESET}"
                )
                full_encode(try_q)
                t0 = time.time()
                adj = measure(filepath, dst_path(try_q), try_q)
                if (math.isfinite(adj.get("mean", float("nan")))
                        and try_q not in s2_seen):
                    s2_seen[try_q] = engine.ssimu2_info(
                        filepath, dst_path(try_q), meta, cfg,
                        ref_index=full_idx,
                    )
                t_vmaf += time.time() - t0
                if not math.isfinite(adj.get("mean", float("nan"))):
                    break
                # A step onto a full-file search probe reuses its bitrate.
                adj_kbps = full_probes.get("kbps", {}).get(try_q)
                if adj_kbps is None:
                    adj_kbps = video_kbps(dst_path(try_q), meta["duration"])
                print(
                    f"{'':>{LABEL_W + 1}}"
                    f"{fmt_vmaf(adj, target, cfg['vmaf_tolerance'])}"
                    f"{fmt_s2(s2_seen.get(try_q))}"
                    f"{size_kbps_suffix(dst_path(try_q), adj_kbps)}"
                )
                full_points[try_q] = {"vmaf": adj, "kbps": adj_kbps}
                best_q, best_vmaf = try_q, adj

                # Re-fit both models from the measured full-encode points.
                qs_v = sorted(
                    c for c, p in full_points.items()
                    if math.isfinite(p["vmaf"].get("mean", float("nan")))
                )
                if len(qs_v) >= 2:
                    c1v, c2v = qs_v[0], qs_v[-1]
                    m = (full_points[c1v]["vmaf"]["mean"]
                         - full_points[c2v]["vmaf"]["mean"]) / (c2v - c1v)
                    if m > 0:
                        slope_v = clamp(m, VMAF_SLOPE_MIN, VMAF_SLOPE_MAX)
                qs_b = sorted(c for c, p in full_points.items() if p["kbps"])
                if len(qs_b) >= 2:
                    c1b, c2b = qs_b[0], qs_b[-1]
                    b1, b2 = full_points[c1b]["kbps"], full_points[c2b]["kbps"]
                    if b1 > 0 and b2 > 0:
                        m = math.log(b1 / b2) / (c2b - c1b)
                        if m > 0:
                            decay_b = clamp(m, DECAY_MIN, DECAY_MAX)

            # The loop can end on an invalid point (e.g. an overshoot
            # probe that undershot while its bounce-back quantizer was
            # already tested). Settle on the highest tested quantizer
            # that satisfies both gates; if none do, the lowest tested
            # one is the closest miss.
            def point_ok(p):
                vm_p = p["vmaf"].get("mean", float("nan"))
                if math.isfinite(vm_p) and vm_p < target - cfg["vmaf_tolerance"]:
                    return False
                # Waive the same hairline floor shortfall the refine loop
                # accepts (ENDGAME_SNAP_GAIN), or selection would discard
                # the point refine just deemed not worth re-encoding.
                if (min_kbps and p["kbps"]
                        and p["kbps"] < min_kbps * (1 - ENDGAME_SNAP_GAIN)):
                    return False
                return True

            if full_points:
                valid = [c for c in full_points if point_ok(full_points[c])]
                pick = max(valid) if valid else min(full_points)
                if pick != best_q and dst_path(pick).exists():
                    best_q = pick
                    best_vmaf = full_points[pick]["vmaf"]

            final = dst_path(best_q)
            if not final.exists():
                print(f" {CROSS} Final encode missing")
                _file_error = True
                continue

            # Every other output at this file's names goes: probes, refine
            # steps, and anything a run under other settings left. A
            # forced encode stays, recognized by its forced block entry
            # at the size on disk: a ladder of forced values is the A/B
            # workflow, and a searched run must not clear it.
            forced = cache.get("forced")
            forced = forced if isinstance(forced, dict) else {}
            for c in all_qs:
                d = dst_path(c)
                if c == best_q or not d.exists():
                    continue
                fe = forced.get(grid.fmt(c))
                if (isinstance(fe, dict) and fe.get("crop") == meta["crop"]
                        and fe.get("size") == d.stat().st_size):
                    continue
                try:
                    d.unlink()
                except OSError:
                    continue
                outputs.pop(grid.fmt(c), None)

            # The final output exists, so this file's segment work dirs
            # (any quantizer — refine may have left several) are spent.
            core_segments.cleanup_file_segments(root_cache, file_hash)

            out_sz = final.stat().st_size
            larger = out_sz >= in_sz
            if larger:
                final.unlink()
                outputs.pop(grid.fmt(best_q), None)

            # The file is finished: one write, the last of this file's
            # cache writes, records where refine settled and how the
            # file ended (see the skip-existing check). A stop before it
            # leaves the file pending, and the next run resumes it. An
            # unmeasured verify stays pending, so the next run measures
            # it again.
            rec_now = cache["recommended"]
            rec_now[engine.rec_q_key] = best_q
            if larger:
                rec_now["outcome"] = "larger"
            elif math.isfinite(best_vmaf["mean"]):
                rec_now["outcome"] = "kept"
            atomic_write_json(cp, cache)

            if larger:
                stats["deleted"] += 1
                print(
                    f" {CROSS} Larger than the source"
                    f" ({BOLD}{fmt_size(out_sz)}{RESET} vs"
                    f" {BOLD}{fmt_size(in_sz)}{RESET}), deleted;"
                    f" the source is kept"
                )
                continue

            # Final SSIMU2 info: reuse the search/verify/refine measurement
            # of this exact encode when there is one, otherwise (a
            # floor-bound full-file probe, whose VMAF and SSIMU2 were
            # skipped) measure once now. Membership, not the value:
            # a None there is a skip or a failure already announced, and
            # measuring again would only repeat it.
            if best_q in s2_seen:
                extra_s2 = s2_seen[best_q]
            else:
                t0 = time.time()
                extra_s2 = engine.ssimu2_info(
                    filepath, final, meta, cfg, ref_index=full_idx,
                )
                t_vmaf += time.time() - t0

            saved = (1.0 - out_sz / in_sz) * 100
            # This encode's video bitrate was already read: by the refine
            # loop for a point it measured, otherwise by the verify or the
            # full-file search (the only point outside full_points is the
            # one whose VMAF measurement failed).
            out_kbps = result_kbps(
                full_points[best_q]["kbps"] if best_q in full_points
                else actual_kbps_now,
                out_sz, meta["duration"],
            )
            # Output bitrate rides the result line next to VMAF (where the
            # eye looks for "how did this encode turn out"); the size line
            # below stays size + saved%.
            kbps_final = (
                f"  {DIM}{MIDDOT}{RESET}  {BOLD}{out_kbps}kbps{RESET}"
                if out_kbps else ""
            )
            in_str = fmt_size(in_sz)
            out_str = fmt_size(out_sz)
            print(SEP)
            print(
                f" {CHECK} {engine.qname} {BOLD}{grid.fmt(best_q)}{RESET}"
                f"  {fmt_vmaf(best_vmaf, target, cfg['vmaf_tolerance'])}"
                f"{kbps_final}"
            )
            if extra_s2:
                print(
                    f"   {DIM}SSIMU2 {extra_s2['mean']:.2f}"
                    f"  P5 {extra_s2['p5']:.2f}  (info only){RESET}"
                )
            print(
                f" {CHECK} {in_str} -> {BOLD}{out_str}{RESET}"
                f" saved {GREEN}{BOLD}{saved:.1f}%{RESET}"
            )
            print(f"   {DIM}Enc {fmt_time(t_enc)} {MIDDOT} VMAF {fmt_time(t_vmaf)}{RESET}")

            stats["proc"] += 1
            if math.isfinite(best_vmaf["mean"]):
                stats["vmaf_sum"] += best_vmaf["mean"]
                stats["vmaf_n"] += 1
            stats["saved"] += in_sz - out_sz
            stats["orig"] += in_sz

        except KeyboardInterrupt:
            _file_error = True
            raise
        except Exception as e:
            _file_error = True
            print(f" {CROSS} {e}")
        finally:
            cleanup_temp()
            # A failed file is counted and keeps its samples for the
            # rerun; a finished one deletes them. A partial clip set is a
            # temp and is already gone either way.
            if _file_error:
                stats["failed"] += 1
            else:
                for p in (sample_src, sample_concat):
                    if p:
                        try:
                            if p.exists():
                                p.unlink()
                        except OSError:
                            pass
            # The search source's index goes with the file it indexes:
            # left behind, it is junk no later run ever reads.
            if sample_idx:
                try:
                    if not sample_src.exists():
                        sample_idx.unlink(missing_ok=True)
                except OSError:
                    pass

    print(SEP)
    if stats["proc"] > 0:
        pct = stats["saved"] / stats["orig"] * 100 if stats["orig"] else 0
        print(f"{CHECK} Processed: {BOLD}{stats['proc']}{RESET}")
        # Forced runs measure no VMAF; an average over zero scores would
        # print a bogus 0.00.
        if stats["vmaf_n"]:
            avg = stats["vmaf_sum"] / stats["vmaf_n"]
            print(f"{CHECK} Avg VMAF: {BOLD}{avg:.2f}{RESET}")
        print(
            f"{CHECK} Saved: {GREEN}{BOLD}{stats['saved'] / 1e9:.2f}GB{RESET}"
            f" ({GREEN}{BOLD}{pct:.1f}%{RESET})"
        )
        print(f"{CHECK} Time: {BOLD}{fmt_time(time.time() - t_start)}{RESET}")
    else:
        print(f"{CHECK} No files processed")
    if stats["deleted"]:
        print(
            f"{ORANGE} Larger than the source: {BOLD}{stats['deleted']}{RESET}"
            f"{ORANGE} (deleted, sources kept){RESET}"
        )
    if stats["failed"]:
        print(f"{CROSS} Failed: {BOLD}{stats['failed']}{RESET}")

    print(f"{SEP}\n{CHECK} Done")
    return 1 if stats["failed"] else 0
