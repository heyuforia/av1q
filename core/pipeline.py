"""The per-file processing pipeline shared by av1q and av1q-essential.

process_videos drives: discovery → skip-existing → probe → crop →
engine gate/meta prep → scene analysis → sampling (engine sample prep)
→ search (core.search) → final encode + verify → calibration
persistence → the consolidated refine loop → final selection and
cleanup. Everything engine-specific goes through the Engine interface;
the printed output is byte-identical to what each launcher printed
before the merge (quantizer labels and grid formatting come from the
engine).

Two cache scopes, deliberately distinct:
  cfg["cache_dir"]        shared between pipelines — sample extraction,
                          crop temp logs (the formats are identical).
  engine.cache_root(cfg)  per-pipeline — result caches, calibration,
                          sample encodes, FFMS2 indexes. Different
                          encoders must never share these.
"""

import math
import os
import shutil
import sys
import time

from . import search as core_search
from . import vmaf as core_vmaf
from .analyze import analyze_complexity, detect_scenes, get_keyframes
from .bitrate import calc_kbps, video_kbps
from .cache import load_cache
from .calibrate import (
    DECAY_MAX, DECAY_MIN, calibration_offset, decay_prior,
    load_global_calibration, update_global_calibration,
)
from .constants import (
    DEFAULT_BITRATE_DECAY, ENDGAME_SNAP_GAIN, INTRA_ONLY_CODECS,
    MIN_BITRATE_KBPS, MINI_SAMPLE_COUNT, MINI_SAMPLE_DURATION,
    MINI_SAMPLE_MIN_RATIO, TARGET_VMAF_BY_RES, VIDEO_EXTENSIONS,
    VMAF_OVERSHOOT,
)
from .crop import crop_token, detect_crop_for_file, load_crop_sidecar
from .probe import get_fps, probe_video, res_tier
from .sampling import extract_samples, sampling_plan, select_samples
from .ui import (
    BOLD, CHECK, CROSS, DIM, GREEN, MIDDOT, ORANGE, PURPLE, RED, RESET, SEP,
    fmt_s2, fmt_size, fmt_time, vmaf_pass_color,
)
from .util import atomic_write_json, clamp, cleanup_temp, partial_hash


def process_videos(cfg, engine):
    grid = engine.grid
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    cache_dir = cfg["cache_dir"]
    ext = cfg["container"]
    vmaf_threads = os.cpu_count() or 4

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        print(f"{CROSS} ffmpeg/ffprobe not found in PATH")
        return 1
    try:
        engine.setup(cfg)
    except FileNotFoundError as e:
        print(f"{CROSS} {e}")
        return 1

    root_cache = engine.cache_root(cfg)
    min_q, max_q = engine.q_bounds(cfg)

    LBL = 10  # label column width for aligned output
    def lbl(tag):
        return f" {ORANGE}{tag:<{LBL}}{RESET}"

    print(f"{PURPLE}{BOLD}{engine.banner}{RESET}{engine.banner_extra}\n{SEP}")

    # Interactive seed prompt: lets a batch of similar files start the
    # search at a known-good quantizer instead of the automatic seed
    # (which falls back to 30 for intra-only sources like ProRes). Enter
    # keeps auto behavior. Only when stdin is a terminal — piped/scripted
    # runs must not block.
    if cfg[engine.seed_key] is None and sys.stdin.isatty():
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

    for pat in engine.tmp_patterns:
        for p in output_dir.rglob(pat):
            try:
                p.unlink()
            except OSError:
                pass

    pattern = "**/*" if cfg["recurse"] else "*"
    files = [
        f for f in input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    total = len(files)

    # A user seed only seeds NEW searches: files with a completed search
    # resume past it (and verified outputs skip entirely), which reads as
    # the seed being silently ignored. Offer the choice up front, once
    # for the whole batch — a per-file prompt would stall unattended
    # runs partway through. Yes clears those files' caches so they get a
    # fresh search from the seed; the default keeps today's behavior.
    user_seed = cfg[engine.seed_key]
    seeded_redo = set()
    if user_seed is not None and files and sys.stdin.isatty():
        prior = []
        for f in files:
            try:
                fh = partial_hash(f)
            except OSError:
                continue
            c, _ = load_cache(root_cache, fh, engine.sig)
            rec = c.get("recommended")
            if (isinstance(rec, dict)
                    and all(
                        rec.get(k) == cfg[k]
                        for k in (*engine.rec_bound_keys, "preset",
                                  "film_grain", *engine.rec_extra_keys)
                    )
                    and rec.get("crop") == (
                        load_crop_sidecar(f, fh) if cfg["use_crops"] else None
                    )
                    and (cfg["target_vmaf"] is None
                         or rec.get("target") == cfg["target_vmaf"])):
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

    stats = {
        "proc": 0, "vmaf_sum": 0.0, "vmaf_n": 0,
        "saved": 0, "orig": 0, "deleted": 0,
    }
    t_start = time.time()
    global_cal = load_global_calibration(root_cache)

    # Whether (and how) a file gets sampled is sampling_plan's call:
    # the configured plan for long sources, a scaled-down mini plan for
    # short ones, full-file search only below the mini amortization gate.
    mini_min = MINI_SAMPLE_COUNT * MINI_SAMPLE_DURATION * MINI_SAMPLE_MIN_RATIO

    all_qs = grid.span(min_q, max_q)

    for idx, filepath in enumerate(files, 1):
        sample_src = sample_concat = None
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
                # result: drop this file's whole cache (entries,
                # calibration, scene data, recommended) so nothing
                # resumes or skips below.
                try:
                    cp.unlink()
                except OSError:
                    pass
                cache, cp = load_cache(root_cache, file_hash, engine.sig)

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
                load_crop_sidecar(filepath, file_hash)
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
                # An output file with a cached full VMAF is necessary but
                # not sufficient: an interrupted search can leave a probe
                # encode that later gets verified, and a verified-but-
                # unconverged file (e.g. seed quantizer at 4x the intended
                # bitrate) must not be skipped forever. Require the
                # cache's `recommended` block — written only when a search
                # completes — to match the current settings, and accept
                # either the recommended quantizer itself or (for outputs
                # predating the post-refine `recommended` update) a
                # verified VMAF inside the acceptance band.
                verified = False
                rec = cache.get("recommended")
                rec_ok = (
                    isinstance(rec, dict)
                    and all(
                        rec.get(k) == cfg[k]
                        for k in (*engine.rec_bound_keys, "preset",
                                  "film_grain", *engine.rec_extra_keys)
                    )
                    and rec.get("crop") == expected_crop
                    # Auto targets vary by resolution and the file hasn't
                    # been probed yet, so a target check is only possible
                    # against an explicit --vmaf.
                    and (cfg["target_vmaf"] is None
                         or rec.get("target") == cfg["target_vmaf"])
                )
                if rec_ok:
                    rec_target = rec.get("target")
                    for c in all_qs:
                        d = dst_path(c)
                        if not d.exists():
                            continue
                        entry = cache["entries"].get(grid.fmt(c))
                        if not (entry and engine.vmaf_key_base in entry
                                and entry.get("size") == d.stat().st_size):
                            continue
                        in_band = (
                            isinstance(rec_target, (int, float))
                            and rec_target - cfg["vmaf_tolerance"]
                            <= entry[engine.vmaf_key_base]
                            <= rec_target + VMAF_OVERSHOOT
                        )
                        if c == rec.get(engine.rec_q_key) or in_band:
                            verified = True
                            break
                if verified:
                    print(f" {PURPLE}{filepath.name:<30}{RESET} {CHECK} exists")
                    continue

            if idx > 1:
                print(SEP)
            print(f"{PURPLE}{BOLD}[{idx}/{total}]{RESET} {PURPLE}{filepath.name}{RESET}")
            if file_hash in seeded_redo:
                print(
                    f"{lbl('redo')}{DIM}previous results cleared, searching"
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

            if meta["codec"] == "av1":
                print(f" {CHECK} Already AV1, skipping")
                continue

            # Engine gate: sources this engine cannot process (e.g. VFR
            # for the CFR-only Y4M pipe).
            gate_reason = engine.gate(filepath, meta)
            if gate_reason:
                print(f" {CROSS} {gate_reason}")
                continue

            just_detected = False
            if cfg["auto_crop"]:
                sidecar = filepath.with_suffix(filepath.suffix + ".crop.json")
                if not sidecar.exists():
                    crop_cfg = {
                        "cache_dir": cache_dir,
                        "sample_count": 8,
                        "window_duration": 2.0,
                        "limit_sdr": 24,
                        "limit_hdr": 128,
                        "round": 2,
                        "min_keep_ratio": 0.10,
                        "agree_ratio": 0.75,
                        "scene_threshold": cfg["scene_threshold"],
                        "short_threshold": cfg["short_threshold"],
                    }
                    try:
                        data = detect_crop_for_file(
                            filepath, meta, crop_cfg, file_hash
                        )
                        atomic_write_json(sidecar, data, indent=2)
                        just_detected = True
                    except Exception as e:
                        print(f"{lbl('crop err')}{e}")

            meta["crop"] = None
            if cfg["use_crops"]:
                crop = load_crop_sidecar(filepath, file_hash)
                if crop:
                    meta["crop"] = crop
                    # detect_crop_for_file already printed the summary line
                    if not just_detected:
                        print(f"{lbl('crop')}{BOLD}{crop}{RESET}")
            dst_path = make_dst_path(meta["crop"])

            # Engine-specific metadata enrichment (e.g. HDR10 static
            # metadata restated as encoder flags on the Y4M-pipe path).
            if engine.prepare_meta(filepath, meta, cfg):
                print(f"{lbl('hdr')}{DIM}static metadata carried over{RESET}")

            tier = max(k for k in TARGET_VMAF_BY_RES if min(meta["w"], meta["h"]) >= k)
            target = cfg.get("target_vmaf") or TARGET_VMAF_BY_RES[tier]

            # Persistent FFMS2 reference index for this source (SSIMU2
            # info column only — display, never gating).
            full_idx = engine.full_ref_index(cfg, filepath, file_hash, in_sz)

            # Resume only from the cache's `recommended` block, written
            # when a search completes. A bare output file at some
            # quantizer is NOT evidence of a finished search: the
            # full-file search path writes its probe encodes straight to
            # the output dir, so an interrupted run leaves the seed
            # encode behind — trusting it shipped seed-quality files at
            # several times the intended bitrate. Leftover probes are
            # still reused (do_enc_full skips existing files, VMAF is
            # cached by size), so re-running the search after an
            # interruption stays cheap.
            existing_q = None
            rec = cache.get("recommended")
            if (rec and rec.get("target") == target
                    and all(
                        rec.get(k) == cfg[k]
                        for k in (*engine.rec_bound_keys, "preset",
                                  "film_grain", *engine.rec_extra_keys)
                    )
                    and rec.get("crop") == meta["crop"]):
                existing_q = grid.quantize(rec[engine.rec_q_key])
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
                    f"{lbl('resume')}{engine.qname} {BOLD}{grid.fmt(existing_q)}{RESET}"
                    f" from previous search{seed_note}"
                )

            sample_scenes = sample_src = None
            plan = sampling_plan(meta["duration"], cfg)

            if existing_q is None and plan:
                n_samples, s_dur, plan_mode = plan
                if plan_mode == "mini":
                    print(
                        f"{lbl('short')}{meta['duration']:.0f}s source →"
                        f" mini-samples ({n_samples}×{s_dur:.0f}s)"
                    )
                if meta["codec"] in INTRA_ONLY_CODECS:
                    print(f"{lbl('skip')}Intra-only codec ({meta['codec']}), using even samples")
                    scenes = []
                    complexity = []
                    keyframes = []
                else:
                    scene_cfg = {"scene_threshold": cfg["scene_threshold"]}
                    if (all(k in cache for k in ("scenes", "complexity", "keyframes"))
                            and cache.get("scene_cfg") == scene_cfg):
                        print(f"{lbl('cache')}Using cached scene data")
                        scenes = cache["scenes"]
                        complexity = cache["complexity"]
                        keyframes = cache["keyframes"]
                    else:
                        print(f"{lbl('analyze')}Detecting scenes...")
                        scenes = detect_scenes(filepath, cfg)
                        complexity = analyze_complexity(filepath)
                        keyframes = get_keyframes(filepath)
                        cache.update(scenes=scenes, complexity=complexity,
                                     keyframes=keyframes, scene_cfg=scene_cfg)
                        atomic_write_json(cp, cache)

                # The plan already decided sampling applies, so disarm
                # select_samples' own short-file bail-out and use the
                # plan's clip length (mini plans cut shorter clips).
                sample_scenes = select_samples(
                    scenes, complexity, meta["duration"], n_samples,
                    keyframes,
                    {**cfg, "sample_duration": s_dur, "short_threshold": 0},
                )
                if sample_scenes:
                    info = (
                        f"samples from {BOLD}{len(scenes)}{RESET} scenes"
                        if scenes else "evenly-spaced samples"
                    )
                    print(f"{lbl('scenes')}{BOLD}{len(sample_scenes)}{RESET} {info}")
                    print(f"{lbl('extract')}Extracting samples...")
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
                        print(f"{lbl('fallback')}Extraction failed, using full encode")
                        sample_scenes = None
                else:
                    print(f"{lbl('scenes')}Using full VMAF")
            elif existing_q is None:
                print(f"{lbl('short')}≤{mini_min:.0f}s, full VMAF")

            t_enc = t_vmaf = 0.0
            sample_enc_dir = root_cache / "_sample_enc"
            sample_enc_dir.mkdir(parents=True, exist_ok=True)
            sample_enc_cache = {}
            enc_tag = engine.signature(cfg, meta.get("crop"))

            expected_frames = 0
            if engine.needs_expected_frames:
                fps_str = get_fps(filepath)
                if fps_str and meta["duration"] > 0:
                    try:
                        if "/" in fps_str:
                            a, b = fps_str.split("/", 1)
                            fps_f = float(a) / float(b)
                        else:
                            fps_f = float(fps_str)
                        expected_frames = int(meta["duration"] * fps_f)
                    except (ValueError, ZeroDivisionError):
                        pass

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

            def do_enc_full(q):
                nonlocal t_enc
                q = grid.quantize(clamp(q, min_q, max_q))
                d = dst_path(q)
                if not d.exists():
                    t0 = time.time()
                    engine.encode(
                        filepath, d, meta, q, cfg,
                        show_progress=True, expected_frames=expected_frames,
                    )
                    t_enc += time.time() - t0
                return d

            min_kbps = MIN_BITRATE_KBPS.get(res_tier(meta["w"], meta["h"]), 0)
            floor_str = (
                f" {DIM}{MIDDOT}{RESET} floor {BOLD}{min_kbps}kbps{RESET}"
                if min_kbps else ""
            )
            print(f"{lbl('target')}VMAF {BOLD}{target:.1f}{RESET}{floor_str}")

            # Engine-cohort bitrate-decay prior for the search's floor
            # model: how fast THIS encoder's bitrate falls per quantizer
            # step (Essential's CRF encodes richer than mainline's CQ at
            # equal numbers). Sizes the first jump toward the floor for
            # the engine instead of the generic ±6 ≈ 2× cold-start, which
            # cost essential 1-2 extra probes per file.
            dec_prior, dec_src = decay_prior(
                cache.get("calibration"), global_cal
            )
            if (min_kbps and dec_prior is not None
                    and abs(dec_prior - DEFAULT_BITRATE_DECAY)
                    >= 0.15 * DEFAULT_BITRATE_DECAY):
                print(
                    f"{lbl('calibr')}bitrate decay {BOLD}{dec_prior:.3f}{RESET}"
                    f"/{engine.qname} {DIM}({dec_src}){RESET}"
                )

            if existing_q is not None:
                best_q = existing_q
            elif sample_src:
                # Apply learned VMAF offset (sample over/under-predicts
                # full VMAF) so the sample search aims at the quantizer
                # that will hit `target` on the full video. Per-file
                # calibration takes precedence; on first encounter we fall
                # back to the cohort average (shrunk toward 0 while the
                # cohort is small) so new files start informed instead of
                # paying a second-encode tax to learn the offset.
                sample_target = target
                off, off_src = calibration_offset(
                    cache.get("calibration"), global_cal
                )
                if off is not None and abs(off) >= 0.1:
                    sample_target = clamp(target + off, 0.0, 100.0)
                    print(
                        f"{lbl('calibr')}sample target"
                        f" {BOLD}{sample_target:.2f}{RESET}"
                        f" {DIM}(offset {off:+.2f} {off_src}){RESET}"
                    )
                best_q, _, _, vt, search_state = core_search.search(
                    sample_src, meta, sample_target, cache, cp,
                    do_enc_sample, cfg, engine, tag="sample",
                    decay_prior=dec_prior,
                    measure_fn=lambda ref, dist, q: measure(
                        ref, dist, q, tag="sample"),
                    probe_fn=probe_video,
                    s2_fn=lambda ref, dist, m, ri: engine.ssimu2_info(
                        ref, dist, m, cfg, ref_index=ri),
                    s2_ref_index=engine.sample_ref_index(cfg, sample_src),
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
                    do_enc_full, cfg, engine, decay_prior=dec_prior,
                    measure_fn=lambda ref, dist, q: measure(ref, dist, q),
                    probe_fn=probe_video,
                    s2_fn=lambda ref, dist, m, ri: engine.ssimu2_info(
                        ref, dist, m, cfg, ref_index=ri),
                    s2_ref_index=full_idx,
                )
                t_vmaf += vt

            if best_q is None:
                print(f" {CROSS} No valid {engine.qname} found")
                continue

            # Mark the search as completed for BOTH search paths. Written
            # before the final encode so an interruption resumes here;
            # the quantizer is synced again after the refine loop if
            # refinement moves it.
            if existing_q is None:
                cache["recommended"] = {
                    engine.rec_q_key: best_q, "target": target,
                    engine.rec_bound_keys[0]: cfg[engine.rec_bound_keys[0]],
                    engine.rec_bound_keys[1]: cfg[engine.rec_bound_keys[1]],
                    "preset": cfg["preset"], "film_grain": cfg["film_grain"],
                    **{k: cfg[k] for k in engine.rec_extra_keys},
                    "crop": meta["crop"],
                }
                atomic_write_json(cp, cache)

            if cfg["dry_run"]:
                entry = cache["entries"].get(grid.fmt(best_q), {})
                vmaf_str = ""
                sv = (entry.get(f"sample_{engine.vmaf_key_base}")
                      or entry.get(engine.vmaf_key_base))
                if isinstance(sv, dict):
                    vmaf_str = f" VMAF {BOLD}{sv['mean']:.2f}{RESET}  P5 {BOLD}{sv['p5']:.2f}{RESET}"
                elif isinstance(sv, (int, float)):
                    vmaf_str = f" VMAF {BOLD}{sv:.2f}{RESET}"
                print(
                    f" {CHECK} Recommended {engine.qname}"
                    f" {BOLD}{grid.fmt(best_q)}{RESET}{vmaf_str}"
                )
                print(f"   Run without --dry-run to encode")
                continue

            # SSIMU2 info per full-encode quantizer (display only).
            s2_seen = {}

            # Final full encode at the candidate quantizer + VMAF verify
            if sample_src or existing_q is not None:
                if not dst_path(best_q).exists():
                    t0 = time.time()
                    engine.encode(
                        filepath, dst_path(best_q), meta, best_q, cfg,
                        show_progress=True, expected_frames=expected_frames,
                    )
                    t_enc += time.time() - t0
                else:
                    print(
                        f"{lbl('reuse')}{engine.qname}"
                        f" {BOLD}{grid.fmt(best_q)}{RESET} encode exists"
                    )

                print(f"{lbl('verify')}Full VMAF...")
                t0 = time.time()
                best_vmaf = measure(filepath, dst_path(best_q), best_q)
                s2_seen[best_q] = engine.ssimu2_info(
                    filepath, dst_path(best_q), meta, cfg, ref_index=full_idx,
                )
                t_vmaf += time.time() - t0
                vc = vmaf_pass_color(best_vmaf["mean"], target, cfg["vmaf_tolerance"])
                print(
                    f"{'':>{LBL + 1}}VMAF {BOLD}{vc}{best_vmaf['mean']:.2f}{RESET}"
                    f"  {DIM}P5 {best_vmaf['p5']:.2f}{RESET}"
                    f"{fmt_s2(s2_seen[best_q])}"
                )

            # Persist sample↔full calibration BEFORE the refine loop so a
            # resumed or repeated run of this file starts with both the
            # bitrate ratio and the VMAF offset.
            entry_at_best = cache.get("entries", {}).get(grid.fmt(best_q), {})
            sample_kbps_at_best = entry_at_best.get(f"sample_kbps_{enc_tag}")
            sample_vmaf_at_best = entry_at_best.get(
                f"sample_{engine.vmaf_key_base}"
            )
            actual_kbps_now = (
                video_kbps(dst_path(best_q), meta["duration"])
                if meta["duration"] > 1 and dst_path(best_q).exists()
                else None
            )

            cal_now = cache.get("calibration")
            cal_now = dict(cal_now) if isinstance(cal_now, dict) else {}
            cal_updated = False
            fresh_offset = None
            fresh_ratio = None
            fresh_decay = None

            if (sample_kbps_at_best and actual_kbps_now
                    and sample_kbps_at_best > 0):
                ratio = actual_kbps_now / sample_kbps_at_best
                if 0.5 <= ratio <= 1.0:
                    cal_now["ratio"] = ratio
                    cal_updated = True
                    fresh_ratio = ratio
                    print(
                        f"{lbl('calibr')}sample {sample_kbps_at_best}kbps ->"
                        f" video {actual_kbps_now}kbps (ratio {ratio:.2f})"
                    )

            if (sample_vmaf_at_best is not None and best_vmaf
                    and math.isfinite(sample_vmaf_at_best)
                    and math.isfinite(best_vmaf.get("mean", float("nan")))):
                offset = sample_vmaf_at_best - best_vmaf["mean"]
                if -3.0 <= offset <= 3.0:
                    cal_now["vmaf_offset"] = offset
                    cal_updated = True
                    fresh_offset = offset

            if search_state and search_state.get("vmaf_slope"):
                sl = search_state["vmaf_slope"]
                if 0.1 <= sl <= 2.0:
                    cal_now["vmaf_slope"] = sl
                    cal_updated = True

            # Bitrate decay actually measured by this search's probes
            # (never the cold-start default — search reports those as
            # None) feeds the per-file calibration and the engine cohort.
            md = search_state.get("measured_decay") if search_state else None
            if isinstance(md, (int, float)) and DECAY_MIN <= md <= DECAY_MAX:
                cal_now["decay"] = md
                cal_updated = True
                fresh_decay = md

            if cal_updated:
                cal_now[engine.cal_q_key] = best_q
                cal_now["enc_tag"] = enc_tag
                cal_now["t"] = time.time()
                cache["calibration"] = cal_now
                atomic_write_json(cp, cache)

                # Roll fresh measurements into the cohort so subsequent
                # new files start with an informed prior. Only the
                # measurements taken this run are rolled in — values
                # carried over from a previous run aren't double-counted.
                # (Each engine has its own cohort file; different
                # encoders must never share calibration.)
                if (fresh_offset is not None or fresh_ratio is not None
                        or fresh_decay is not None):
                    update_global_calibration(
                        root_cache,
                        vmaf_offset=fresh_offset,
                        ratio=fresh_ratio,
                        decay=fresh_decay,
                    )
                    global_cal = load_global_calibration(root_cache)

            # Consolidated refine for quality/bitrate misses in BOTH
            # directions. Deficits (VMAF below target, bitrate below the
            # floor) step the quantizer down; overshoot (VMAF more than
            # VMAF_OVERSHOOT above target with bitrate headroom over the
            # floor) steps it up — without this, any candidate that
            # arrives here too low (e.g. a resumed quantizer verified
            # against a changed target) ships an oversized file. Each
            # move is a slope-sized jump, not a step-by-1, so it
            # converges in 1-2 encodes.
            slope_v = clamp(
                (search_state.get("vmaf_slope") if search_state else None) or 0.5,
                0.1, 2.0,
            )
            decay_b = clamp(
                (search_state.get("bitrate_decay") if search_state else None)
                or (math.log(2) / 6),
                0.05, 0.4,
            )
            full_points = {}

            def record_point(q, vm, kbps=None):
                if (kbps is None and min_kbps and meta["duration"] > 1
                        and dst_path(q).exists()):
                    kbps = video_kbps(dst_path(q), meta["duration"])
                full_points[q] = {"vmaf": vm, "kbps": kbps}

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
                    deficits.append(
                        ("VMAF", max(grid.step, grid.ceil(d / slope_v)))
                    )
                if min_kbps and cur_kbps and cur_kbps < min_kbps:
                    step_b = max(
                        grid.step,
                        grid.ceil(math.log(min_kbps / cur_kbps) / decay_b),
                    )
                    deficits.append(("bitrate", step_b))

                if deficits:
                    if best_q <= min_q:
                        short_names = ", ".join(n for n, _ in deficits)
                        print(
                            f"{lbl('refine')}at min {engine.qname}"
                            f" {BOLD}{grid.fmt(min_q)}{RESET},"
                            f" accepting ({short_names} short)"
                        )
                        break
                    step = max(s for _, s in deficits)
                    try_q = grid.quantize(max(min_q, best_q - step))
                    desc = ", ".join(n for n, _ in deficits) + " short"
                else:
                    overshoot = vm_mean - target
                    # Bitrate headroom over the floor caps how far the
                    # quantizer can rise (log-linear model, same as the
                    # search). A file that's over target because the floor
                    # pinned its quantizer gets ceiling == best_q and is
                    # accepted as-is.
                    ceiling = max_q
                    if min_kbps and cur_kbps:
                        headroom = grid.floor(
                            math.log(cur_kbps / min_kbps) / decay_b
                        )
                        ceiling = min(
                            ceiling, grid.quantize(best_q + max(0, headroom))
                        )
                    if overshoot <= VMAF_OVERSHOOT or best_q >= ceiling:
                        break
                    # Aim mid-band (flooring the step skews the landing
                    # toward the upper half) so slope error stays inside
                    # the acceptance band instead of below it.
                    step = max(
                        grid.step,
                        grid.floor((overshoot - VMAF_OVERSHOOT / 2) / slope_v),
                    )
                    try_q = grid.quantize(min(ceiling, best_q + step))
                    # Same economics as the search's endgame snap: a full
                    # re-encode predicted to trim under ENDGAME_SNAP_GAIN
                    # of bitrate costs more than it buys — without this,
                    # refine would spend the encode the search just saved.
                    if ((try_q - best_q) * decay_b
                            < -math.log1p(-ENDGAME_SNAP_GAIN)):
                        print(
                            f"{lbl('refine')}{engine.qname}"
                            f" {grid.fmt(try_q)} would trim under"
                            f" {ENDGAME_SNAP_GAIN:.0%} bitrate — keeping"
                            f" {engine.qname} {BOLD}{grid.fmt(best_q)}{RESET}"
                        )
                        break
                    desc = f"VMAF {overshoot:.1f} over target"

                if try_q == best_q or try_q in full_points:
                    break

                print(
                    f"{lbl('refine')}{desc} -> {engine.qname}"
                    f" {BOLD}{grid.fmt(try_q)}{RESET}"
                    f" {DIM}(jump {grid.fmt_delta(try_q - best_q)}){RESET}"
                )
                if not dst_path(try_q).exists():
                    t0 = time.time()
                    engine.encode(
                        filepath, dst_path(try_q), meta, try_q, cfg,
                        show_progress=True, expected_frames=expected_frames,
                    )
                    t_enc += time.time() - t0
                t0 = time.time()
                adj = measure(filepath, dst_path(try_q), try_q)
                if math.isfinite(adj.get("mean", float("nan"))):
                    s2_seen[try_q] = engine.ssimu2_info(
                        filepath, dst_path(try_q), meta, cfg,
                        ref_index=full_idx,
                    )
                t_vmaf += time.time() - t0
                if not math.isfinite(adj.get("mean", float("nan"))):
                    break
                vc_a = vmaf_pass_color(adj["mean"], target, cfg["vmaf_tolerance"])
                print(
                    f"{'':>{LBL + 1}}VMAF {BOLD}{vc_a}{adj['mean']:.2f}{RESET}"
                    f"  {DIM}P5 {adj['p5']:.2f}{RESET}"
                    f"{fmt_s2(s2_seen.get(try_q))}"
                )
                record_point(try_q, adj)
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
                        slope_v = clamp(m, 0.1, 2.0)
                qs_b = sorted(c for c, p in full_points.items() if p["kbps"])
                if len(qs_b) >= 2:
                    c1b, c2b = qs_b[0], qs_b[-1]
                    b1, b2 = full_points[c1b]["kbps"], full_points[c2b]["kbps"]
                    if b1 > 0 and b2 > 0:
                        m = math.log(b1 / b2) / (c2b - c1b)
                        if m > 0:
                            decay_b = clamp(m, 0.05, 0.4)

            # The loop can end on an invalid point (e.g. an overshoot
            # probe that undershot while its bounce-back quantizer was
            # already tested). Settle on the highest tested quantizer
            # that satisfies both gates; if none do, the lowest tested
            # one is the closest miss.
            def point_ok(p):
                vm_p = p["vmaf"].get("mean", float("nan"))
                if math.isfinite(vm_p) and vm_p < target - cfg["vmaf_tolerance"]:
                    return False
                if min_kbps and p["kbps"] and p["kbps"] < min_kbps:
                    return False
                return True

            if full_points:
                valid = [c for c in full_points if point_ok(full_points[c])]
                pick = max(valid) if valid else min(full_points)
                if pick != best_q and dst_path(pick).exists():
                    best_q = pick
                    best_vmaf = full_points[pick]["vmaf"]

            # Keep the resume point in sync with the refined result so a
            # rerun resumes at the final quantizer, not the pre-refine one.
            rec_now = cache.get("recommended")
            if isinstance(rec_now, dict) and rec_now.get(engine.rec_q_key) != best_q:
                rec_now[engine.rec_q_key] = best_q
                atomic_write_json(cp, cache)

            final = dst_path(best_q)
            if not final.exists():
                print(f" {CROSS} Final encode missing")
                continue

            for c in all_qs:
                if c != best_q and dst_path(c).exists():
                    try:
                        dst_path(c).unlink()
                    except OSError:
                        pass

            out_sz = final.stat().st_size

            if out_sz >= in_sz:
                final.unlink()
                stats["deleted"] += 1
                print(
                    f" {CROSS} Larger ({BOLD}{out_sz / 1e6:.1f}MB{RESET}"
                    f" vs {BOLD}{in_sz / 1e6:.1f}MB{RESET}) - deleted"
                )
                continue

            # Final SSIMU2 info: reuse the verify/refine measurement of
            # this exact encode when there is one, otherwise (full-file
            # search path) measure once now.
            extra_s2 = s2_seen.get(best_q)
            if extra_s2 is None:
                t0 = time.time()
                extra_s2 = engine.ssimu2_info(
                    filepath, final, meta, cfg, ref_index=full_idx,
                )
                t_vmaf += time.time() - t0

            saved = (1.0 - out_sz / in_sz) * 100
            out_kbps = calc_kbps(out_sz, meta["duration"])
            kbps_str = f" ({BOLD}{out_kbps}kbps{RESET})" if out_kbps else ""
            in_str = fmt_size(in_sz)
            out_str = fmt_size(out_sz)
            vc = vmaf_pass_color(best_vmaf["mean"], target, cfg["vmaf_tolerance"])
            print(SEP)
            print(
                f" {CHECK} {engine.qname} {BOLD}{grid.fmt(best_q)}{RESET}"
                f"  VMAF {BOLD}{vc}{best_vmaf['mean']:.2f}{RESET}"
                f"  {DIM}P5 {best_vmaf['p5']:.2f}{RESET}"
            )
            if extra_s2:
                print(
                    f"   {DIM}SSIMU2 {extra_s2['mean']:.2f}"
                    f"  P5 {extra_s2['p5']:.2f}  (info only){RESET}"
                )
            print(
                f" {CHECK} {in_str} -> {BOLD}{out_str}{RESET}{kbps_str}"
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
            if not _file_error:
                for p in (sample_src, sample_concat):
                    if p:
                        try:
                            if p.exists():
                                p.unlink()
                        except OSError:
                            pass

    if total > 0:
        print(SEP)
    if stats["proc"] > 0:
        avg = stats["vmaf_sum"] / stats["vmaf_n"] if stats["vmaf_n"] else 0
        pct = stats["saved"] / stats["orig"] * 100 if stats["orig"] else 0
        print(f"{CHECK} Processed: {BOLD}{stats['proc']}{RESET}")
        print(f"{CHECK} Avg VMAF: {BOLD}{avg:.2f}{RESET}")
        print(
            f"{CHECK} Saved: {GREEN}{BOLD}{stats['saved'] / 1e9:.2f}GB{RESET}"
            f" ({GREEN}{BOLD}{pct:.1f}%{RESET})"
        )
        if stats["deleted"] > 0:
            print(f"{ORANGE} Deleted: {BOLD}{stats['deleted']}{RESET}")
        print(f"{CHECK} Time: {BOLD}{fmt_time(time.time() - t_start)}{RESET}")
    else:
        print(f"{CHECK} No files processed")

    print(f"{SEP}\n{CHECK} Done")
    return 0
