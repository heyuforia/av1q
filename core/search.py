"""The adaptive quantizer search — the shared brain of both pipelines.

One implementation serves av1q (integer CQ grid) and av1q-essential
(quarter-step CRF grid): all stepping goes through the engine's Grid and
all measurement through injected closures, so the search never knows
which encoder or cache layout sits behind it. Grid bounds are on-grid,
so quantize-then-clamp equals clamp-then-quantize (see Grid in
core/engines/base.py).

Seeding lives here too: initial_cq_seed maps source-bitrate headroom
over the floor to a starting quantizer.
"""

import math
import time

from .bitrate import effective_sample_floor, measured_kbps
from .constants import (
    BITRATE_BAND, DEFAULT_VMAF_SLOPE, ENDGAME_SNAP_GAIN,
    FLOOR_BOUND_KBPS_RATIO, FLOOR_BOUND_VMAF_MARGIN, INTRA_ONLY_CODECS,
    MIN_BITRATE_KBPS, VMAF_OVERSHOOT, VMAF_SLOPE_MAX, VMAF_SLOPE_MIN,
)
from .probe import res_tier
from .ui import BOLD, DIM, RESET, fmt_s2, fmt_size, label
from .util import atomic_write_json, clamp, partial_hash


def search_ref_index(engine, cfg, source, tag):
    """The search source's FFMS2 index for the SSIMU2 column, named as the
    pipeline names it: the sample's own index when tagged, the source's
    persistent one otherwise. For callers without the file hash at hand
    (the launchers' search wrappers); None when the file can't be read."""
    if tag:
        return engine.sample_ref_index(cfg, source)
    try:
        return engine.full_ref_index(cfg, partial_hash(source))
    except OSError:
        return None


def initial_cq_seed(source_kbps, floor_kbps, min_cq, max_cq, default_cq=30):
    """Starting CQ tuned to source bitrate headroom over the floor.

    High source/floor ratio = lots of compression headroom → start lower.
    Low ratio or unknown data → fall back to default_cq. Seeding slightly
    below the optimal CQ is preferred: it costs marginal bitrate, while
    seeding above costs a full extra encode to step down.
    """
    lo = max(min_cq, min(min_cq + 2, max_cq))
    hi = max(lo, max_cq - 2)
    if not source_kbps or not floor_kbps or source_kbps <= 0 or floor_kbps <= 0:
        return max(min_cq, min(default_cq, max_cq))
    ratio = source_kbps / floor_kbps
    if ratio < 1.5:
        return max(min_cq, min(default_cq, max_cq))
    cq = round(36 - 4 * math.log2(ratio))
    return max(lo, min(cq, hi))


def _hyperbolic_crossing(pts, lf):
    """Floor crossing predicted by log kbps = A + B/(q - C) fitted through
    three same-side (q, log kbps) points.

    With every measured point above the floor, a secant systematically
    undershoots the crossing — the R-Q curve keeps flattening beyond the
    data (rate ~ 1/Q, so d(log R)/dQ shrinks as Q rises). Fitting the
    pole C captures that convexity. Returns the crossing quantizer as a
    float, or None when the points are non-monotone, collinear, or the
    fit falls outside the convex regime it models (pole at/inside the
    data, asymptote at/above the floor).
    """
    (q1, l1), (q2, l2), (q3, l3) = sorted(pts)
    if not (l1 > l2 > l3 > lf):
        return None
    r = (l1 - l2) / (l2 - l3)
    den = (q2 - q1) - r * (q3 - q2)
    if abs(den) < 1e-12:
        return None
    c = ((q2 - q1) * q3 - r * (q3 - q2) * q1) / den
    if c >= q1:
        return None
    b = (l1 - l2) / (1.0 / (q1 - c) - 1.0 / (q2 - c))
    a = l1 - b / (q1 - c)
    if b <= 0 or lf <= a:
        return None
    return c + b / (lf - a)


def search(source, meta, target, cache, cache_path, enc_func, cfg, engine,
           *, tag=None, measure_fn=None, probe_fn=None, s2_fn=None,
           s2_ref_index=None, decay_prior=None, ratio_prior=None):
    """Find the optimal quantizer that hits the target VMAF.

    Adaptive Newton-style search: encode at successive grid points,
    measure VMAF, fit the local quality slope, jump. Also tracks the
    per-resolution bitrate floor — when bitrate is the binding constraint
    instead of VMAF, switches to bitrate-targeting mode on a log-linear
    bitrate model, seeded from the engine's own cold-start decay (or the
    caller's cohort prior) and refined with measured points.

    Injected seams (supplied by the launchers' compat wrappers so their
    module globals stay monkeypatchable):
      measure_fn(ref, dist, q) -> {'mean','p5'}        cached VMAF
      probe_fn(path)           -> probe_video() dict   (duration lookup)
      s2_fn(ref, dist, meta, ref_index) -> dict|None   SSIMU2 info column

    Returns (best, vmaf_result, enc_time, vmaf_time, state). state
    carries the fitted models for the caller's refine loop (the VMAF
    slope and the measured decay are None unless probes measured them)
    and each probe's bitrate and SSIMU2 result, keyed by quantizer. On
    the full path those probes are the full encodes themselves.
    """
    grid = engine.grid
    min_q, max_q = engine.q_bounds(cfg)
    tol = cfg["vmaf_tolerance"]
    slope = DEFAULT_VMAF_SLOPE
    # True once `slope` has been fitted from two real probe pairs. The
    # cold-start guess can size a jump that clamps onto a grid bound
    # purely as a slope artifact, so decisions that treat a bound landing
    # as *proof* (the min_q short-circuit below) must wait for a measured
    # slope. An explicit flag rather than len(tested) >= 2: a NaN-VMAF
    # entry must never count as a measurement.
    slope_measured = False
    enc_time = vmaf_time = 0.0
    tested = {}
    tested_paths = {}
    # Probes whose VMAF measurement was asked for and failed. A skipped
    # measurement (floor-bound probes, the min_q short-circuit) is assumed
    # passing by monotonicity; a failed one proves nothing, so selection
    # never picks it over a probe that measured.
    failed = set()
    # SSIMU2 result per probe that ran it (None: a skip already announced).
    s2_seen = {}

    min_kbps = MIN_BITRATE_KBPS.get(res_tier(meta["w"], meta["h"]), 0)
    # Duration turns each probe's size into its bitrate: the readout on
    # every probe line, and the floor checks when a floor applies. The
    # full path's source IS the file, so its probed duration applies; the
    # sample path probes the search source for its own (short) one.
    # Without a duration neither can run, which must not happen silently.
    why = None
    if tag:
        src_duration = 0.0
        try:
            src_duration = probe_fn(source)["duration"]
        except (RuntimeError, OSError, ValueError) as e:
            why = (str(e).strip().splitlines() or [type(e).__name__])[0]
    else:
        src_duration = meta.get("duration") or 0.0
    if src_duration <= 0:
        print(
            f"{label('bitrate')}{DIM}{'sample' if tag else 'source'}"
            f" duration unknown{f' ({why})' if why else ''}: no bitrate"
            f" readout{' or floor check' if min_kbps else ''} in this"
            f" search{RESET}"
        )

    floor_cap = max_q
    bitrate_points = {}
    enc_tag = engine.signature(cfg, meta.get("crop"))

    # Fallback d(log kbps)/dQ before two probes have measured the local
    # slope: the engine cohort's learned decay when the caller supplies
    # one (calibrate.decay_prior), else this engine's own cold start (how
    # its quantizer scale maps to bitrate is encoder physics, not a
    # shared constant).
    default_decay = engine.default_decay
    if decay_prior and 0 < decay_prior < 1:
        default_decay = decay_prior

    def eff_floor():
        # Sample path converts the video floor into a sample-bitrate threshold.
        # Full path already measures video-only kbps, so compare raw. The
        # ratio is the caller's pick (calibrate.ratio_prior), never read from
        # the cache here: only the caller knows how the sample was drawn,
        # and the file's own ratio counts only for a sample drawn the same way.
        if not tag:
            return min_kbps
        return effective_sample_floor(
            min_kbps, cfg["bitrate_margin"], ratio=ratio_prior,
        )

    # The full path accepts a probe within ENDGAME_SNAP_GAIN under the
    # floor, the same sliver refine and final selection waive: every
    # probe there is a full encode, and one spent lifting the bitrate by
    # less than that buys nothing. Sample probes are cheap predictions
    # and get no waiver. The waiver never moves the floor MODEL's aim.
    waiver = 1.0 if tag else 1.0 - ENDGAME_SNAP_GAIN

    def meets_floor(kbps):
        return kbps >= eff_floor() * waiver

    def in_bitrate_band(q):
        """Full path only: q's video meets the floor and sits inside the
        bitrate band [floor, floor × BITRATE_BAND], which refine accepts
        as-is. A full encode spent trimming it buys only the last few
        percent."""
        kbps = bitrate_points.get(q)
        return bool(
            tag is None and min_kbps and kbps and meets_floor(kbps)
            and kbps <= min_kbps * BITRATE_BAND
        )

    def local_decay(target_kbps):
        """Bitrate decay d(log kbps)/dQ from the two tested points nearest
        target_kbps in log distance. None with <2 points or non-monotone
        data (caller falls back to default_decay).
        """
        if len(bitrate_points) < 2 or target_kbps <= 0:
            return None
        near = sorted(
            bitrate_points,
            key=lambda c: abs(math.log(bitrate_points[c] / target_kbps)),
        )[:2]
        c1, c2 = sorted(near)
        b1, b2 = bitrate_points[c1], bitrate_points[c2]
        if b1 <= 0 or b2 <= 0 or b1 == b2:
            return None
        m = math.log(b1 / b2) / (c2 - c1)
        return m if m > 0 else None

    def crossing(target_kbps):
        """Quantizer, off-grid, where the bitrate model predicts
        target_kbps.

        Brent-style local estimation on the tested (q, log kbps) points.
        The R-Q curve is hyperbolic in quantizer (libaom and SVT-AV1 both
        model rate as R ∝ 1/Q), so it flattens at low q: a chord anchored
        on a distant high-bitrate point overestimates the decay near the
        target and lands probes 1-2 steps too high, each corrected one
        encode at a time via floor_cap.

        With points on both sides of the target, inverse quadratic
        interpolation through the three nearest (trusted only inside the
        bracket, per Brent); otherwise a secant through the two nearest;
        a single point extrapolates with the default decay. When three or
        more points all sit ABOVE the target (no bracket yet), the secant
        undershoots every time for the convexity reason above — there the
        hyperbolic fit through the three nearest points extends the
        estimate, capped at one extra secant-jump so a noisy fit can't
        sail far past the target.
        """
        above = [c for c in bitrate_points if bitrate_points[c] >= target_kbps]
        below = [c for c in bitrate_points if bitrate_points[c] < target_kbps]
        nearest = sorted(
            bitrate_points,
            key=lambda c: abs(math.log(bitrate_points[c] / target_kbps)),
        )
        xt = math.log(target_kbps)

        if above and below and len(nearest) >= 3:
            pts = [(math.log(bitrate_points[c]), c) for c in nearest[:3]]
            (x1, y1), (x2, y2), (x3, y3) = pts
            if x1 != x2 and x1 != x3 and x2 != x3:
                cand = (
                    y1 * (xt - x2) * (xt - x3) / ((x1 - x2) * (x1 - x3))
                    + y2 * (xt - x1) * (xt - x3) / ((x2 - x1) * (x2 - x3))
                    + y3 * (xt - x1) * (xt - x2) / ((x3 - x1) * (x3 - x2))
                )
                if max(above) <= cand < min(below):
                    return cand

        decay = local_decay(target_kbps) or default_decay
        ref_q = nearest[0]
        sec = ref_q + math.log(bitrate_points[ref_q] / target_kbps) / decay
        if not below and len(above) >= 3:
            cand = _hyperbolic_crossing(
                [(c, math.log(bitrate_points[c])) for c in nearest[:3]], xt,
            )
            if cand is not None and cand > sec:
                sec = min(cand, ref_q + 2 * (sec - ref_q))
        return sec

    def estimate_max_q_for_floor():
        """Highest quantizer the floor model lets the search encode at.

        The sample path takes the grid point at or below the crossing of
        the sample threshold itself: its probes are cheap predictions,
        and refine backstops the full encode.

        The full path's probes are full encodes, so they aim at the
        CENTER of the bitrate band [floor, floor × BITRATE_BAND], as a
        miss past the waiver costs another encode: the grid point nearest
        the center crossing, never past the floor crossing. One rule in
        both directions. The grid point at or below the center crossing
        (refine's deficit step) is not it: on an integer grid a step is
        wider than the band, so that point lands above the band about
        half the time and buys a climb when VMAF is over, and in
        simulation it spent more full encodes than the edge aim it
        replaced. The floor crossing bound is the edge aim itself, so
        this never encodes at a higher quantizer than that aim would.

        The result is always clamped into the measured bracket: never
        below a quantizer known to clear the floor, never at/above one
        known to violate it.
        """
        if not min_kbps or not bitrate_points:
            return max_q
        floor = eff_floor()
        above = [c for c in bitrate_points if bitrate_points[c] >= floor]
        below = [c for c in bitrate_points if bitrate_points[c] < floor]

        est = grid.floor(crossing(floor))
        if not tag:
            center = crossing(floor * math.sqrt(BITRATE_BAND))
            est = min(grid.quantize(center), est)

        if below:
            est = min(est, grid.quantize(min(below) - grid.step))
        if above:
            est = max(est, max(above))
        return est

    def fit_slope(q0, vm0, q1, vm1):
        """Refit the VMAF slope from two probes when both measured."""
        nonlocal slope, slope_measured
        if (q0 != q1 and math.isfinite(vm0["mean"])
                and math.isfinite(vm1["mean"])):
            slope = clamp(
                abs(vm0["mean"] - vm1["mean"]) / abs(q0 - q1),
                VMAF_SLOPE_MIN, VMAF_SLOPE_MAX,
            )
            slope_measured = True

    def test(q, measure=True):
        nonlocal enc_time, vmaf_time, floor_cap
        q = grid.quantize(clamp(q, min_q, max_q))

        t0 = time.time()
        dst = enc_func(q)
        enc_time += time.time() - t0

        if measure:
            t0 = time.time()
            vm = measure_fn(source, dst, q)
            vmaf_time += time.time() - t0
            if not math.isfinite(vm["mean"]):
                failed.add(q)
        else:
            vm = {"mean": float("nan"), "p5": float("nan")}

        tested[q] = vm
        tested_paths[q] = dst
        # SSIMU2 info column (display only, present when FFVship is).
        s2 = None
        if measure and math.isfinite(vm["mean"]):
            t0 = time.time()
            s2 = s2_seen[q] = s2_fn(source, dst, meta, s2_ref_index)
            vmaf_time += time.time() - t0
        size = dst.stat().st_size if dst.exists() else 0
        kbps = None
        if src_duration > 1:
            kbps = measured_kbps(dst, src_duration, tag)
        kbps_str = f" {kbps}kbps" if kbps else ""
        vmaf_field = (
            f"  VMAF {BOLD}{vm['mean']:.2f}{RESET}  P5 {BOLD}{vm['p5']:.2f}{RESET}"
            if math.isfinite(vm["mean"])
            else f"  {DIM}VMAF skipped{RESET}"
        )
        print(
            f"{label('search')}{engine.qname} {BOLD}{grid.fmt(q)}{RESET}"
            f"{vmaf_field}{fmt_s2(s2)}"
            f"  {DIM}{fmt_size(size)}{kbps_str}{RESET}"
        )

        if kbps:
            bitrate_points[q] = kbps
            if tag:
                cache["entries"].setdefault(grid.fmt(q), {})[
                    f"{tag}_kbps_{enc_tag}"
                ] = kbps
                atomic_write_json(cache_path, cache)

            # Anything above q would read lower still, so q caps the
            # search even when the full path's waiver accepts q itself.
            ef = eff_floor()
            if kbps < ef:
                floor_cap = min(floor_cap, grid.quantize(q - grid.step))
                if not meets_floor(kbps):
                    print(
                        f"{label('bitrate')}{kbps}kbps"
                        f" {'sample' if tag else 'video'} at"
                        f" {engine.qname} {grid.fmt(q)} below {min_kbps}kbps"
                        f" floor (threshold {int(ef)}kbps), capping at"
                        f" {engine.qname} {BOLD}{grid.fmt(floor_cap)}{RESET}"
                    )

        return q, vm

    # Seed the first quantizer from source-bitrate headroom over the
    # floor. Falls back to 30 when the source bitrate or floor is unknown
    # — or when the source is an intra-only mezzanine codec
    # (ProRes/DNxHD): those bitrates say nothing about AV1
    # compressibility and would seed several steps too low, wasting a
    # probe near-lossless. meta["bitrate"] is the container's rate, audio
    # included, so the ratio reads high by the audio's share.
    src_kbps_hint = None
    if meta.get("bitrate") and meta.get("codec") not in INTRA_ONLY_CODECS:
        src_kbps_hint = int(meta["bitrate"] / 1000)
    user_seed = engine.seed_override(cfg)
    if user_seed is not None:
        seed_q = grid.quantize(clamp(user_seed, min_q, max_q))
        print(
            f"{label('seed')}{engine.qname}"
            f" {BOLD}{grid.fmt(seed_q)}{RESET} {DIM}(user){RESET}"
        )
    else:
        seed_q = grid.quantize(initial_cq_seed(
            src_kbps_hint, min_kbps, min_q, max_q
        ))
        if seed_q != 30:
            print(
                f"{label('seed')}{engine.qname}"
                f" {BOLD}{grid.fmt(seed_q)}{RESET}"
                f" {DIM}(source {src_kbps_hint or '?'}kbps vs floor {min_kbps or '-'}kbps){RESET}"
            )

    q, vm = test(seed_q)
    if not math.isfinite(vm["mean"]):
        return None, None, enc_time, vmaf_time, None

    # Both paths aim at the center of the acceptance band [target - tol,
    # target + VMAF_OVERSHOOT], so slope error spreads symmetrically
    # inside it. They stop differently. The full path ships the first
    # probe refine would accept as-is, because every probe there is a
    # full encode. The sample path's landing is a PREDICTION INPUT: the
    # final encode lands wherever the sample landed plus the sample→full
    # offset error, so any slack left here goes straight into the
    # verify/refine miss budget, where correcting it costs a full
    # re-encode. Sample probes are cheap, so that path converges on the
    # center itself.
    aim = target + (VMAF_OVERSHOOT - tol) / 2

    # Floor-bound detection: VMAF comfortably above target AND bitrate well
    # below floor. In this regime VMAF is not binding — it's a pure bitrate
    # targeting problem. Skip VMAF on intermediate probes; verify once on
    # the final candidate.
    floor_bound = bool(
        min_kbps and vm["mean"] > target + FLOOR_BOUND_VMAF_MARGIN
        and q in bitrate_points
        and bitrate_points[q] < eff_floor() * FLOOR_BOUND_KBPS_RATIO
    )
    if floor_bound:
        print(
            f"{label('mode')}floor-bound "
            f"{DIM}(skipping VMAF on intermediate probes){RESET}"
        )

    # Proactive bitrate jump: a seed that passes VMAF but misses the floor
    # goes straight to the extrapolated floor quantizer. Missing the floor
    # puts the model's crossing below the seed, so the jump always moves
    # down; at min_q there is nowhere to go.
    seed_kbps = bitrate_points.get(q)
    if (vm["mean"] >= target - tol and seed_kbps
            and not meets_floor(seed_kbps)):
        floor_q = grid.quantize(
            clamp(estimate_max_q_for_floor(), min_q, q - grid.step)
        )
        if floor_q < q:
            prev_q, prev_vm = q, vm
            q, vm = test(floor_q, measure=not floor_bound)
            fit_slope(prev_q, prev_vm, q, vm)

    if floor_bound:
        # Bitrate-only convergence: keep picking the estimated floor
        # quantizer until we bracket it; accept when at the ceiling with
        # floor met, or on the full path inside the bitrate band.
        for _ in range(4):
            effective_max = min(max_q, floor_cap, estimate_max_q_for_floor())
            current_kbps = bitrate_points.get(q, 0)
            # meets_floor compares against the sample-converted threshold,
            # not the raw video floor: with a learned ratio > 1 the raw
            # floor sits ABOVE the threshold, and gating on it walked past
            # points the model already accepted.
            if ((q >= effective_max or in_bitrate_band(q)) and current_kbps
                    and meets_floor(current_kbps)):
                print(
                    f"{label('accept')}bitrate floor met at"
                    f" {engine.qname} {BOLD}{grid.fmt(q)}{RESET}"
                )
                break

            next_q = grid.quantize(max(min_q, effective_max))
            if (next_q == q and not meets_floor(current_kbps)
                    and q - grid.step >= min_q):
                next_q = grid.quantize(q - grid.step)
            if next_q == q or next_q in tested:
                break

            q, vm = test(next_q, measure=False)
            if q not in bitrate_points:
                break
    else:
        # The sample path gets one extra iteration: converging on the aim
        # instead of stopping anywhere in the band occasionally takes one
        # more cheap probe.
        for _ in range(5 if tag else 4):
            # A failed measurement ends the probing (selection below never
            # picks it); every other probe in this loop is measured.
            if not math.isfinite(vm["mean"]):
                break
            # A probe under the floor is never the answer (selection
            # below discards it), so landing on the VMAF aim there is not
            # convergence: the floor model steps down to the cap instead.
            q_violates_floor = (
                q in bitrate_points and not meets_floor(bitrate_points[q])
            )
            in_band = target - tol <= vm["mean"] <= target + VMAF_OVERSHOOT
            # The full path stops where refine would accept: tol past the
            # band top is the same noise hysteresis, since re-encoding a
            # whole file over a sub-noise excess is spurious precision.
            if (tag is None and not q_violates_floor
                    and target - tol <= vm["mean"]
                    <= target + VMAF_OVERSHOOT + tol):
                break
            # Sample-path convergence: within tol of the aim is done (tol
            # doubles as the epsilon — VMAF differences under it are
            # noise). Grid resolution bounds it below: when the remaining
            # delta rounds to no step, the `next_q == q` break fires.
            if (tag is not None and abs(vm["mean"] - aim) <= tol
                    and not q_violates_floor):
                break
            delta = (vm["mean"] - aim) / slope
            bitrate_bound = min(floor_cap, estimate_max_q_for_floor())
            effective_max = min(max_q, bitrate_bound)

            # At the quantizer ceiling (can't go higher without violating
            # the floor), accept any overshoot as long as VMAF meets target
            # and this point's bitrate isn't already below the predicted
            # floor. On the full path a video inside the bitrate band is
            # at that ceiling too.
            if (vm["mean"] >= target - tol and not q_violates_floor
                    and (q >= effective_max or in_bitrate_band(q))):
                # Say what held it: the floor model, the user's own grid
                # bound, or the bitrate band. Naming bitrate for a file
                # that simply ran out of grid reads as "nothing more to
                # gain here" when the actual answer is "raise the max".
                # Strict <: with no floor, or no bitrate measured yet,
                # the estimator returns max_q itself, and that tie is not
                # evidence of a bitrate limit.
                if q < effective_max:
                    held_by = (
                        f"is within {BITRATE_BAND - 1:.0%} of the"
                        f" {min_kbps}kbps floor"
                    )
                elif min_kbps and bitrate_bound < max_q:
                    held_by = "is at bitrate ceiling"
                else:
                    held_by = f"is the highest {engine.qname} allowed"
                print(
                    f"{label('accept')}VMAF passes and"
                    f" {engine.qname} {BOLD}{grid.fmt(q)}{RESET}"
                    f" {held_by}"
                )
                break

            next_q = grid.quantize(clamp(q + delta, min_q, effective_max))
            if next_q == q and not in_band:
                # Outside the band with a sub-step delta: force one step
                # toward it rather than stalling short. Inside the band a
                # sub-step delta means grid-converged — fall through to
                # the `next_q == q` break below.
                next_q = grid.quantize(
                    q + (grid.step if vm["mean"] > aim else -grid.step)
                )
            next_q = grid.quantize(clamp(next_q, min_q, effective_max))

            # Quality-ceiling short-circuit (sample path only). A downward
            # jump that clamps to min_q — the max-quality grid bound — while
            # VMAF is still in deficit means min_q is forced: nothing encodes
            # at higher quality and we're below target, so the final
            # selection takes min_q whatever its VMAF reads. On the sample
            # path that probe only measures a number that changes no
            # decision, so skip the encode entirely and let the final
            # full-file encode verify. The full path's min_q encode is the
            # deliverable, so it has nothing to skip — this is the low-q
            # mirror of the high-q bitrate-ceiling accept above.
            #
            # Gated on a MEASURED slope: "min_q is forced" is only proven
            # when the deficit extrapolates from real slope data. Off the
            # cold-start guess a steep curve makes the first jump's raw
            # Newton target overshoot the grid and park on min_q when an
            # interior quantizer would pass — firing there selects max
            # quality unprobed and the full encode pays a multi-re-encode
            # overshoot walk-back. With the guard the loop probes min_q
            # normally (a cheap sample encode); if it overshoots, the
            # loop's own stepping walks back to the interior.
            if (tag is not None and slope_measured and next_q == min_q
                    and next_q not in tested and vm["mean"] < target - tol):
                print(
                    f"{label('accept')}{engine.qname}"
                    f" {BOLD}{grid.fmt(min_q)}{RESET} is max quality and VMAF"
                    f" still short — selecting it without a sample probe"
                )
                tested[next_q] = {"mean": float("nan"), "p5": float("nan")}
                q, vm = next_q, tested[next_q]
                break

            # Full-file endgame snap. Sample probes are cheap, but on the
            # full-file path every probe is a full encode and the current
            # one ships as-is when accepted: a final climb toward the
            # bitrate ceiling that is predicted to trim under
            # ENDGAME_SNAP_GAIN of bitrate costs more than it buys.
            if (tag is None and min_kbps and next_q > q
                    and vm["mean"] >= target - tol and not q_violates_floor
                    and q in bitrate_points):
                snap_decay = local_decay(eff_floor()) or default_decay
                if (next_q - q) * snap_decay < -math.log1p(-ENDGAME_SNAP_GAIN):
                    print(
                        f"{label('accept')}{engine.qname}"
                        f" {grid.fmt(next_q)} would trim under"
                        f" {ENDGAME_SNAP_GAIN:.0%} bitrate — keeping"
                        f" {engine.qname} {BOLD}{grid.fmt(q)}{RESET}"
                    )
                    break

            if next_q == q or next_q in tested:
                break

            prev_q, prev_vm = q, vm
            q, vm = test(next_q)
            fit_slope(prev_q, prev_vm, q, vm)

    def valid_q(c):
        if c in failed:
            return False
        vm_c = tested[c]
        if math.isfinite(vm_c["mean"]) and vm_c["mean"] < target - tol:
            return False
        kbps = bitrate_points.get(c)
        return not kbps or meets_floor(kbps)

    valid = [c for c in tested if valid_q(c)]
    if tag is None or floor_bound:
        # Full path / floor-bound: highest valid quantizer = smallest
        # file that still passes (floor-bound entries carry NaN VMAF, so
        # aim distance is meaningless there).
        best = max(valid, default=None)
    else:
        # The sample path stepped toward the aim, so several in-band
        # candidates can exist; pick the landing closest to it — a
        # band-bottom pick would hand its slack straight to the
        # full-encode miss budget. VMAF-less entries (the min_q
        # short-circuit) rank last; ties prefer the higher quantizer
        # (smaller file).
        def aim_dist(c):
            m = tested[c].get("mean", float("nan"))
            return abs(m - aim) if math.isfinite(m) else float("inf")
        best = min(valid, key=lambda c: (aim_dist(c), -c), default=None)
    if best is None:
        # No quantizer satisfies both VMAF target and bitrate floor in the
        # tested range. Prefer the lowest one tested — highest bitrate
        # (best shot at the floor) and highest VMAF by monotonicity.
        # Returning max-VMAF here would in floor-bound mode mean the seed
        # that violated the floor — forcing the caller to re-encode the
        # full source to push the quantizer lower.
        best = min(tested) if tested else None

    # Guarantee a VMAF measurement on the returned candidate (floor-bound
    # path may have skipped it). Monotonicity makes a failure here very
    # unlikely — floor-bound only triggers when the seed already cleared
    # target by FLOOR_BOUND_VMAF_MARGIN, and every subsequent probe is at
    # a lower quantizer (higher VMAF).
    if (best is not None and not math.isfinite(tested[best]["mean"])
            and best in tested_paths and tested_paths[best].exists()):
        t0 = time.time()
        tested[best] = measure_fn(source, tested_paths[best], best)
        vmaf_time += time.time() - t0

    # Slopes for caller's post-search refinement. Decay is fitted from the
    # points nearest the floor — the regime where the refine loop uses it.
    # Neither measured value falls back to its cold-start guess, so the
    # caller never rolls a guess into the calibration.
    bitrate_decay = default_decay
    measured_decay = None
    if min_kbps:
        m = local_decay(eff_floor())
        if m:
            bitrate_decay = measured_decay = m
    state = {
        "vmaf_slope": slope if slope_measured else None,
        "bitrate_decay": bitrate_decay,
        "measured_decay": measured_decay,
        "kbps": dict(bitrate_points),
        "ssimu2": s2_seen,
    }

    return best, tested.get(best), enc_time, vmaf_time, state
