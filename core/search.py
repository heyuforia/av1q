"""The adaptive quantizer search — the shared brain of both pipelines.

One implementation serves av1q (integer CQ grid) and av1q-essential
(quarter-step CRF grid): all stepping goes through the engine's Grid and
all measurement through injected closures, so the search never knows
which encoder or cache layout sits behind it. av1q's integer-grid
behavior is the contract of record; the generalization must not change
it (quantize/clamp ordering is equivalent because bounds are on-grid —
see Grid in core/engines/base.py).

Seeding lives here too: initial_cq_seed maps source-bitrate headroom
over the floor to a starting quantizer.
"""

import math
import time

from .bitrate import effective_sample_floor, measured_kbps
from .constants import (
    DEFAULT_BITRATE_DECAY, ENDGAME_SNAP_GAIN, INTRA_ONLY_CODECS,
    MIN_BITRATE_KBPS, VMAF_OVERSHOOT,
)
from .probe import res_tier
from .ui import BOLD, DIM, ORANGE, RESET, fmt_s2
from .util import atomic_write_json, clamp


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
           s2_ref_index=None, decay_prior=None):
    """Find the optimal quantizer that hits the target VMAF.

    Adaptive Newton-style search: encode at successive grid points,
    measure VMAF, fit the local quality slope, jump. Also tracks the
    per-resolution bitrate floor — when bitrate is the binding constraint
    instead of VMAF, switches to bitrate-targeting mode on a log-linear
    bitrate model (±6 CRF ≈ 2× bitrate, refined with measured points).

    Injected seams (supplied by the launchers' compat wrappers so their
    module globals stay monkeypatchable):
      measure_fn(ref, dist, q) -> {'mean','p5'}        cached VMAF
      probe_fn(path)           -> probe_video() dict   (duration lookup)
      s2_fn(ref, dist, meta, ref_index) -> dict|None   SSIMU2 info column

    Returns (best, vmaf_result, enc_time, vmaf_time, state) where state
    carries the fitted slopes for the caller's refine loop.
    """
    grid = engine.grid
    min_q, max_q = engine.q_bounds(cfg)
    tol = cfg["vmaf_tolerance"]
    slope = 0.5
    enc_time = vmaf_time = 0.0
    tested = {}
    tested_paths = {}

    min_kbps = MIN_BITRATE_KBPS.get(res_tier(meta["w"], meta["h"]), 0)
    # Duration drives the per-probe bitrate readout, which is shown for
    # every probe now — not just files with a non-zero floor (SD is tier 0
    # with min_kbps == 0 but its encodes still have a bitrate worth seeing).
    # The full path's source IS the file, so its meta duration applies and
    # no extra probe is needed; the sample path must probe the concat for
    # its own (short) duration. The floor machinery below stays gated on
    # min_kbps regardless.
    if tag:
        src_duration = 0.0
        try:
            src_duration = probe_fn(source)["duration"]
        except Exception:
            pass
    else:
        src_duration = meta.get("duration") or 0.0

    floor_cap = max_q
    bitrate_points = {}
    enc_tag = engine.signature(cfg, meta.get("crop"))

    # Fallback d(log kbps)/dQ before two probes have measured the local
    # slope: the engine cohort's learned decay when the caller supplies
    # one (calibrate.decay_prior), else the generic ±6 ≈ 2× default.
    default_decay = DEFAULT_BITRATE_DECAY
    if decay_prior and 0 < decay_prior < 1:
        default_decay = decay_prior

    def eff_floor():
        # Sample path converts the video floor into a sample-bitrate threshold.
        # Full path already measures video-only kbps, so compare raw.
        if not tag:
            return min_kbps
        return effective_sample_floor(
            min_kbps, cfg["bitrate_margin"], cache.get("calibration")
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

    def estimate_max_q_for_floor():
        """Highest quantizer whose bitrate predicts the full video at/above
        floor.

        Brent-style local estimation on the tested (q, log kbps) points.
        The R-Q curve is hyperbolic in quantizer (libaom and SVT-AV1 both
        model rate as R ∝ 1/Q), so it flattens at low q: a chord anchored
        on a distant high-bitrate point overestimates the decay near the
        floor, landing probes 1-2 steps too high — the old first/last-point
        fit then corrected one full encode at a time via floor_cap.

        With points on both sides of the floor, inverse quadratic
        interpolation through the three nearest (trusted only inside the
        bracket, per Brent); otherwise a secant through the two nearest;
        a single point extrapolates with the default decay. When three or
        more points all sit ABOVE the floor (no bracket yet), the secant
        undershoots every time for the convexity reason above — there the
        hyperbolic fit through the three nearest points extends the
        estimate, capped at one extra secant-jump so a noisy fit can't
        sail far past the floor. The result is always clamped into the
        measured bracket: never below a quantizer known to clear the
        floor, never at/above one known to violate it.
        """
        if not min_kbps or not bitrate_points:
            return max_q
        floor = eff_floor()
        above = [c for c in bitrate_points if bitrate_points[c] >= floor]
        below = [c for c in bitrate_points if bitrate_points[c] < floor]
        nearest = sorted(
            bitrate_points,
            key=lambda c: abs(math.log(bitrate_points[c] / floor)),
        )

        est = None
        if above and below and len(nearest) >= 3:
            pts = [(math.log(bitrate_points[c]), c) for c in nearest[:3]]
            (x1, y1), (x2, y2), (x3, y3) = pts
            xt = math.log(floor)
            if x1 != x2 and x1 != x3 and x2 != x3:
                cand = (
                    y1 * (xt - x2) * (xt - x3) / ((x1 - x2) * (x1 - x3))
                    + y2 * (xt - x1) * (xt - x3) / ((x2 - x1) * (x2 - x3))
                    + y3 * (xt - x1) * (xt - x2) / ((x3 - x1) * (x3 - x2))
                )
                if max(above) <= cand < min(below):
                    est = grid.floor(cand)

        if est is None:
            decay = local_decay(floor) or default_decay
            ref_q = nearest[0]
            sec = ref_q + math.log(bitrate_points[ref_q] / floor) / decay
            if not below and len(above) >= 3:
                cand = _hyperbolic_crossing(
                    [(c, math.log(bitrate_points[c])) for c in nearest[:3]],
                    math.log(floor),
                )
                if cand is not None and cand > sec:
                    sec = min(cand, ref_q + 2 * (sec - ref_q))
            est = grid.floor(sec)

        if below:
            est = min(est, grid.quantize(min(below) - grid.step))
        if above:
            est = max(est, max(above))
        return est

    def test(q, measure=True):
        nonlocal enc_time, vmaf_time, floor_cap
        q = grid.quantize(clamp(q, min_q, max_q))
        if q in tested:
            # Upgrade a previously-skipped VMAF measurement if now needed
            if measure and not math.isfinite(
                tested[q].get("mean", float("nan"))
            ) and q in tested_paths and tested_paths[q].exists():
                t0 = time.time()
                tested[q] = measure_fn(source, tested_paths[q], q)
                vmaf_time += time.time() - t0
            return q, tested[q]

        t0 = time.time()
        dst = enc_func(q)
        enc_time += time.time() - t0

        if measure:
            t0 = time.time()
            vm = measure_fn(source, dst, q)
            vmaf_time += time.time() - t0
        else:
            vm = {"mean": float("nan"), "p5": float("nan")}

        tested[q] = vm
        tested_paths[q] = dst
        # SSIMU2 info column (display only, present when FFVship is).
        s2 = None
        if measure and math.isfinite(vm["mean"]):
            t0 = time.time()
            s2 = s2_fn(source, dst, meta, s2_ref_index)
            vmaf_time += time.time() - t0
        sz_mb = dst.stat().st_size / 1e6 if dst.exists() else 0
        if src_duration > 1:
            kbps = measured_kbps(dst, src_duration, tag)
        else:
            kbps = None
        kbps_str = f" {kbps}kbps" if kbps else ""
        vmaf_field = (
            f"  VMAF {BOLD}{vm['mean']:.2f}{RESET}  P5 {BOLD}{vm['p5']:.2f}{RESET}"
            if math.isfinite(vm["mean"])
            else f"  {DIM}VMAF skipped{RESET}"
        )
        print(
            f" {ORANGE}{'search':<10}{RESET}{engine.qname} {BOLD}{grid.fmt(q)}{RESET}"
            f"{vmaf_field}{fmt_s2(s2)}"
            f"  {DIM}{sz_mb:.1f}MB{kbps_str}{RESET}"
        )

        if kbps:
            bitrate_points[q] = kbps
            if tag:
                cache["entries"].setdefault(grid.fmt(q), {})[
                    f"{tag}_kbps_{enc_tag}"
                ] = kbps
                atomic_write_json(cache_path, cache)

        if min_kbps and src_duration > 1 and kbps:
            ef = eff_floor()
            if kbps <= ef:
                floor_cap = min(floor_cap, grid.quantize(q - grid.step))
                label = "sample" if tag else "video"
                print(
                    f" {ORANGE}{'bitrate':<10}{RESET}{kbps}kbps {label} at"
                    f" {engine.qname} {grid.fmt(q)} below {min_kbps}kbps floor"
                    f" (threshold {int(ef)}kbps), capping at"
                    f" {engine.qname} {BOLD}{grid.fmt(floor_cap)}{RESET}"
                )

        return q, vm

    # Seed the first quantizer from source bitrate instead of a hardcoded
    # 30. High source/floor ratio has more compression headroom, so we
    # start closer to the answer. Falls back to 30 when source bitrate or
    # floor is unknown — or when the source is an intra-only mezzanine
    # codec (ProRes/DNxHD): those bitrates say nothing about AV1
    # compressibility and would seed several steps too low, wasting a
    # probe near-lossless.
    src_kbps_hint = None
    if meta.get("bitrate") and meta.get("codec") not in INTRA_ONLY_CODECS:
        src_kbps_hint = int(meta["bitrate"] / 1000)
    user_seed = engine.seed_override(cfg)
    if user_seed is not None:
        seed_q = grid.quantize(clamp(user_seed, min_q, max_q))
        print(
            f" {ORANGE}{'seed':<10}{RESET}{engine.qname}"
            f" {BOLD}{grid.fmt(seed_q)}{RESET} {DIM}(user){RESET}"
        )
    else:
        seed_q = grid.quantize(initial_cq_seed(
            src_kbps_hint, min_kbps, min_q, max_q
        ))
        if seed_q != 30:
            print(
                f" {ORANGE}{'seed':<10}{RESET}{engine.qname}"
                f" {BOLD}{grid.fmt(seed_q)}{RESET}"
                f" {DIM}(source {src_kbps_hint or '?'}kbps vs floor {min_kbps or '-'}kbps){RESET}"
            )

    q, vm = test(seed_q)
    if not math.isfinite(vm["mean"]):
        return None, None, enc_time, vmaf_time, None

    # Floor-bound detection: VMAF comfortably above target AND bitrate well
    # below floor. In this regime VMAF is not binding — it's a pure bitrate
    # targeting problem. Skip VMAF on intermediate probes; verify once on
    # the final candidate. Monotonicity (lower q → higher VMAF) keeps this
    # safe as long as the seed test already cleared target.
    floor_bound = bool(
        min_kbps and vm["mean"] > target + 2.0
        and q in bitrate_points
        and bitrate_points[q] < eff_floor() * 0.80
    )
    if floor_bound:
        print(
            f" {ORANGE}{'mode':<10}{RESET}floor-bound "
            f"{DIM}(skipping VMAF on intermediate probes){RESET}"
        )

    # Proactive bitrate jump: go straight to the extrapolated floor
    # quantizer. Only when the model puts the crossing below the current
    # one — when est >= q the current probe already sits at the predicted
    # ceiling and the main loop accepts it without spending another encode.
    if (min_kbps and (floor_bound or vm["mean"] >= target - tol)
            and q in bitrate_points
            and bitrate_points[q] < eff_floor() * 1.10
            and q - grid.step >= min_q):
        est_q = estimate_max_q_for_floor()
        floor_q = grid.quantize(clamp(est_q, min_q, q - grid.step))
        if est_q < q and floor_q not in tested:
            prev_q, prev_vm = q, vm
            q, vm = test(floor_q, measure=not floor_bound)
            if (math.isfinite(vm["mean"]) and math.isfinite(prev_vm["mean"])
                    and prev_q != q):
                slope = clamp(
                    abs(prev_vm["mean"] - vm["mean"]) / abs(prev_q - q),
                    0.1, 1.5,
                )

    if floor_bound:
        # Bitrate-only convergence: keep picking the estimated floor
        # quantizer until we bracket it; accept when at the ceiling with
        # floor met.
        for _ in range(4):
            effective_max = min(max_q, floor_cap, estimate_max_q_for_floor())
            current_kbps = bitrate_points.get(q, 0)
            q_violates_floor = (
                q in bitrate_points and bitrate_points[q] < eff_floor()
            )
            if (q >= effective_max and current_kbps >= min_kbps
                    and not q_violates_floor):
                print(
                    f" {ORANGE}{'accept':<10}{RESET}bitrate floor met at"
                    f" {engine.qname} {BOLD}{grid.fmt(q)}{RESET}"
                )
                break

            next_q = grid.quantize(
                clamp(estimate_max_q_for_floor(), min_q, effective_max)
            )
            if (next_q == q and current_kbps < min_kbps
                    and q - grid.step >= min_q):
                next_q = grid.quantize(q - grid.step)
            if next_q == q or next_q in tested:
                break

            q, vm = test(next_q, measure=False)
            if q not in bitrate_points:
                break
    else:
        for _ in range(4):
            if target - tol <= vm["mean"] <= target + VMAF_OVERSHOOT:
                break
            delta = (vm["mean"] - target) / slope
            effective_max = min(max_q, floor_cap, estimate_max_q_for_floor())

            # At the quantizer ceiling (can't go higher without violating
            # the floor), accept any overshoot as long as VMAF meets target
            # and this point's bitrate isn't already below the predicted
            # floor.
            q_violates_floor = (
                min_kbps and q in bitrate_points
                and bitrate_points[q] < eff_floor()
            )
            if (vm["mean"] >= target - tol
                    and q >= effective_max and not q_violates_floor):
                print(
                    f" {ORANGE}{'accept':<10}{RESET}VMAF passes and"
                    f" {engine.qname} {BOLD}{grid.fmt(q)}{RESET}"
                    f" is at bitrate ceiling"
                )
                break

            next_q = grid.quantize(clamp(q + delta, min_q, effective_max))
            if next_q == q:
                next_q = grid.quantize(
                    q + (grid.step if vm["mean"] > target else -grid.step)
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
            if (tag is not None and next_q == min_q and next_q not in tested
                    and vm["mean"] < target - tol):
                print(
                    f" {ORANGE}{'accept':<10}{RESET}{engine.qname}"
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
                        f" {ORANGE}{'accept':<10}{RESET}{engine.qname}"
                        f" {grid.fmt(next_q)} would trim under"
                        f" {ENDGAME_SNAP_GAIN:.0%} bitrate — keeping"
                        f" {engine.qname} {BOLD}{grid.fmt(q)}{RESET}"
                    )
                    break

            if next_q == q or next_q in tested:
                break

            prev_q, prev_vm = q, vm
            q, vm = test(next_q)
            if not math.isfinite(vm["mean"]):
                break
            if prev_q != q:
                slope = clamp(
                    abs(prev_vm["mean"] - vm["mean"]) / abs(prev_q - q),
                    0.1, 1.5,
                )

    def valid_q(c):
        vm_c = tested[c]
        if math.isfinite(vm_c["mean"]) and vm_c["mean"] < target - tol:
            return False
        if min_kbps and src_duration > 1 and c in tested_paths:
            kbps = measured_kbps(tested_paths[c], src_duration, tag)
            if kbps and kbps < eff_floor():
                return False
        return True

    best = max((c for c in tested if valid_q(c)), default=None)
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
    # target by 2, and every subsequent probe is at a lower quantizer
    # (higher VMAF).
    if (best is not None and not math.isfinite(tested[best]["mean"])
            and best in tested_paths and tested_paths[best].exists()):
        t0 = time.time()
        tested[best] = measure_fn(source, tested_paths[best], best)
        vmaf_time += time.time() - t0

    # Slopes for caller's post-search refinement. Decay is fitted from the
    # points nearest the floor — the regime where the refine loop uses it.
    # measured_decay is None unless probes actually measured it, so the
    # caller never rolls the cold-start default into the calibration.
    bitrate_decay = default_decay
    measured_decay = None
    if min_kbps:
        m = local_decay(eff_floor())
        if m:
            bitrate_decay = measured_decay = m
    state = {
        "vmaf_slope": slope,
        "bitrate_decay": bitrate_decay,
        "measured_decay": measured_decay,
    }

    return best, tested.get(best), enc_time, vmaf_time, state
