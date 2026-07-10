# AV1Q

Automatically encode videos to AV1 at the best quality-to-size ratio, using VMAF to find the optimal settings for each file.

![AV1Q-Cli-Screenshot](https://raw.githubusercontent.com/heyuforia/av1q/refs/heads/main/av1q-cli.png)

## What it does

Drop videos into a folder, run the script, and each file gets encoded to AV1 at a quality level that hits a target VMAF score. Instead of guessing CRF values or doing manual test encodes, the tool figures it out for you.

1. **Detects scenes** and extracts short representative samples from complex parts of the video
2. **Searches for the optimal CQ** by encoding only the samples — not the full file — and measuring VMAF after each attempt
3. **Encodes the full video** at the chosen CQ, then verifies the final VMAF score matches the target
4. **Refines in 1-2 passes** if VMAF or the bitrate floor comes up short — using the measured quality/bitrate slopes to jump directly to the right CQ rather than stepping by one

## Features

- **VMAF-targeted encoding** — hits a perceptual quality target instead of using a fixed CRF
- **Resolution-aware defaults** — auto-selects VMAF targets (94 for HD, 93 for SD, 90 for 4K)
- **Bitrate floors** — per-resolution minimum bitrates (1 Mbps 720p, 1.8 Mbps 1080p, 2.5 Mbps 1440p, 4.5 Mbps 2160p, 8 Mbps 4320p) prevent VMAF-misleading low-bitrate encodes; starvation backstops, not targets
- **Scene-based sampling** — fast quality estimation without encoding the whole file during search
- **P5 quality reporting** — measures and reports 5th-percentile worst-frame VMAF alongside the mean, so you can spot files whose worst moments lag
- **SSIMULACRA2 reporting** — every VMAF score is shown next to a GPU-computed [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2) score as a second opinion; informational only, it never influences the encode
- **HDR & color preservation** — carries over color primaries, transfer, matrix, and range
- **10-bit output** by default, film grain synthesis included
- **Hardware-accelerated decoding** — CUDA, D3D11VA (Windows), VideoToolbox (macOS), VAAPI (Linux) speed up quality measurement and scene/crop detection; encoding itself is always CPU
- **Optional auto-crop** — `--auto-crop` detects letterbox/pillarbox bars inline before each encode (or use the standalone `av1q-crop.py` to pre-scan a library); confidence-gated so ambiguous detections aren't silently applied
- **File-based caching** — skips re-analysis and re-measurement on subsequent runs, and interrupted searches resume where they left off
- **Resumable encodes** — full encodes of long sources (15 min+) are written as finalized segments, so an interrupted encode picks up at the last finished segment instead of restarting from frame 0 (disable with `--no-resume`)
- **Batch processing** with recursive subdirectory support
- **Cross-platform** — Windows, macOS, Linux

## Requirements

- **Python 3.8+**
- **ffmpeg** and **ffprobe** in PATH, built with:
  - `libsvtav1` (SVT-AV1 encoder)
  - `libvmaf` (VMAF quality metrics)

Most ffmpeg builds from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (Windows) or [BtbN](https://github.com/BtbN/FFmpeg-Builds/releases) (Linux/Windows) include both. On macOS: `brew install ffmpeg`.

### Optional

- [colorama](https://pypi.org/project/colorama/) for Windows terminal colors (`pip install colorama`). Works without it.
- [FFVship](https://codeberg.org/Line-fr/Vship) powers the SSIMULACRA2 info scores. On Windows it's downloaded automatically on first run (the build matching your GPU — NVIDIA, AMD, or generic Vulkan). Without it the SSIMULACRA2 column simply disappears; everything else works the same.

## Usage

**Basic** — processes all videos in `./Video Input`, outputs to `./AV1 Output`:

```
python av1q.py
```

**Custom directories:**

```
python av1q.py -i /path/to/videos -o /path/to/output
```

**Override quality target and encoding speed:**

```
python av1q.py --vmaf 95 --preset 6
```

**Start the search at a known CQ** — when run in a terminal, the script asks once at startup:

```
Seed CQ 18–38 (Enter = auto):
```

Press Enter for the automatic seed, or type a CQ to start every file's search there — handy when a batch of similar clips all land around the same value, since a good seed can collapse the search to a single encode. The seed is only a starting point: VMAF is still measured and the search still corrects a wrong guess. `--seed-cq 24` does the same non-interactively (and skips the prompt); piped/scripted runs without the flag skip the prompt entirely.

If some files in the batch were already encoded by a previous run, a seed alone won't redo them — their finished search result is reused. When a seed is given interactively, av1q lists those files and asks once whether to re-encode them with a fresh search from the seed (clearing their cached results) or keep the previous answers.

**Auto-crop letterboxed or pillarboxed videos** — add `--auto-crop`:

```
python av1q.py --auto-crop
```

Before each encode, the script samples 8 short windows of the video, detects black bars, and applies the crop if the detection is confident. Results are cached as `<file>.crop.json` sidecars beside each source, so re-runs reuse them without re-scanning. Low-confidence detections (dark sources, mixed aspect ratios) are saved for manual review but not auto-applied. Pass `--no-crops` to ignore sidecars entirely.

For batch pre-scanning a whole library before encoding (so you can review borderline sidecars first), the companion `av1q-crop.py` does the same detection standalone and writes the same sidecar format:

```
python av1q-crop.py
python av1q.py
```

### Options

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | `./Video Input` | Input directory |
| `-o`, `--output` | `./AV1 Output` | Output directory |
| `--vmaf` | Auto by resolution | Target VMAF score |
| `--preset` | `4` | SVT-AV1 preset (0-10, lower = slower + better) |
| `--min-cq` | `18` | Minimum CQ (highest quality bound) |
| `--max-cq` | `38` | Maximum CQ (lowest quality bound) |
| `--film-grain` | `24` | Film grain synthesis level (0-50) |
| `--samples` | `8` | Number of sample segments for estimation |
| `--seed-cq` | Prompted / auto | Starting CQ for the search (skips the interactive prompt) |
| `--no-10bit` | — | Disable forced 10-bit encoding |
| `--no-recurse` | — | Don't process subdirectories |
| `--overwrite` | — | Re-encode even if output exists |
| `--dry-run` | — | Find optimal CQ but skip final encoding |
| `--auto-crop` | — | Detect letterbox/pillarbox inline before each encode (skips files that already have a sidecar) |
| `--no-crops` | — | Ignore `*.crop.json` sidecars (auto-applied by default) |
| `--no-resume` | — | Disable resumable segmented encoding for long sources |

## How it works

The script uses an adaptive search (similar to Newton's method) to converge on the right CQ value in 2-4 iterations rather than brute-forcing every option:

1. **Analyze** — scene detection identifies visually distinct segments; frame complexity analysis ranks them
2. **Sample** — the most complex scenes are extracted as short clips and concatenated. Short files (~15–60s at defaults) scale down to a mini plan (3×2s clips) instead of skipping sampling; only files at ~15s or under search on the full file directly — there even tiny probes would cover most of the file anyway
3. **Search** — the sample is encoded at a seed CQ derived from the source's bitrate headroom over the floor (higher headroom → lower starting CQ; falls back to 30 when unknown, and can be overridden interactively or with `--seed-cq`), VMAF is measured, and the next CQ is estimated from the slope of quality-vs-CQ. The search estimates a bitrate ceiling from measured data (starting from the ±6 CQ ≈ 2× bitrate rule of thumb — or the decay rate the encoder actually exhibited on previously processed files — and refining with actual measurements) to avoid jumping past the bitrate floor. When the bitrate floor is the binding constraint rather than VMAF, the search switches to bitrate targeting mode — testing additional sample CQs to compute the exact decay rate for the content and interpolating to the CQ that hits the floor. Repeats until the target is bracketed
4. **Encode** — full video is encoded at the best CQ found
5. **Verify & refine** — full-file VMAF and bitrate are checked against their targets (P5, the 5th-percentile worst-frame score, is measured and reported but not used as a gate). Misses in either direction trigger a corrective re-encode: a shortfall steps the CQ down, while VMAF landing well above target (with bitrate headroom over the floor) steps it up to reclaim wasted bitrate. Each jump is computed from the measured quality/bitrate slopes — converging in 1-2 iterations rather than stepping by one — and a re-encode predicted to trim less than ~3% bitrate is skipped as costing more than it saves
6. **Calibrate** — sample-vs-full deltas (bitrate ratio, VMAF offset, quality slope, bitrate decay) are cached per file and rolled into cross-file averages, so re-runs of the same file — and new files once a few have been processed — aim at the right CQ on the first probe

Files that end up *larger* after encoding are automatically deleted.

## License

MIT
