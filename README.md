# AV1Q

Automatically encode videos to AV1 at the best quality-to-size ratio, using VMAF to find the optimal settings for each file.

![AV1Q-Cli-Screenshot](https://raw.githubusercontent.com/heyuforia/av1q/refs/heads/main/av1q-cli.png)

## What it does

Drop videos into a folder and run the script. Each file is encoded to AV1 at a quality level that hits a target VMAF score, so there's no guessing at CRF values and no manual test encodes.

For every file:

1. **Detect scenes** and extract short samples from the most complex parts of the video
2. **Search for the optimal CQ** by encoding only those samples, measuring VMAF after each attempt
3. **Encode the full video** at the chosen CQ, then verify the final VMAF matches the target
4. **Refine** if the result missed, using the measured quality and bitrate slopes to jump straight to the corrected CQ

## Features

- **VMAF-targeted encoding.** Hits a perceptual quality target instead of a fixed CRF.
- **Resolution-aware defaults.** VMAF targets auto-select by resolution: 94 for HD, 93 for SD, 90 for 4K.
- **Bitrate floors.** Per-resolution minimums (1 Mbps at 720p, 1.8 at 1080p, 2.5 at 1440p, 4.5 at 2160p, 8 at 4320p) prevent VMAF-misleading low-bitrate encodes. They're starvation backstops, not targets.
- **Scene-based sampling.** Fast quality estimation, so the search never encodes the whole file.
- **P5 quality reporting.** The 5th-percentile worst-frame VMAF is reported next to the mean, so you can spot files whose worst moments lag.
- **SSIMULACRA2 reporting.** Every VMAF score is shown next to a GPU-computed [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2) score as a second opinion. Informational only, it never influences the encode.
- **HDR and color preservation.** Carries over color primaries, transfer, matrix, and range.
- **10-bit output** by default, with film grain synthesis.
- **Hardware-accelerated decoding.** CUDA, D3D11VA (Windows), VideoToolbox (macOS), and VAAPI (Linux) speed up quality measurement and scene detection. Encoding itself is always CPU.
- **Optional auto-crop.** `--auto-crop` detects letterbox and pillarbox bars before each encode, or use the standalone `av1q-crop.py` to pre-scan a library. Confidence-gated, so ambiguous detections are never silently applied.
- **File-based caching.** Analysis and measurements are reused on later runs, and interrupted searches resume where they left off.
- **Resumable encodes.** Full encodes of sources 15 minutes and longer are written as finalized segments, so an interrupted encode picks up at the last finished segment instead of restarting from frame 0 (disable with `--no-resume`).
- **Batch processing** with recursive subdirectory support.
- **Cross-platform.** Windows, macOS, Linux.

## Requirements

- **Python 3.8+**
- **ffmpeg** and **ffprobe** in PATH, built with:
  - `libsvtav1` (SVT-AV1 encoder)
  - `libvmaf` (VMAF quality metrics)

Most ffmpeg builds from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (Windows) or [BtbN](https://github.com/BtbN/FFmpeg-Builds/releases) (Linux/Windows) include both. On macOS: `brew install ffmpeg`.

### Optional

- [colorama](https://pypi.org/project/colorama/) for Windows terminal colors (`pip install colorama`). Works without it.
- [FFVship](https://codeberg.org/Line-fr/Vship) powers the SSIMULACRA2 info scores. On Windows it downloads automatically on first run, picking the build that matches your GPU (NVIDIA, AMD, or generic Vulkan). Without it the SSIMULACRA2 column disappears and everything else works the same.

## Usage

**Basic.** Processes all videos in `./Video Input`, outputs to `./AV1 Output`:

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

**Start the search at a known CQ.** When run in a terminal, the script asks once at startup:

```
Seed CQ 18–38 (Enter = auto):
```

Press Enter for the automatic seed, or type a CQ to start every file's search there. This helps when a batch of similar clips all land around the same value, since a good seed can collapse the search to a single encode. The seed is only a starting point: VMAF is still measured and the search still corrects a wrong guess. `--seed-cq 24` does the same without the prompt, and piped or scripted runs skip the prompt entirely.

A seed alone won't redo files that a previous run already encoded, since their finished search result is reused. When a seed is given interactively, av1q lists those files and asks once whether to re-encode them with a fresh search from the seed or keep the previous results.

**Encode at a fixed CQ.** When you already know the CQ you want, `--force-cq` skips the search entirely and encodes each file at exactly that value, with no sampling, no VMAF measurement, and no refinement:

```
python av1q.py --force-cq 33
```

Finished files are remembered and skipped on later runs, and outputs at different forced values sit side by side in the output folder, so encoding a small ladder like 30, 33, 36 is an easy way to compare quality by eye. The chosen value is treated as final, so the output is kept even when it ends up larger than the source.

**Auto-crop letterboxed or pillarboxed videos:**

```
python av1q.py --auto-crop
```

Before each encode the script samples 8 short windows, detects black bars, and applies the crop if the detection is confident. Results are cached as `<file>.crop.json` sidecars beside each source, so re-runs reuse them. Low-confidence detections (dark sources, mixed aspect ratios) are saved for manual review but never auto-applied. Pass `--no-crops` to ignore sidecars entirely.

To pre-scan a whole library before encoding, so you can review borderline sidecars first, the companion `av1q-crop.py` runs the same detection standalone and writes the same sidecar format:

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
| `--samples` | `8` | Base number of sample segments, scales up with duration |
| `--seed-cq` | Prompted / auto | Starting CQ for the search (skips the interactive prompt) |
| `--force-cq` | | Encode at exactly this CQ, skipping sampling, search, VMAF, and refinement |
| `--no-10bit` | | Disable forced 10-bit encoding |
| `--no-recurse` | | Don't process subdirectories |
| `--overwrite` | | Re-encode even if output exists |
| `--dry-run` | | Find optimal CQ but skip final encoding |
| `--auto-crop` | | Detect letterbox/pillarbox inline before each encode |
| `--no-crops` | | Ignore `*.crop.json` sidecars (auto-applied by default) |
| `--no-resume` | | Disable resumable segmented encoding for long sources |

## How it works

The search is adaptive, similar to Newton's method: it converges on the right CQ in 2 to 4 iterations instead of testing every value.

1. **Analyze.** Scene detection finds visually distinct segments, and packet-size analysis ranks them by complexity. Both read the container without decoding, so this stays fast on long 4K sources.
2. **Sample.** The most complex scenes are cut out and concatenated into one short clip. The number of scenes sampled grows with duration, so a feature-length film is represented as well as a short one. Files roughly 15 to 60 seconds long get a smaller plan of 3 clips at 2 seconds each. Files at 15 seconds and under skip sampling and search on the full file, where a probe would cover most of the file anyway.
3. **Search.** The sample is encoded at a seed CQ derived from the source's bitrate headroom over the floor, VMAF is measured, and the next CQ is estimated from the slope of quality against CQ. The search also tracks the bitrate floor and estimates a ceiling from the measured data, so it never jumps past it. When the floor is the binding constraint rather than VMAF, the search switches to bitrate targeting: it measures the exact bitrate decay rate for the content and interpolates to the CQ that lands on the floor.
4. **Encode.** The full video is encoded at the best CQ found.
5. **Verify and refine.** Full-file VMAF and bitrate are checked against their targets. P5 is measured and reported but is not a gate. A miss in either direction triggers a corrective re-encode: a shortfall lowers the CQ, while VMAF landing well above target with bitrate headroom to spare raises it to reclaim wasted bitrate. Each jump is sized from the measured slopes and converges in 1 or 2 passes. A re-encode predicted to trim less than about 3% of bitrate is skipped as costing more than it saves.
6. **Calibrate.** Sample-to-full deltas (bitrate ratio, VMAF offset, quality slope, bitrate decay) are cached per file and rolled into cross-file averages. Re-runs of the same file, and new files once a few have been processed, aim at the right CQ on the first probe.

Files that end up larger after encoding are deleted automatically. Forced encodes are the exception and are always kept.

## License

MIT
