"""Pre-encode sharpening through ffmpeg's cas filter (AMD FidelityFX
Contrast Adaptive Sharpening), at SHARPEN_STRENGTH unless --sharpen sets
another strength or 0 for off.

The encoder is fed the sharpened picture, and the VMAF reference is
sharpened the same way at the same point of its chain, so the target
measures fidelity to what was encoded. Scored against the unsharpened
source instead, the standard VMAF models credit the sharpening itself
(Netflix's "enhancement gain", the reason the NEG models exist), and the
search would spend that credit on a higher quantizer.
"""


def parse_strength(raw):
    """--sharpen as typed -> cfg['sharpen']: None when absent or zero
    (off), else rounded to two decimals so 0.5 and 0.50 are one setting.
    Raises ValueError outside cas's 0-1, nan included."""
    if raw is None:
        return None
    if not 0 <= raw <= 1:
        raise ValueError(raw)
    return round(raw, 2) or None


def sharpen_token(strength):
    """Filename and cache-key token for a strength ('' when off), e.g.
    '_sharp0.5'. The launchers round the strength to two decimals, so
    one strength always spells one token."""
    return f"_sharp{strength:g}" if strength else ""


def sharpen_filter(meta):
    """The cas filter for this file's picture, or None when off. Every
    chain places it right after the crop: the bars are never sharpened,
    and the encoder and the VMAF reference see one operation."""
    strength = meta.get("sharpen")
    return f"cas=strength={strength:g}" if strength else None
