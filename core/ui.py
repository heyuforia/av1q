"""Terminal presentation: ANSI colors, shared glyphs, and small formatters.

Importing this module initializes colorama (or enables VT processing on
Windows) exactly once, for every launcher.
"""

import os

try:
    from colorama import init as colorama_init
    colorama_init()
except ImportError:
    if os.name == "nt":
        os.system("")


GREEN = "\033[38;5;46m"
ORANGE = "\033[38;5;208m"
PURPLE = "\033[38;5;141m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
SEP = f"{DIM}{'─' * 48}{RESET}"
MIDDOT = "·"  # named constant for readability where it's used as a separator


def fmt_time(seconds):
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"


def fmt_size(n):
    """Human-readable byte size: GB at/above 1e9, otherwise MB."""
    return f"{n / 1e9:.2f}GB" if n >= 1e9 else f"{n / 1e6:.1f}MB"


def vmaf_pass_color(mean, target, tol):
    """GREEN when a VMAF mean meets target within tolerance, else no color."""
    return GREEN if mean >= target - tol else ""


def fmt_s2(s2):
    """Dim SSIMU2 info field appended to a VMAF result line ('' when absent)."""
    if not s2:
        return ""
    return f"  {DIM}SSIMU2 {s2['mean']:.2f}  P5 {s2['p5']:.2f}{RESET}"
