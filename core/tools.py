"""Discovery and first-run download of the external tool binaries that
live under <repo>/tools (FFVship today; the Essential encoder's lookup
joins it with the engine split)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .ui import DIM, ORANGE, RESET

# core/ sits one level below the repo root, where the launchers and the
# tools/ directory live.
_ROOT = Path(__file__).resolve().parent.parent

_ffvship_exe = False  # False = not probed yet; None = probed, absent


def _gpu_vendor():
    """Pick the FFVship build for this machine's GPU.

    Returns 'nvidia', 'amd', or 'Vulkan' (the universal fallback build),
    matching the Vship release asset names FFVship_<vendor>.zip.

    Reads the display-adapter class key from the registry — instant.
    Only falls back to a PowerShell CIM query if that yields nothing,
    because PowerShell cold start makes that path take 10+ seconds.
    """
    out = ""
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Class"
            r"\{4d36e968-e325-11ce-bfc1-08002be10318}",
        ) as base:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(base, i)
                except OSError:
                    break
                try:
                    with winreg.OpenKey(base, sub) as k:
                        out += winreg.QueryValueEx(k, "DriverDesc")[0].lower() + "\n"
                except OSError:
                    pass
    except Exception:
        pass
    if not out:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=20,
            ).stdout.lower()
        except Exception:
            return "Vulkan"
    if any(k in out for k in ("nvidia", "geforce", "quadro")):
        return "nvidia"
    if any(k in out for k in ("amd", "radeon")):
        return "amd"
    return "Vulkan"


def _http_download(url, label, total=0):
    """GET url and return the body, rendering a progress bar while streaming.

    Same visual language as the encode bar. `total` is the expected size
    in bytes (falls back to the Content-Length header, then to a plain
    MB counter when neither is known).
    """
    import urllib.request

    chunks, done, bar_w = [], 0, 20
    # Keep the whole line under ~80 cols: a console-wrapped line defeats
    # the \r overwrite and the bar prints as a wall of repeated lines.
    if len(label) > 24:
        label = label[:23] + "…"

    def render():
        if total:
            filled = int(bar_w * done / total)
            sys.stdout.write(
                f"\r {ORANGE}{'download':<10}{RESET}{label} "
                f"[{'█' * filled}{'░' * (bar_w - filled)}] "
                f"{done / total * 100:5.1f}%  "
                f"{done / (1 << 20):.1f}/{total / (1 << 20):.1f}MB"
            )
        else:
            sys.stdout.write(
                f"\r {ORANGE}{'download':<10}{RESET}{label} "
                f"{done / (1 << 20):.1f}MB"
            )
        sys.stdout.flush()

    # Render the 0% bar before opening the connection — the release hosts
    # can take 20s+ to answer, and a blank console reads as a hang.
    render()
    with urllib.request.urlopen(url, timeout=120) as r:
        total = total or int(r.headers.get("Content-Length") or 0)
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
            done += len(chunk)
            render()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return b"".join(chunks)


def _download_ffvship(dest):
    """First-run fetch of FFVship from the Vship releases into dest/.

    Picks the build matching the detected GPU and extracts the zip flat
    (FFVship.exe + DLLs directly in dest, no nested folder). Returns the
    exe path, or None on any failure — FFVship stays strictly optional,
    so a dead network or an odd GPU must never break the pipeline.
    """
    import io
    import urllib.request
    import zipfile

    if sys.platform != "win32":
        return None  # published zips are Windows binaries
    vendor = _gpu_vendor()
    api = "https://codeberg.org/api/v1/repos/Line-fr/Vship/releases/latest"
    sys.stdout.write(
        f" {ORANGE}{'download':<10}{RESET}FFVship "
        f"{DIM}contacting codeberg.org…{RESET}"
    )
    sys.stdout.flush()
    try:
        with urllib.request.urlopen(api, timeout=60) as r:
            rel = json.load(r)
        want = f"ffvship_{vendor}.zip".lower()
        asset = next(
            a for a in rel.get("assets", []) if a["name"].lower() == want
        )
        data = _http_download(
            asset["browser_download_url"], asset["name"], asset["size"]
        )
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for m in z.infolist():
                if not m.is_dir():
                    (dest / Path(m.filename).name).write_bytes(z.read(m))
        exe = dest / "FFVship.exe"
        return exe if exe.is_file() else None
    except Exception as e:
        print(f"\r{DIM}FFVship download failed ({e}){' ' * 24}{RESET}")
        return None


def find_ffvship_optional():
    """Locate FFVship under ./tools (any depth) or PATH; None if absent.

    On Windows, a miss triggers a one-time auto-download of the build
    matching the detected GPU into tools/FFVship/.
    """
    global _ffvship_exe
    if _ffvship_exe is False:
        _ffvship_exe = None
        tools = _ROOT / "tools"
        if tools.is_dir():
            hits = [
                h for pat in ("FFVship.exe", "FFVship")
                for h in sorted(tools.rglob(pat)) if h.is_file()
            ]
            if hits:
                _ffvship_exe = hits[0]
        if _ffvship_exe is None:
            w = shutil.which("FFVship")
            if w:
                _ffvship_exe = Path(w)
        if _ffvship_exe is None:
            _ffvship_exe = _download_ffvship(tools / "FFVship")
    return _ffvship_exe


def find_tool(patterns, fallback_name, hint):
    """Locate a tool binary under ./tools (any depth), else PATH.

    Glob patterns keep the lookup version-agnostic so dropping in an
    upgraded binary (new version in the filename) keeps working.
    """
    tools = _ROOT / "tools"
    if tools.is_dir():
        for pat in patterns:
            hits = sorted(tools.rglob(pat))
            for h in hits:
                if h.is_file():
                    return h
    w = shutil.which(fallback_name)
    if w:
        return Path(w)
    raise FileNotFoundError(
        f"{fallback_name} not found under {tools} or PATH.\n  {hint}"
    )


def _download_encoder(dest):
    """First-run fetch of SVT-AV1-Essential from its GitHub releases
    into dest/ (tools/SVT-AV1-Essential, mirroring FFVship's subfolder).

    Release assets are bare executables. Prefers the CPU-Optimized build
    but smoke-tests it and falls back to Generic — Optimized uses newer
    CPU instructions and dies with an illegal instruction on older chips.
    Keeps the release filename (version visible, find_tool glob matches).
    Returns the exe path, or None so find_encoder can raise its usual
    FileNotFoundError with the manual-install hint.
    """
    import urllib.request

    plat = {"win32": "Windows", "darwin": "MacOS",
            "linux": "Linux"}.get(sys.platform)
    if plat is None:
        return None
    wanted = (["MacOS_Arm"] if plat == "MacOS"
              else [f"{plat}_Optimized", f"{plat}_Generic"])
    api = ("https://api.github.com/repos/nekotrix/SVT-AV1-Essential/"
           "releases/latest")
    sys.stdout.write(
        f" {ORANGE}{'download':<10}{RESET}SvtAv1EncApp "
        f"{DIM}contacting github.com…{RESET}"
    )
    sys.stdout.flush()
    try:
        with urllib.request.urlopen(api, timeout=60) as r:
            rel = json.load(r)
    except Exception as e:
        print(f"\r{DIM}encoder download failed ({e}){' ' * 24}{RESET}")
        return None
    for suffix in wanted:
        asset = next(
            (a for a in rel.get("assets", []) if suffix in a["name"]), None)
        if asset is None:
            continue
        try:
            data = _http_download(
                asset["browser_download_url"], "SvtAv1EncApp", asset["size"])
            dest.mkdir(parents=True, exist_ok=True)
            exe = dest / asset["name"]
            exe.write_bytes(data)
            if sys.platform != "win32":
                os.chmod(exe, 0o755)
            probe = subprocess.run([str(exe), "--version"],
                                   capture_output=True, timeout=15)
            if probe.returncode == 0:
                return exe
            print(f"{DIM}{asset['name']} can't run on this CPU — "
                  f"trying Generic{RESET}")
            exe.unlink()
        except Exception as e:
            print(f"\r{DIM}{asset['name']} failed ({e}){' ' * 24}{RESET}")
    return None


def find_encoder():
    try:
        return find_tool(
            ["SvtAv1EncApp*.exe", "SvtAv1EncApp*"], "SvtAv1EncApp",
            "Download from https://github.com/nekotrix/SVT-AV1-Essential/releases",
        )
    except FileNotFoundError:
        exe = _download_encoder(_ROOT / "tools" / "SVT-AV1-Essential")
        if exe:
            return exe
        raise
