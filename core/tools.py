"""Where the external binaries come from: ffmpeg and ffprobe, where a
build dropped into the av1q folder outranks PATH, and the tool binaries
under <repo>/tools (FFVship, SvtAv1EncApp), each downloaded at a pinned
version on the first run that finds none."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .ui import DIM, RESET, label as stage_label
from .util import run_cmd, suppress_win_error_dialog

# core/ sits one level below the repo root, where the launchers and the
# tools/ directory live.
_ROOT = Path(__file__).resolve().parent.parent

_ffvship_exe = False  # False = not probed yet; None = probed, absent
_ff_pair = None  # memo: (ffmpeg, ffprobe) commands, resolved once
_ff_options = None  # memo: option names the resolved ffmpeg takes

# First-run downloads are pinned to the builds this code was tested
# with, and a file is refused unless its SHA-256 matches: a binary that
# will run here must be byte for byte the tested one. Every lookup takes
# a copy already under tools/ or on PATH first, so raising a pin changes
# fresh installs only; delete the old copy to fetch the new one.
_FFVSHIP_VERSION = "v5.1.1"
_FFVSHIP_URL = ("https://codeberg.org/Line-fr/Vship/releases/download/"
                "{version}/FFVship_{vendor}.zip")
_FFVSHIP_SHA256 = {  # per GPU build, keyed the way _gpu_vendor names it
    "nvidia":
        "19bee2924482b4ae7a1939008b5b7fc71c0d827d437522b3a5d982d13e11bc85",
    "amd":
        "cd49e059f428d156ed8a61b213941644dab905edb37ade125a6fbfcb377f4b05",
    "Vulkan":
        "412ebf2be21ad3a3afec191d4156f9d45d00845e01044506703c26d10da3a659",
}
_ENCODER_VERSION = "4.0.1"
_ENCODER_RELEASE = ("https://github.com/nekotrix/SVT-AV1-Essential/"
                    f"releases/tag/v{_ENCODER_VERSION}-Essential")
_ENCODER_URL = ("https://github.com/nekotrix/SVT-AV1-Essential/releases/"
                "download/v{version}-Essential/{name}")
# Per platform, in the order tried. The Optimized builds use newer CPU
# instructions and die on older chips, so the Generic build follows.
_ENCODER_BUILDS = {
    "win32": (
        ("Windows_Optimized.exe",
         "7ab3a726b5644562ca944f8dcb5df4c3b68f7ac20d308fa8269932ce24cfa16a"),
        ("Windows_Generic.exe",
         "ae27af4a7aaaf4e97183c0be0336da1b9742ed18d9ba02c1b90094a159f1935c"),
    ),
    "linux": (
        ("Linux_Optimized",
         "b4c26f03e37981324b681c4bf87ee89575fb490f60b13c54219dccf0c357818e"),
        ("Linux_Generic",
         "57dc7352148effeaa7bc45d20f55b50ea09215f49ddde29e7301a7e35944405b"),
    ),
    "darwin": (
        ("MacOS_Arm",
         "ea56e7314dcce6c25372c24f8a5d4b9a2521cc1af7c91d2598fbd71283654ffb"),
    ),
}


def _exe_name(stem):
    """`stem` spelled the way this platform names an executable."""
    return f"{stem}.exe" if os.name == "nt" else stem


def _exe_in(directory, name):
    """The `name` executable sitting directly in `directory`, or None.

    Windows requires the .exe suffix to execute at all, so accepting an
    extensionless file there would bless a Linux build extracted by
    mistake: the preflight would pass and the failure would surface as a
    RuntimeError cascade deep in a run instead of "ffmpeg not found".
    """
    exe = directory / _exe_name(name)
    return exe if exe.is_file() else None


def _local_ff_dirs():
    """Folders that may hold a dropped-in ffmpeg build, in priority order.

    The repo root itself (binaries dropped next to the launchers), then
    <repo>/ffmpeg, then <repo>/tools alongside the other vendored
    binaries. The two named folders are searched recursively, shallowest
    first, so an unzipped release with its binaries in bin/ is found
    while a copy sitting directly in the folder still wins. The root is
    NOT recursed: the work folders under it hold whole video libraries.
    """
    yield _ROOT
    for base in (_ROOT / "ffmpeg", _ROOT / "tools"):
        if not base.is_dir():
            continue
        yield base
        yield from sorted(
            (d for d in base.rglob("*") if d.is_dir()),
            key=lambda d: (len(d.parts), d.as_posix()),
        )


def _resolve_ff():
    """(ffmpeg, ffprobe) commands: a local pair when one exists, else the
    bare names for PATH lookup.

    Both come from the same folder or neither does. Pairing a local
    ffmpeg with a PATH ffprobe mixes two builds, and the version skew
    surfaces as parse failures deep in a run instead of as one clear
    error. The bare names are the fallback (never None) so every command
    line stays valid and a missing ffmpeg is reported by the pipeline's
    preflight rather than a traceback.
    """
    global _ff_pair
    if _ff_pair is None:
        _ff_pair = ("ffmpeg", "ffprobe")
        for d in _local_ff_dirs():
            found = (_exe_in(d, "ffmpeg"), _exe_in(d, "ffprobe"))
            if all(found):
                _ff_pair = tuple(str(e) for e in found)
                break
    return _ff_pair


def ffmpeg_exe():
    """The ffmpeg command every call site invokes."""
    return _resolve_ff()[0]


def ffprobe_exe():
    """The ffprobe command every call site invokes."""
    return _resolve_ff()[1]


def local_ffmpeg_dir():
    """Folder the dropped-in ffmpeg came from, or None when the run is
    using the PATH build."""
    exe = _resolve_ff()[0]
    return Path(exe).parent if exe != "ffmpeg" else None


def have_ffmpeg():
    """True when both ffmpeg and ffprobe resolve to something runnable."""
    return all(os.path.isabs(c) or shutil.which(c) for c in _resolve_ff())


def missing_ffmpeg_components(encoders=(), filters=()):
    """Names among `encoders` / `filters` that the resolved ffmpeg build
    lacks, asked of the binary itself (-encoders and -filters list what
    was compiled in; each row is flags, name, description). Raises like
    run_cmd when ffmpeg cannot be executed at all."""
    missing = []
    for flag, wanted in (("-encoders", encoders), ("-filters", filters)):
        if not wanted:
            continue
        out = run_cmd([ffmpeg_exe(), "-hide_banner", flag]).stdout
        present = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) > 1:
                present.add(parts[1])
        missing += [name for name in wanted if name not in present]
    return missing


def ffmpeg_options():
    """Names, without the dash, of the command-line options the resolved
    ffmpeg build takes, read once from its long help, where each option
    opens a line as '-name[:<stream_spec>] <arg>  description'.

    ffmpeg fails a whole command on an option its build does not know,
    so an option newer than the oldest build this runs on is passed
    only when it is listed here. Empty when ffmpeg cannot run: every
    caller then leaves its option out."""
    global _ff_options
    if _ff_options is None:
        try:
            out = run_cmd([ffmpeg_exe(), "-hide_banner", "-h", "long"]).stdout
        except (OSError, RuntimeError):
            out = ""
        _ff_options = frozenset(
            line.split(None, 1)[0][1:].split("[", 1)[0]
            for line in out.splitlines() if line.startswith("-")
        )
    return _ff_options


def _find_in_tools(stem):
    """First binary under <repo>/tools, at any depth in path order, whose
    name matches the glob `stem` spelled as this platform names an
    executable; None when there is none.

    The same rule as _exe_in: on Windows only a .exe counts, so a Linux
    build or a release archive sharing the name is passed over instead
    of found and then failed on at every file.
    """
    tools = _ROOT / "tools"
    if not tools.is_dir():
        return None
    for hit in sorted(tools.rglob(_exe_name(stem))):
        if hit.is_file():
            return hit
    return None


def _gpu_vendor():
    """Pick the FFVship build for this machine's GPU.

    Returns 'nvidia', 'amd', or 'Vulkan' (the universal fallback build),
    matching the Vship release asset names FFVship_<vendor>.zip.

    Reads the display-adapter class key from the registry, which is
    instant. Only falls back to a PowerShell CIM query if that yields
    nothing, because PowerShell cold start makes that path take 10+
    seconds.
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
    MB counter when neither is known). Off a terminal the bar is never
    redrawn, as with the encode bars: only the finished line is written.
    On failure a drawn bar is cleared, so the caller's reason starts on
    a clean line.
    """
    import urllib.request

    chunks, done, bar_w = [], 0, 20
    tty = sys.stdout.isatty()
    redraw = "\r" if tty else ""
    # Keep the whole line under ~80 cols: a console-wrapped line defeats
    # the \r overwrite and the bar prints as a wall of repeated lines.
    if len(label) > 24:
        label = label[:23] + "…"

    def render(final=False):
        if not (tty or final):
            return
        if total:
            filled = int(bar_w * done / total)
            body = (
                f"[{'█' * filled}{'░' * (bar_w - filled)}] "
                f"{done / total * 100:5.1f}%  "
                f"{done / (1 << 20):.1f}/{total / (1 << 20):.1f}MB"
            )
        else:
            body = f"{done / (1 << 20):.1f}MB"
        sys.stdout.write(
            f"{redraw}{stage_label('download')}{label} {body}"
        )
        sys.stdout.flush()

    # Render the 0% bar before opening the connection: the release hosts
    # can take 20s+ to answer, and a blank console reads as a hang.
    render()
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            total = total or int(r.headers.get("Content-Length") or 0)
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
                done += len(chunk)
                render()
    except BaseException:
        if tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        raise
    render(final=True)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return b"".join(chunks)


def _fetch_pinned(url, label, sha256):
    """The body of a pinned release file; raises ValueError unless its
    SHA-256 is the pinned one."""
    data = _http_download(url, label)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("checksum mismatch, the file was refused")
    return data


def _download_ffvship(dest):
    """First-run fetch of the pinned FFVship build for this machine's GPU
    into dest/ (FFVship.exe plus its DLLs, flat).

    Returns the exe path, or None on any failure with its reason printed:
    FFVship stays strictly optional, so a dead network or an odd GPU must
    never break the pipeline.
    """
    import io
    import zipfile

    if sys.platform != "win32":
        return None  # published zips are Windows binaries
    vendor = _gpu_vendor()
    name = f"FFVship {_FFVSHIP_VERSION} {vendor}"
    exe = dest / "FFVship.exe"
    staged = dest / f"partial-{exe.name}"
    try:
        data = _fetch_pinned(
            _FFVSHIP_URL.format(version=_FFVSHIP_VERSION, vendor=vendor),
            name, _FFVSHIP_SHA256[vendor],
        )
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            # Flattened to base names: the release nests its files in one
            # folder, and a base name joined to dest cannot leave it.
            files = {
                Path(m.filename).name: m
                for m in z.infolist() if not m.is_dir()
            }
            if exe.name not in files:
                raise ValueError(f"no {exe.name} in the archive")
            dest.mkdir(parents=True, exist_ok=True)
            for n, m in files.items():
                if n != exe.name:
                    (dest / n).write_bytes(z.read(m))
            # The exe lands last and whole: an install cut short leaves
            # no FFVship.exe, so the next run downloads again instead of
            # finding a copy with DLLs missing.
            staged.write_bytes(z.read(files[exe.name]))
        staged.replace(exe)
        return exe
    except Exception as e:
        print(f"{DIM}{name} download failed ({e}){RESET}")
        return None
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass


def find_ffvship_optional():
    """Locate FFVship under ./tools (any depth) or PATH; None if absent.

    On Windows, a miss triggers a one-time auto-download of the pinned
    build matching the detected GPU into tools/FFVship/.
    """
    global _ffvship_exe
    if _ffvship_exe is False:
        on_path = shutil.which("FFVship")
        _ffvship_exe = (
            _find_in_tools("FFVship")
            or (Path(on_path) if on_path else None)
            or _download_ffvship(_ROOT / "tools" / "FFVship")
        )
    return _ffvship_exe


def _download_encoder(dest):
    """First-run fetch of the pinned SVT-AV1-Essential build into dest/,
    trying this platform's builds in order.

    Each build is written under a name the lookup never matches and
    smoke-tested there, then renamed to its release name (version
    visible, matched by the lookup's glob). So neither a download cut
    short nor a build this CPU cannot run is ever found by a later run.
    Returns the exe path, or None with each reason printed.
    """
    for build, sha256 in _ENCODER_BUILDS.get(sys.platform, ()):
        name = f"SvtAv1EncApp-{_ENCODER_VERSION}-Essential-{build}"
        exe = dest / name
        staged = dest / f"partial-{name}"
        try:
            data = _fetch_pinned(
                _ENCODER_URL.format(version=_ENCODER_VERSION, name=name),
                f"SvtAv1EncApp v{_ENCODER_VERSION}", sha256,
            )
            dest.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
            if sys.platform != "win32":
                os.chmod(staged, 0o755)
            # A build this CPU cannot run dies on an illegal instruction,
            # and the crash box would hold the run until clicked away.
            with suppress_win_error_dialog():
                probe = subprocess.run([str(staged), "--version"],
                                       capture_output=True, timeout=15)
            if probe.returncode == 0:
                staged.replace(exe)
                return exe
            print(f"{DIM}{name} can't run on this CPU{RESET}")
        except Exception as e:
            print(f"{DIM}{name} failed ({e}){RESET}")
        finally:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass
    return None


def find_encoder():
    """The SvtAv1EncApp binary: under ./tools (any depth), else PATH,
    else the pinned build downloaded into tools/SVT-AV1-Essential/.
    Raises FileNotFoundError when all three come up empty.

    The glob keeps the lookup version-agnostic, so a binary dropped in
    by hand is found whatever version its filename carries.
    """
    on_path = shutil.which("SvtAv1EncApp")
    exe = (
        _find_in_tools("SvtAv1EncApp*")
        or (Path(on_path) if on_path else None)
        or _download_encoder(_ROOT / "tools" / "SVT-AV1-Essential")
    )
    if exe is None:
        raise FileNotFoundError(
            f"SvtAv1EncApp not found under {_ROOT / 'tools'} or PATH, and"
            f" none could be downloaded.\n  Download it from"
            f" {_ENCODER_RELEASE}"
        )
    return exe
