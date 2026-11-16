"""The only code that touches the operating system.

Safety model
------------
* The LLM never sees a "run command" tool. It picks from a fixed set of
  hardware tools (tools.py), and those tools can only use the commands listed
  in COMMANDS below, by id.
* Each command has a fixed argv. Nothing from the conversation is ever
  interpolated into it, with one exception (a block-device name for smartctl),
  which is checked against a strict pattern and against /sys/block first.
* Binaries are resolved only from root-owned system directories, never from
  $PATH, so a `lscpu` dropped in ~/bin can't be picked up. They run with
  shell=False, an empty stdin, a fixed minimal environment (LC_ALL=C for
  stable output), a timeout and an output cap.
* Commands that need root are run through `sudo -n` (non-interactive, never
  prompts). That only works if the user ran `hwqa --sudo` (cached credentials)
  or installed the narrow sudoers rule in packaging/. Otherwise the tool
  reports "needs root" and the assistant says so instead of guessing.
* File reads are limited to /proc and /sys hardware entries, os-release and
  the PCI/USB id databases, after resolving symlinks.

FixtureSystem replays recorded outputs from a directory. Tests use it, and it
lets you try the assistant on a machine that isn't Linux.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

TRUSTED_DIRS = ("/usr/sbin", "/usr/bin", "/sbin", "/bin", "/usr/lib/wsl/lib")
SAFE_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C", "LANG": "C"}
MAX_OUTPUT = 2_000_000

_BLOCK_DEVICE = re.compile(r"^(nvme\d{1,2}n\d{1,2}|sd[a-z]{1,2}|vd[a-z]{1,2}|mmcblk\d{1,2})$")


@dataclass(frozen=True)
class Param:
    """The single argument a command may take. `check` gets the value and the
    system, and must return the argv fragment or raise ValueError."""
    name: str
    check: Callable[[str, "System"], str]


def _block_device(value: str, system: "System") -> str:
    if not isinstance(value, str) or not _BLOCK_DEVICE.match(value):
        raise ValueError(f"not a block device name: {value!r}")
    if not system.exists(f"/sys/block/{value}"):
        raise ValueError(f"no such disk: {value}")
    return f"/dev/{value}"


@dataclass(frozen=True)
class CommandSpec:
    binary: str
    args: tuple = ()
    needs_root: bool = False
    timeout: float = 5.0
    param: Optional[Param] = None
    purpose: str = ""


COMMANDS: dict[str, CommandSpec] = {
    "lscpu": CommandSpec("lscpu", purpose="CPU model, cores, clocks, caches"),
    "lsblk": CommandSpec(
        "lsblk",
        ("-J", "-b", "-o", "NAME,KNAME,TYPE,SIZE,MODEL,VENDOR,ROTA,TRAN,RM,MOUNTPOINT,FSTYPE"),
        purpose="disks and partitions",
    ),
    "free": CommandSpec("free", ("-b",), purpose="memory usage (fallback for /proc/meminfo)"),
    "lspci": CommandSpec("lspci", ("-vmm", "-nn", "-k"), purpose="PCI devices: GPU, Wi-Fi, Ethernet"),
    "lsusb": CommandSpec("lsusb", purpose="USB devices"),
    "nvidia_smi": CommandSpec(
        "nvidia-smi",
        ("--query-gpu=name,memory.total,driver_version,pci.bus_id", "--format=csv,noheader,nounits"),
        timeout=10.0,
        purpose="NVIDIA GPU memory and driver",
    ),
    "dmidecode_memory": CommandSpec("dmidecode", ("-t", "memory"), needs_root=True,
                                    purpose="RAM slots, modules, maximum capacity"),
    "dmidecode_system": CommandSpec("dmidecode", ("-t", "system"), needs_root=True,
                                    purpose="system serial number"),
    "smartctl": CommandSpec("smartctl", ("-H", "-A", "-j"), needs_root=True, timeout=15.0,
                            param=Param("device", _block_device), purpose="disk health (SMART)"),
}

_READ_EXACT = {
    "/proc/cpuinfo", "/proc/meminfo", "/proc/version", "/proc/mounts",
    "/etc/os-release", "/usr/lib/os-release",
    "/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids",
    "/usr/share/hwdata/usb.ids", "/usr/share/misc/usb.ids",
}
_READ_PREFIXES = ("/sys/", "/proc/device-tree/")
# Kernel-provided symlinks whose target is per-process (/proc/mounts -> /proc/self/mounts).
# Nobody but the kernel can redirect these, so the resolved path needn't be on the list.
_KERNEL_LINKS = {"/proc/mounts"}


def read_allowed(path: str) -> bool:
    return path in _READ_EXACT or path.startswith(_READ_PREFIXES)


@dataclass
class CommandResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    error: Optional[str] = None      # missing | needs_root | timeout | failed | invalid
    message: str = ""


@dataclass
class AuditEntry:
    what: str
    outcome: str
    ms: int = 0


class System:
    """Interface shared by LiveSystem and FixtureSystem."""

    is_root: bool = False
    audit: list

    def run(self, cmd_id: str, param: Optional[str] = None) -> CommandResult: ...
    def read(self, path: str, max_bytes: int = 262_144) -> Optional[str]: ...
    def listdir(self, path: str) -> list[str]: ...
    def exists(self, path: str) -> bool: ...
    def readlink(self, path: str) -> Optional[str]: ...
    def has_command(self, cmd_id: str) -> bool: ...
    def kernel_release(self) -> str: ...
    def disk_usage(self, mountpoint: str) -> Optional[tuple[int, int, int]]: ...

    @property
    def privilege(self) -> str:
        """root | sudo | unknown | none: how root-only commands will be run."""
        return "root" if self.is_root else "none"


def resolve_binary(name: str) -> Optional[str]:
    for d in TRUSTED_DIRS:
        path = os.path.join(d, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode) or not os.access(path, os.X_OK):
            continue
        # Refuse binaries that someone other than root could have replaced.
        if st.st_uid != 0 or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            continue
        return path
    return None


class LiveSystem(System):
    def __init__(self, allow_sudo: bool = True):
        self.is_root = hasattr(os, "geteuid") and os.geteuid() == 0
        self.allow_sudo = allow_sudo
        self.audit: list[AuditEntry] = []
        self._which: dict[str, Optional[str]] = {}
        self._sudo_works: Optional[bool] = None

    # -------------------------------------------------------------- commands

    def which(self, binary: str) -> Optional[str]:
        if binary not in self._which:
            self._which[binary] = resolve_binary(binary)
        return self._which[binary]

    def has_command(self, cmd_id: str) -> bool:
        spec = COMMANDS.get(cmd_id)
        return bool(spec and self.which(spec.binary))

    @property
    def privilege(self) -> str:
        if self.is_root:
            return "root"
        if not self.allow_sudo or not self.which("sudo"):
            return "none"
        return {True: "sudo", False: "none", None: "unknown"}[self._sudo_works]

    def prime_sudo(self) -> bool:
        """Ask for the sudo password once, up front (hwqa --sudo). Later
        privileged commands use `sudo -n` and ride on the cached credentials."""
        sudo = self.which("sudo")
        if self.is_root or not sudo:
            return self.is_root
        ok = subprocess.run([sudo, "-v"], check=False).returncode == 0
        self._sudo_works = ok or None
        return ok

    def run(self, cmd_id: str, param: Optional[str] = None) -> CommandResult:
        spec = COMMANDS.get(cmd_id)
        if spec is None:
            # Not reachable from the LLM; tools only use ids from COMMANDS.
            raise KeyError(f"command {cmd_id!r} is not allow-listed")
        binary = self.which(spec.binary)
        if binary is None:
            self._log(cmd_id, "missing")
            return CommandResult(False, error="missing", message=f"{spec.binary} is not installed")

        argv = [binary, *spec.args]
        if spec.param is not None:
            try:
                argv.append(spec.param.check(param, self))
            except ValueError as exc:
                self._log(cmd_id, f"rejected: {exc}")
                return CommandResult(False, error="invalid", message=str(exc))

        via_sudo = spec.needs_root and not self.is_root
        if via_sudo:
            sudo = self.which("sudo")
            if not self.allow_sudo or sudo is None or self._sudo_works is False:
                self._log(" ".join(argv), "needs root")
                return CommandResult(False, error="needs_root", message=f"{spec.binary} needs root")
            argv = [sudo, "-n", "--", *argv]

        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=spec.timeout,
                env=SAFE_ENV,
                shell=False,
                check=False,
                close_fds=True,
            )
        except subprocess.TimeoutExpired:
            self._log(" ".join(argv), "timeout", start)
            return CommandResult(False, error="timeout", message=f"{spec.binary} timed out after {spec.timeout:.0f}s")
        except OSError as exc:
            self._log(" ".join(argv), f"error: {exc}", start)
            return CommandResult(False, error="failed", message=str(exc))

        out = proc.stdout[:MAX_OUTPUT].decode("utf-8", "replace")
        err = proc.stderr[:20_000].decode("utf-8", "replace")
        if via_sudo:
            if proc.returncode != 0 and ("password is required" in err or "a terminal is required" in err):
                self._sudo_works = False
                self._log(" ".join(argv), "needs root (sudo not configured)", start)
                return CommandResult(False, stderr=err, returncode=proc.returncode, error="needs_root",
                                     message=f"{spec.binary} needs root and passwordless sudo isn't set up")
            if proc.returncode == 0:
                self._sudo_works = True
        if proc.returncode != 0:
            # smartctl uses bit flags in its exit code; output is still valid JSON.
            if cmd_id == "smartctl" and out.strip().startswith("{"):
                self._log(" ".join(argv), f"exit {proc.returncode} (smartctl status bits)", start)
                return CommandResult(True, out, err, proc.returncode)
            self._log(" ".join(argv), f"exit {proc.returncode}", start)
            return CommandResult(False, out, err, proc.returncode, error="failed",
                                 message=(err.strip().splitlines() or [f"exit code {proc.returncode}"])[-1])
        self._log(" ".join(argv), "ok", start)
        return CommandResult(True, out, err, proc.returncode)

    # -------------------------------------------------------------- files

    def read(self, path: str, max_bytes: int = 262_144) -> Optional[str]:
        real = os.path.realpath(path)
        if not (read_allowed(path) and (read_allowed(real) or path in _KERNEL_LINKS)):
            raise PermissionError(f"read outside the allow-list: {path}")
        try:
            with open(real, "rb") as fh:
                return fh.read(max_bytes).decode("utf-8", "replace")
        except OSError:
            # Unreadable (root-only), or sysfs returning EINVAL for a value that
            # doesn't apply right now (e.g. link speed of a disconnected NIC).
            return None

    def listdir(self, path: str) -> list[str]:
        if not read_allowed(path.rstrip("/") + "/"):
            raise PermissionError(f"listing outside the allow-list: {path}")
        try:
            return sorted(os.listdir(path))
        except OSError:
            return []

    def exists(self, path: str) -> bool:
        allowed = read_allowed(path) or read_allowed(path.rstrip("/") + "/")
        return allowed and os.path.exists(path)

    def readlink(self, path: str) -> Optional[str]:
        if not read_allowed(path):
            raise PermissionError(f"readlink outside the allow-list: {path}")
        try:
            return os.readlink(path)
        except OSError:
            return None

    def kernel_release(self) -> str:
        return os.uname().release if hasattr(os, "uname") else ""

    def disk_usage(self, mountpoint: str) -> Optional[tuple[int, int, int]]:
        try:
            u = shutil.disk_usage(mountpoint)
            return u.total, u.used, u.free
        except OSError:
            return None

    def _log(self, what: str, outcome: str, start: Optional[float] = None) -> None:
        ms = int((time.monotonic() - start) * 1000) if start else 0
        self.audit.append(AuditEntry(what, outcome, ms))


@dataclass
class FixtureSystem(System):
    """Replays a recorded machine.

    Layout of a fixture directory:
      machine.json        {"is_root": false, "sudo": false, "kernel": "...", "disk_usage": {...}, "links": {...}}
      commands/<id>.out   stdout of an allow-listed command; absent = not installed
      fs/...              files under /proc, /sys, /etc mirrored by path
    """

    root: Path
    is_root: Optional[bool] = None
    sudo: Optional[bool] = None
    audit: list = field(default_factory=list)

    def __post_init__(self):
        self.root = Path(self.root)
        meta_path = self.root / "machine.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if self.is_root is None:
            self.is_root = bool(self.meta.get("is_root", False))
        if self.sudo is None:
            self.sudo = bool(self.meta.get("sudo", False))

    @property
    def privilege(self) -> str:
        return "root" if self.is_root else ("sudo" if self.sudo else "none")

    def _fs(self, path: str) -> Path:
        return self.root / "fs" / path.lstrip("/")

    def has_command(self, cmd_id: str) -> bool:
        return (self.root / "commands" / f"{cmd_id}.out").exists()

    def run(self, cmd_id: str, param: Optional[str] = None) -> CommandResult:
        spec = COMMANDS.get(cmd_id)
        if spec is None:
            raise KeyError(f"command {cmd_id!r} is not allow-listed")
        if not self.has_command(cmd_id):
            self.audit.append(AuditEntry(cmd_id, "missing"))
            return CommandResult(False, error="missing", message=f"{spec.binary} is not installed")
        if spec.param is not None:
            try:
                spec.param.check(param, self)
            except ValueError as exc:
                self.audit.append(AuditEntry(cmd_id, f"rejected: {exc}"))
                return CommandResult(False, error="invalid", message=str(exc))
        if spec.needs_root and not (self.is_root or self.sudo):
            self.audit.append(AuditEntry(cmd_id, "needs root"))
            return CommandResult(False, error="needs_root", message=f"{spec.binary} needs root")
        self.audit.append(AuditEntry(cmd_id, "ok (fixture)"))
        out = (self.root / "commands" / f"{cmd_id}.out").read_text(encoding="utf-8")
        return CommandResult(True, out, "", 0)

    def read(self, path: str, max_bytes: int = 262_144) -> Optional[str]:
        if not read_allowed(path):
            raise PermissionError(f"read outside the allow-list: {path}")
        if path in self.meta.get("root_only_files", []) and not self.is_root:
            return None
        p = self._fs(path)
        return p.read_text(encoding="utf-8")[:max_bytes] if p.is_file() else None

    def listdir(self, path: str) -> list[str]:
        p = self._fs(path)
        names = {c.name for c in p.iterdir() if not c.name.startswith(".")} if p.is_dir() else set()
        prefix = path.rstrip("/") + "/"
        for link in self.meta.get("links", {}):
            if link.startswith(prefix) and "/" not in link[len(prefix):]:
                names.add(link[len(prefix):])
        return sorted(names)

    def exists(self, path: str) -> bool:
        return self._fs(path).exists() or path in self.meta.get("links", {})

    def readlink(self, path: str) -> Optional[str]:
        return self.meta.get("links", {}).get(path)

    def kernel_release(self) -> str:
        return self.meta.get("kernel", "")

    def disk_usage(self, mountpoint: str) -> Optional[tuple[int, int, int]]:
        u = self.meta.get("disk_usage", {}).get(mountpoint)
        return tuple(u) if u else None
