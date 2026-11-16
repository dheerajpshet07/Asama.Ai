"""The safety properties the design depends on."""
import subprocess
import sys

import pytest

from hwqa import system as sysmod
from hwqa.system import COMMANDS, SAFE_ENV, LiveSystem, read_allowed
from hwqa.toolbox import ToolBox, redact


# ---------------------------------------------------------------- what the model can reach

def test_model_cannot_run_arbitrary_commands(laptop):
    box = ToolBox(laptop)
    assert not any("command" in name or "shell" in name or "exec" in name for name in box.tools)
    res = box.call("run_command", {"cmd": "dmidecode; rm -rf /"})
    assert res["ok"] is False and "unknown tool" in res["error"]


@pytest.mark.parametrize("args, error", [
    ({"device": "nvme0n1; rm -rf /"}, "invalid value"),
    ({"device": "../../etc/shadow"}, "invalid value"),
    ({"device": 7}, "must be a string"),
    ({"device": "nvme0n1", "flags": "-s on"}, "unexpected argument"),
    ({}, "missing required argument"),
])
def test_tool_arguments_are_validated(laptop, args, error):
    res = ToolBox(laptop).call("get_disk_health", args)
    assert res["ok"] is False and error in res["error"]


def test_command_param_must_name_an_existing_disk(laptop):
    res = laptop.run("smartctl", "sdz")
    assert not res.ok and res.error == "invalid"


def test_unknown_command_id_is_a_programming_error(laptop):
    with pytest.raises(KeyError):
        laptop.run("rm")


def test_every_allow_listed_command_is_read_only():
    # Guard against someone adding a mutating flag to the table later.
    dangerous = {"-s", "--set", "-w", "--write", "-t", "--test", "-X", "--abort"}
    for cmd_id, spec in COMMANDS.items():
        args = set(spec.args)
        if spec.binary == "dmidecode":
            assert args <= {"-t", "memory", "system"}
        else:
            assert not args & dangerous, cmd_id


# ---------------------------------------------------------------- files

@pytest.mark.parametrize("path", ["/etc/shadow", "/home/user/.ssh/id_rsa", "/proc/self/environ",
                                  "/proc/1/cmdline", "/root/.bashrc", "/etc/passwd"])
def test_reads_outside_allow_list_are_refused(path):
    assert not read_allowed(path)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs a real /sys")
def test_symlink_escape_is_refused(tmp_path):
    s = LiveSystem()
    with pytest.raises(PermissionError):
        s.read("/sys/../etc/shadow")


def test_fixture_enforces_allow_list(laptop):
    with pytest.raises(PermissionError):
        laptop.read("/etc/shadow")


# ---------------------------------------------------------------- process execution

class FakeRun:
    def __init__(self, returncode=0, stdout=b"ok", stderr=b""):
        self.calls = []
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return self.result


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(sysmod, "resolve_binary", lambda name: f"/usr/bin/{name}")
    s = LiveSystem()
    s.is_root = False
    return s


def test_commands_run_without_shell_with_fixed_env(live, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    assert live.run("lsblk").ok
    argv, kw = fake.calls[0]
    assert argv == ["/usr/bin/lsblk", *COMMANDS["lsblk"].args]
    assert kw["shell"] is False and kw["env"] == SAFE_ENV and kw["stdin"] is subprocess.DEVNULL
    assert kw["timeout"] == COMMANDS["lsblk"].timeout


def test_root_commands_use_non_interactive_sudo(live, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    live.run("dmidecode_memory")
    argv, _ = fake.calls[0]
    assert argv == ["/usr/bin/sudo", "-n", "--", "/usr/bin/dmidecode", "-t", "memory"]


def test_sudo_password_prompt_is_reported_not_waited_on(live, monkeypatch):
    fake = FakeRun(returncode=1, stdout=b"", stderr=b"sudo: a password is required\n")
    monkeypatch.setattr(subprocess, "run", fake)
    res = live.run("dmidecode_memory")
    assert res.error == "needs_root" and "sudo" in res.message
    assert live.privilege == "none"
    # and we don't keep trying
    live.run("dmidecode_memory")
    assert len(fake.calls) == 1


def test_no_sudo_flag_means_no_escalation(monkeypatch):
    monkeypatch.setattr(sysmod, "resolve_binary", lambda name: f"/usr/bin/{name}")
    s = LiveSystem(allow_sudo=False)
    s.is_root = False
    fake = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    assert s.run("dmidecode_memory").error == "needs_root"
    assert fake.calls == []


def test_param_goes_through_validation_into_argv(live, monkeypatch):
    fake = FakeRun(stdout=b"{}")
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(live, "exists", lambda p: p == "/sys/block/nvme0n1")
    live.run("smartctl", "nvme0n1")
    argv, _ = fake.calls[0]
    assert argv[-1] == "/dev/nvme0n1" and argv[:3] == ["/usr/bin/sudo", "-n", "--"]
    assert live.run("smartctl", "nvme0n1 -s on").error == "invalid"


def test_timeouts_and_missing_binaries(live, monkeypatch):
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])
    monkeypatch.setattr(subprocess, "run", boom)
    assert live.run("lscpu").error == "timeout"
    monkeypatch.setattr(sysmod, "resolve_binary", lambda name: None)
    assert LiveSystem().run("lscpu").error == "missing"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="checks real file ownership")
def test_binaries_only_from_trusted_root_owned_dirs(tmp_path, monkeypatch):
    fake = tmp_path / "lscpu"
    fake.write_text("#!/bin/sh\necho pwned\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin")
    found = sysmod.resolve_binary("lscpu")
    assert found is None or not found.startswith(str(tmp_path))
    monkeypatch.setattr(sysmod, "TRUSTED_DIRS", (str(tmp_path),))
    assert sysmod.resolve_binary("lscpu") is None  # not root-owned


# ---------------------------------------------------------------- privacy

def test_identifiers_are_redacted_for_remote_models(laptop):
    box = ToolBox(laptop, redact_identifiers=True)
    overview = box.call("get_system_overview", {"include_serial": True})
    assert overview["data"]["serial_number"] == "[redacted]"
    net = box.call("get_network_info", {"include_mac": True})
    assert all(i.get("mac_address") in (None, "[redacted]") for i in net["data"]["interfaces"])
    assert redact("MAC a4:c3:f0:12:34:56 seen") == "MAC [redacted MAC] seen"


def test_local_sessions_keep_identifiers(laptop):
    box = ToolBox(laptop, redact_identifiers=False)
    assert box.call("get_system_overview", {"include_serial": True})["data"]["serial_number"] == "ACM14G3-77X2Q9"
