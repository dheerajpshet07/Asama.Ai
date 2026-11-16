from hwqa.system import FixtureSystem
from hwqa.toolbox import ToolBox
from hwqa.units import GiB

from .conftest import FIXTURES


def call(system, tool, **args):
    res = ToolBox(system).call(tool, args)
    assert res["ok"], res
    return res


# ---------------------------------------------------------------- memory

def test_brief_example_memory_upgrade(laptop):
    d = call(laptop, "get_memory_info")["data"]
    assert d["installed_bytes"] == 16 * GiB
    assert (d["slots_used"], d["slots_total"]) == (1, 2)
    assert d["max_capacity_bytes"] == 64 * GiB
    up = d["upgrade"]
    assert up["largest_module_for_free_slot_bytes"] == 32 * GiB
    assert up["max_total_by_adding_bytes"] == 48 * GiB
    assert up["max_total_by_replacing_bytes"] == 64 * GiB
    assert up["module_type"] == "DDR4" and up["form_factor"] == "SODIMM" and up["speed"] == "3200 MT/s"
    assert any("spec sheet" in n for n in up["notes"])


def test_memory_without_root_says_what_is_missing_and_how_to_fix(laptop_noroot):
    res = call(laptop_noroot, "get_memory_info")
    assert res["data"]["usable_bytes"] == 16131828 * 1024
    assert "slots_total" not in res["data"] and "upgrade" not in res["data"]
    u = res["unavailable"][0]
    assert "slot" in u["what"] and "root" in u["reason"] and "--sudo" in u["fix"]


def test_soldered_memory_upgrade():
    root = FixtureSystem(FIXTURES / "thinkpad_soldered", is_root=True)
    d = call(root, "get_memory_info")["data"]
    assert d["installed_bytes"] == 16 * GiB
    soldered = [s for s in d["slots"] if s["soldered"]]
    assert len(soldered) == 1 and soldered[0]["size_bytes"] == 8 * GiB
    up = d["upgrade"]
    assert up["upgradeable_slots"] == 1 and up["free_slots"] == 0
    # replace the 8 GB SODIMM with a 32 GB one: 8 soldered + 32 = 40 GB
    assert up["max_total_by_replacing_bytes"] == 40 * GiB and up["possible"] is True


def test_all_soldered_cannot_upgrade():
    from hwqa.tools import memory_upgrade
    up = memory_upgrade([{"installed": True, "size_bytes": 16 * GiB, "soldered": True, "type": "LPDDR5"}], 16 * GiB)
    assert up["possible"] is False and up["upgradeable_slots"] == 0


def test_missing_max_capacity_is_not_guessed():
    from hwqa.tools import memory_upgrade
    up = memory_upgrade([{"installed": True, "size_bytes": 8 * GiB, "soldered": False},
                         {"installed": False, "soldered": False}], None)
    assert up["possible"] is None and "max_total_by_adding_bytes" not in up


# ---------------------------------------------------------------- cpu / system

def test_cpu(laptop):
    d = call(laptop, "get_cpu_info")["data"]
    assert d["model"] == "11th Gen Intel Core i5-1135G7 @ 2.40GHz"
    assert (d["physical_cores"], d["threads"], d["max_mhz"], d["base_ghz"]) == (4, 8, 4200, 2.4)
    assert d["hardware_virtualisation"] == "VT-x"
    assert {"AVX2", "AVX-512", "AES-NI"} <= set(d["features"])


def test_cpu_counts_cores_from_cpuinfo(thinkpad):
    d = call(thinkpad, "get_cpu_info")["data"]
    assert (d["physical_cores"], d["threads"], d["max_mhz"]) == (6, 12, 4387)
    assert d["hardware_virtualisation"] == "AMD-V"


def test_system_overview_and_lenovo_naming(laptop, thinkpad):
    d = call(laptop, "get_system_overview")["data"]
    assert d["model"] == "Acme Book 14 G3" and d["form_factor"] == "notebook"
    assert d["os"] == "Ubuntu 22.04.4 LTS" and "serial_number" not in d
    t = call(thinkpad, "get_system_overview")["data"]
    assert t["model"] == "ThinkPad E14 Gen 4" and t["model_code"] == "21EB0041US"


def test_serial_needs_root_when_sysfs_is_protected(laptop_noroot):
    res = call(laptop_noroot, "get_system_overview", include_serial=True)
    assert "serial_number" not in res["data"]
    assert res["unavailable"][0]["what"] == "serial number"


def test_wsl_is_detected(tmp_path):
    (tmp_path / "machine.json").write_text('{"kernel": "6.6.36.3-microsoft-standard-WSL2"}')
    res = call(FixtureSystem(tmp_path), "get_system_overview")
    assert res["data"]["virtualisation"].startswith("WSL2")
    assert any("WSL2" in n for n in res["notes"])


# ---------------------------------------------------------------- storage

def test_storage_lsblk(laptop):
    d = call(laptop, "get_storage_info")["data"]
    assert [x["name"] for x in d["disks"]] == ["nvme0n1"]  # loop devices skipped
    disk = d["disks"][0]
    assert disk["is_nvme"] and disk["media"] == "SSD" and disk["holds_root_filesystem"]
    root = next(p for p in disk["partitions"] if p["mountpoint"] == "/")
    assert root["usage"]["free_bytes"] == 289744478208


def test_storage_falls_back_to_sysfs(thinkpad):
    res = call(thinkpad, "get_storage_info")
    disks = {x["name"]: x for x in res["data"]["disks"]}
    assert set(disks) == {"nvme0n1", "sda"}
    assert disks["nvme0n1"]["is_nvme"] and disks["nvme0n1"]["size"] == "512 GB"
    assert disks["nvme0n1"]["holds_root_filesystem"]
    assert disks["sda"]["removable"] is True
    assert "/sys/block" in res["sources"]


def test_storage_device_filter(laptop):
    d = call(laptop, "get_storage_info", device="sdq")["data"]
    assert d["disks"] == [] and "nvme0n1" in d["error"]


def test_disk_health(laptop, laptop_noroot):
    d = call(laptop, "get_disk_health", device="nvme0n1")["data"]
    assert d["passed"] is True and d["percentage_used"] == 3
    res = call(laptop_noroot, "get_disk_health", device="nvme0n1")
    assert "root" in res["unavailable"][0]["reason"]


def test_disk_health_tool_only_exists_with_smartctl(thinkpad):
    assert "get_disk_health" not in ToolBox(thinkpad).tools


# ---------------------------------------------------------------- gpu / display

def test_gpu_lspci_merged_with_nvidia_smi(laptop):
    gpus = call(laptop, "get_gpu_info")["data"]["gpus"]
    intel, nvidia = gpus
    assert intel["name"] == "Intel Iris Xe Graphics" and intel["kind"] == "integrated" and intel["driver"] == "i915"
    assert nvidia["name"] == "NVIDIA GeForce MX450" and nvidia["kind"] == "discrete"
    assert nvidia["vram_bytes"] == 2048 * 1024 * 1024 and nvidia["driver_version"] == "535.161.07"


def test_gpu_without_lspci_uses_drm_and_pci_ids(thinkpad):
    res = call(thinkpad, "get_gpu_info")
    g = res["data"]["gpus"][0]
    assert g["vendor"].startswith("Advanced Micro Devices") and g["chip"] == "Cezanne"
    assert g["kind"] == "integrated" and g["driver"] == "amdgpu" and g["vram_bytes"] == 512 * 1024 ** 2
    assert any("PCI ID database" in n for n in res["notes"])


def test_display(laptop):
    d = call(laptop, "get_display_info")["data"]
    assert d["connected"] == [{"connector": "eDP-1", "connected": True, "internal": True,
                               "preferred_mode": "1920x1200"}]


# ---------------------------------------------------------------- battery / network / usb / temps

def test_battery_health_and_peripheral_batteries_ignored(laptop):
    d = call(laptop, "get_battery_info")["data"]
    assert len(d["batteries"]) == 1
    b = d["batteries"][0]
    assert (b["charge_percent"], b["health_percent"], b["cycle_count"]) == (87, 79, 312)
    assert d["ac_adapters"] == [{"name": "AC", "type": "Mains", "plugged_in": False}]


def test_battery_charge_counters_and_zero_cycles(thinkpad):
    b = call(thinkpad, "get_battery_info")["data"]["batteries"][0]
    assert b["health_percent"] == 91 and "cycle_count" not in b


def test_network(laptop, thinkpad):
    d = call(laptop, "get_network_info")["data"]
    kinds = {i["name"]: i["kind"] for i in d["interfaces"]}
    assert kinds == {"wlp0s20f3": "wifi", "docker0": "virtual"}  # loopback skipped
    assert d["controllers"][0]["model"] == "Wi-Fi 6 AX201" and d["bluetooth"] is True
    assert "mac_address" not in d["interfaces"][0]
    t = call(thinkpad, "get_network_info")
    assert t["data"]["interfaces"][0]["driver"] == "rtw89_8852ae"
    assert any("lspci" in n for n in t["notes"])


def test_usb(laptop, thinkpad):
    names = [d["name"] for d in call(laptop, "get_usb_devices")["data"]["devices"]]
    assert "Logitech, Inc. Unifying Receiver" in names and not any("root hub" in n for n in names)
    assert call(thinkpad, "get_usb_devices")["data"]["devices"] == [
        {"vendor_id": "0781", "product_id": "5583", "name": "SanDisk Ultra Fit"}]


def test_temperatures(laptop):
    sensors = call(laptop, "get_temperatures")["data"]["sensors"]
    assert {"sensor": "nvme Composite", "celsius": 38.9} in sensors


def test_capabilities(thinkpad):
    d = call(thinkpad, "get_capabilities")["data"]
    assert d["commands"]["lspci"].startswith("missing") and d["commands"]["dmidecode"] == "installed"
    assert "no root" in d["privilege"]


# ---------------------------------------------------------------- caching

def test_static_results_are_cached_live_ones_expire(laptop):
    box = ToolBox(laptop)
    box.call("get_cpu_info")
    assert box.call("get_cpu_info")["cached"] is True
    box.tools["get_battery_info"].ttl = 0
    box.call("get_battery_info")
    assert "cached" not in box.call("get_battery_info")


def test_tool_crash_does_not_kill_session(laptop, monkeypatch):
    from hwqa import tools
    box = ToolBox(laptop)
    monkeypatch.setattr(box.tools["get_cpu_info"], "fn", lambda s: 1 / 0)
    res = box.call("get_cpu_info")
    assert res["ok"] is False and "ZeroDivisionError" in res["error"]
    assert tools  # module still importable / unchanged
