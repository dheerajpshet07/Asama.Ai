import json

from hwqa import parsers
from hwqa.units import GiB, fmt_disk, fmt_ram, parse_size

from .conftest import read_fixture


def test_units():
    assert parse_size("16 GB") == 16 * GiB
    assert parse_size("16384 MB") == 16 * GiB          # older dmidecode
    assert parse_size("No Module Installed") is None
    assert fmt_ram(16 * GiB) == "16 GB"
    assert fmt_ram(16518991872) == "15.4 GB"
    assert fmt_disk(512110190592) == "512 GB"
    assert fmt_disk(1_000_204_886_016) == "1 TB"


def test_lscpu_new_indented_format():
    info = parsers.parse_lscpu(read_fixture("laptop_16gb", "commands", "lscpu.out"))
    assert info["Model name"].startswith("11th Gen Intel")
    assert info["CPU(s)"] == "8"
    assert info["Core(s) per socket"] == "4"
    assert info["L3"] == "8 MiB (1 instance)"
    assert "Caches (sum of all)" not in info  # section headers have no value


def test_lscpu_old_flat_format():
    info = parsers.parse_lscpu(read_fixture("thinkpad_soldered", "commands", "lscpu.out"))
    assert info["Model name"] == "AMD Ryzen 5 5625U with Radeon Graphics"
    assert info["L2 cache"] == "3 MiB"


def test_lscpu_keeps_repeated_keys():
    info = parsers.parse_lscpu("Model name: Cortex-A55\nModel name: Cortex-A76\n")
    assert info["Model name"] == "Cortex-A55"
    assert info["Model name[]"] == ["Cortex-A55", "Cortex-A76"]


def test_meminfo_and_free():
    mi = parsers.parse_meminfo(read_fixture("laptop_16gb", "fs", "proc", "meminfo"))
    assert mi["MemTotal"] == 16131828 * 1024
    f = parsers.parse_free(read_fixture("laptop_16gb", "commands", "free.out"))
    assert f["mem_total"] == 16518991872 and f["swap_total"] == 2147479552


def test_dmidecode_memory():
    recs = parsers.parse_dmidecode(read_fixture("laptop_16gb", "commands", "dmidecode_memory.out"))
    arrays = [r for r in recs if r.type == 16]
    devices = [r for r in recs if r.type == 17]
    assert arrays[0].props["Maximum Capacity"] == "64 GB"
    assert [d.props["Size"] for d in devices] == ["16 GB", "No Module Installed"]
    assert devices[0].title == "Memory Device"


def test_dmidecode_list_values():
    text = "Handle 0x0000, DMI type 0, 26 bytes\nBIOS Information\n\tVendor: X\n\tCharacteristics:\n\t\tPCI is supported\n\t\tUSB legacy is supported\n"
    rec = parsers.parse_dmidecode(text)[0]
    assert rec.props["Characteristics"] == ["PCI is supported", "USB legacy is supported"]


def test_lsblk_new_style_json():
    devs = parsers.parse_lsblk_json(read_fixture("laptop_16gb", "commands", "lsblk.out"))
    nvme = next(d for d in devs if d["name"] == "nvme0n1")
    assert nvme["rotational"] is False and nvme["transport"] == "nvme" and nvme["size"] == 512110190592
    assert [c["mountpoint"] for c in nvme["children"]] == ["/boot/efi", "/"]


def test_lsblk_old_string_json_and_mountpoints_list():
    old = {"blockdevices": [{"name": "sda", "type": "disk", "size": "500107862016", "rota": "1", "rm": "0",
                             "tran": "sata", "model": "WDC WD5000LPVX ", "mountpoints": [None, "/data"]}]}
    d = parsers.parse_lsblk_json(json.dumps(old))[0]
    assert d["size"] == 500107862016 and d["rotational"] is True and d["removable"] is False
    assert d["model"] == "WDC WD5000LPVX" and d["mountpoint"] == "/data"


def test_lspci_machine_readable():
    devs = parsers.parse_lspci_vmm(read_fixture("laptop_16gb", "commands", "lspci.out"))
    vga = next(d for d in devs if d["class_id"] == "0300")
    assert vga["vendor"] == "Intel Corporation" and vga["vendor_id"] == "8086"
    assert vga["device"] == "TigerLake-LP GT2 [Iris Xe Graphics]" and vga["device_id"] == "9a49"
    assert vga["driver"] == "i915"
    wifi = next(d for d in devs if d["class_id"] == "0280")
    assert wifi["device"] == "Wi-Fi 6 AX201"


def test_lsusb_and_nvidia_smi():
    usb = parsers.parse_lsusb(read_fixture("laptop_16gb", "commands", "lsusb.out"))
    assert {"vendor_id": "046d", "product_id": "c52b"}.items() <= usb[3].items()
    gpus = parsers.parse_nvidia_smi("NVIDIA GeForce RTX 4050 Laptop GPU, 6141, 617.42, 00000000:01:00.0\n")
    assert gpus == [{"name": "NVIDIA GeForce RTX 4050 Laptop GPU", "memory_mib": 6141,
                     "driver_version": "617.42", "bus_id": "00000000:01:00.0"}]


def test_pci_ids_lookup():
    db = read_fixture("thinkpad_soldered", "fs", "usr", "share", "hwdata", "pci.ids")
    assert parsers.lookup_pci_ids(db, "1002", "1638") == (
        "Advanced Micro Devices, Inc. [AMD/ATI]", "Cezanne [Radeon Vega Series / Radeon Vega Mobile Series]")
    assert parsers.lookup_pci_ids(db, "10de", "ffff") == ("NVIDIA Corporation", None)
    assert parsers.lookup_pci_ids(db, "dead") == (None, None)


def test_os_release_and_smartctl():
    assert parsers.parse_os_release('PRETTY_NAME="Ubuntu 22.04.4 LTS"\nID=ubuntu\n')["PRETTY_NAME"] == "Ubuntu 22.04.4 LTS"
    s = parsers.parse_smartctl_json(read_fixture("laptop_16gb", "commands", "smartctl.out"))
    assert s["passed"] is True and s["percentage_used"] == 3 and s["power_on_hours"] == 2789
