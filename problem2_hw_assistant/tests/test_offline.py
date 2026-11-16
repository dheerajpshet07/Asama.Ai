from hwqa.backends.offline import OfflineBackend
from hwqa.session import Session


def ask(system, *questions):
    s = Session(system, OfflineBackend())
    return [s.ask(q) for q in questions]


def test_brief_example(laptop):
    (turn,) = ask(laptop, "Can I add more RAM, and how much?")
    assert turn.answer.startswith("You have 16 GB of RAM in 1 of 2 slots")
    assert "The board supports up to 64 GB, so you can add a stick of up to 32 GB in the free slot" in turn.answer
    assert "48 GB total" in turn.answer
    assert turn.unverified == []
    assert turn.tool_calls == [("get_memory_info", {})]


def test_how_much_ram(laptop):
    (turn,) = ask(laptop, "How much RAM do I have?")
    assert "16 GB" in turn.answer and "15.4 GB" in turn.answer and "upgrade" not in turn.answer.lower()


def test_is_ssd_nvme(laptop):
    (turn,) = ask(laptop, "Is my SSD NVMe?")
    assert turn.answer.startswith("Yes.") and "SAMSUNG" in turn.answer


def test_follow_up_uses_conversation(laptop):
    ram, gpu, more = ask(laptop, "How much RAM do I have?", "and what about the GPU?", "how much vram?")
    assert "GeForce MX450" in gpu.answer and gpu.tool_calls[0][0] == "get_gpu_info"
    assert "2 GB VRAM" in more.answer  # short follow-up stays on the GPU topic


def test_says_it_does_not_know_and_why(laptop_noroot):
    (turn,) = ask(laptop_noroot, "Can I upgrade my RAM?")
    assert "I can't determine the installed modules, slot count and maximum supported memory" in turn.answer
    assert "--sudo" in turn.answer
    assert "64 GB" not in turn.answer and "32 GB" not in turn.answer


def test_soldered_answer():
    from hwqa.system import FixtureSystem
    from .conftest import FIXTURES
    (turn,) = ask(FixtureSystem(FIXTURES / "thinkpad_soldered", is_root=True), "can I upgrade my ram?")
    assert "Every upgradeable slot is in use" in turn.answer and "40 GB" in turn.answer
    assert "8 GB of it is soldered" in turn.answer


def test_multiple_topics_in_one_question(laptop):
    (turn,) = ask(laptop, "What CPU and GPU do I have?")
    assert "i5-1135G7" in turn.answer and "Iris Xe" in turn.answer


def test_virtual_disk_is_not_called_hdd(tmp_path):
    import json
    from hwqa.system import FixtureSystem
    (tmp_path / "commands").mkdir()
    (tmp_path / "commands" / "lsblk.out").write_text(json.dumps({"blockdevices": [
        {"name": "sdd", "type": "disk", "size": 1099511627776, "model": "Virtual Disk", "vendor": "Msft",
         "rota": True, "tran": None, "rm": False, "mountpoint": "/", "fstype": "ext4"}]}))
    (tmp_path / "machine.json").write_text('{"kernel": "6.6.36-microsoft-standard-WSL2"}')
    (turn,) = ask(FixtureSystem(tmp_path), "is my ssd nvme?")
    assert turn.answer.startswith("I can't tell from here") and "HDD" not in turn.answer


def test_unknown_question_gets_help(laptop):
    (turn,) = ask(laptop, "write me a poem")
    assert turn.tool_calls == [] and "I can answer questions" in turn.answer
