from hwqa.grounding import unverified_claims
from hwqa.toolbox import ToolBox

RESULT = {"data": {"installed_bytes": 17179869184, "max_mhz": 4200, "slots_total": 2, "used_percent": 37.2,
                   "sensors": [{"celsius": 38.85}]}}


def test_numbers_from_tools_pass():
    text = "You have 16 GB in 1 of 2 slots, boosting to 4.2 GHz. 37% used; NVMe at 39°C."
    assert unverified_claims(text, [RESULT]) == []


def test_invented_numbers_are_flagged():
    assert unverified_claims("It supports up to 128 GB and runs at 5.1 GHz.", [RESULT]) == ["128 GB", "5.1 GHz"]


def test_numbers_without_units_are_ignored():
    assert unverified_claims("It's the 2023 model, gen 11.", [RESULT]) == []


def test_against_real_tool_output(laptop):
    box = ToolBox(laptop)
    box.call("get_memory_info")
    ok = "The board supports up to 64 GB, so you can add a stick of up to 32 GB (48 GB total)."
    assert unverified_claims(ok, box.cached_results()) == []
    assert unverified_claims("You could go to 96 GB.", box.cached_results()) == ["96 GB"]
