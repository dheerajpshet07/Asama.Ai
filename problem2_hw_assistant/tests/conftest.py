import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"

from hwqa.system import FixtureSystem  # noqa: E402


@pytest.fixture
def laptop():
    """The brief's example machine: 16 GB in 1 of 2 slots, board max 64 GB. Passwordless sudo works."""
    return FixtureSystem(FIXTURES / "laptop_16gb")


@pytest.fixture
def laptop_noroot():
    return FixtureSystem(FIXTURES / "laptop_16gb", sudo=False)


@pytest.fixture
def thinkpad():
    """8 GB soldered + one slot; no root; lspci, lsusb, lsblk not installed."""
    return FixtureSystem(FIXTURES / "thinkpad_soldered")


def read_fixture(*parts) -> str:
    return (FIXTURES.joinpath(*parts)).read_text(encoding="utf-8")
