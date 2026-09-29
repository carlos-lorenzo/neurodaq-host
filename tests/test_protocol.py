"""Protocol tests: packet build/verify, checksum, size rejection."""
import struct
from neurodaq_host.io import protocol as P


def _packet(seq: int, ch_val: int = 100, corrupt_crc: bool = False) -> bytes:
    payload = b"".join([
        struct.pack("<3B", 0, 0, 0) + b"\x00"
        + struct.pack("<8i", *([ch_val] * 8)) + b"\x00" * 4
        + struct.pack("<Q", 0)
        for _ in range(25)
    ])
    raw = struct.pack("<II", P.MAGIC_HEADER, seq) + payload
    assert len(raw) == P.EXPECTED_SIZE - 1
    crc = 0
    for b in raw[8:]:
        crc ^= b
    if corrupt_crc:
        crc ^= 0xFF
    return raw + struct.pack("<B", crc)


def test_valid_packet():
    ok, crc_ok, seq, unpacked = P.verify_packet(_packet(42))
    assert ok and crc_ok and seq == 42 and unpacked is not None


def test_bad_checksum_flagged():
    ok, crc_ok, seq, _ = P.verify_packet(_packet(7, corrupt_crc=True))
    assert ok and not crc_ok and seq == 7


def test_wrong_size_rejected():
    ok, _, _, _ = P.verify_packet(b"\x00" * 100)
    assert not ok


def test_bad_magic_rejected():
    raw = _packet(1)
    bad = struct.pack("<II", 0xDEADBEEF, 1) + raw[8:]
    ok, crc_ok, _, _ = P.verify_packet(bad)
    assert ok and not crc_ok  # size fine, magic wrong -> no trust
