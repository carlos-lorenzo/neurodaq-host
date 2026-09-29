"""Wire protocol: magic + seq + XOR checksum (canonical, from _deprecated/udp_receiver)."""
import struct

MAGIC_HEADER = 0x21474545
UDP_PORT_DATA = 3333
TCP_PORT_DEFAULT = 3334  # control port (old GUI wrongly defaulted to 3333)
N_CHANNELS = 8
N_SAMPLES_PER_PACKET = 25

WIRE_PACKET_FORMAT = f"< I I {('3B x 8i 4x Q' * 25)} B"
EXPECTED_SIZE = struct.calcsize(WIRE_PACKET_FORMAT)


def verify_packet(raw: bytes) -> tuple[bool, bool, int, tuple | None]:
    """Returns (size_ok, checksum_ok, seq, unpacked). Never raises on bad data."""
    if len(raw) != EXPECTED_SIZE:
        return False, False, -1, None
    try:
        unpacked = struct.unpack(WIRE_PACKET_FORMAT, raw)
    except struct.error:
        return True, False, -1, None
    if unpacked[0] != MAGIC_HEADER:
        return True, False, -1, None
    seq = unpacked[1]
    calc = 0
    for b in raw[8:-1]:
        calc ^= b
    return True, (calc == unpacked[-1]), seq, unpacked
