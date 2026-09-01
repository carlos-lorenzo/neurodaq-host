import struct
import socket

from itertools import batched

from pylsl import StreamInfo, StreamOutlet


# Layout Breakdown:
# <   : Little-Endian
# I   : uint32_t (magic header)
# Q   : uint64_t (sequence number)
# 3B8iQ: 3 uint8_t (status bytes) and 8 int32_t (samples) and 1 uint64_t (timestamp)
# B : uint8_t (checksum XOR)
WIRE_PACKET_FORMAT = f"< I I {("3B x 8i 4x Q" * 25)} B"


EXPECTED_SIZE = struct.calcsize(WIRE_PACKET_FORMAT)  # Exactly 824 bytes
SAMPLE_BYTES = 3 + 8 + 1

MAGIC_HEADER = 0x21474545  # EEG! in hex little-endian
N_SAMPLES = 25
N_CHANNELS = 8
SAMPLE_RATE = 250  # Hz
CHANNEL_TO_STREAM = [0, 1]  # Map channels to LSL stream

VREF = 4.5                       # ADS1299 Internal Reference
GAIN = 24.0                   # Matches ADS1299_PGA_GAIN_24

# Scale factor to convert 24-bit raw counts to Microvolts
SCALE_FACTOR_UV = (2.0 * VREF / GAIN) / (2**24 - 1) * 1000000.0

last_packet = -1


def parse_samples(raw_samples):
    """Parses a block of raw samples from the network packet."""
    samples = []
    for sample in batched(raw_samples, SAMPLE_BYTES):
        samples.append(sample[3:-1])

    return samples


def parse_network_packet(raw_bytes) -> tuple[list, int]:
    if len(raw_bytes) != EXPECTED_SIZE:
        return None

    unpacked = struct.unpack(WIRE_PACKET_FORMAT, raw_bytes)

    magic_header = unpacked[0]

    assert magic_header == MAGIC_HEADER, f"Invalid magic header: {magic_header:08X}"

    seq_num = unpacked[1]

    received_crc = unpacked[-1]

    # Eventually check crc

    # (Optional) Verify your data payload against received_crc here...
    # Define tru
    calculated_crc = 0
    for byte in raw_bytes[8:-1]:
        calculated_crc ^= byte

    if calculated_crc != received_crc:
        print(
            f"Checksum mismatch: expected {calculated_crc:02X}, got {received_crc:02X}")

    return parse_samples(unpacked[2:-1]), seq_num


def setup_lsl():
    """Configures the Lab Streaming Layer (LSL) outlet """
    print(f"Creating LSL Stream: {N_CHANNELS} channels at {SAMPLE_RATE} SPS")
    info = StreamInfo(
        name='Neurodaq Raw EEG',
        type='EEG',
        channel_count=len(CHANNEL_TO_STREAM),
        nominal_srate=SAMPLE_RATE,
        channel_format='int32',
        source_id='neurodaq_eeg_stream_raw'
    )

    chns = info.desc().append_child("channels")
    for i in range(len(CHANNEL_TO_STREAM)):
        ch = chns.append_child("channel")
        ch.append_child_value("label", f"Ch{i+1}")
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")

    outlet = StreamOutlet(info)
    return outlet


def push_packet_to_lsl(outlet, samples):
    """Pushes a single packet of EEG samples to the LSL outlet."""
    for sample in samples:
        # Convert raw counts to microvolts
        microvolt_sample = [int(s * SCALE_FACTOR_UV) for s in sample]
        selected_data = [microvolt_sample[ch] for ch in CHANNEL_TO_STREAM]
        outlet.push_sample(selected_data)


IP = "0.0.0.0"
PORT = 3333

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((IP, PORT))

print(f"Listening for UDP packets on {IP}:{PORT}")
print(f"Expected packet size: {EXPECTED_SIZE} bytes")

outlet = setup_lsl()

while True:
    data, addr = sock.recvfrom(2048)
    samples, seq = parse_network_packet(data)
    push_packet_to_lsl(outlet, samples)
    print("Dropped packet\n" if last_packet >
          0 and seq != last_packet + 1 else "", end="")
    last_packet = seq
