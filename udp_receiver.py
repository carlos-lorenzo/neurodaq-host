import serial
import struct
import time
import threading
import sys
import socket

# Try importing pylsl; if not installed, LSL is gracefully disabled while BrainFlow UDP still works
try:
    from pylsl import StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False


HOST = "0.0.0.0"   # Listen on all network interfaces
PORT = 3333


NUM_CHANNELS = 8

CHANNEL_TO_STREAM = [0]  # , 1, 2, 3, 4, 5, 6, 7
SAMPLE_RATE = 250          # Matches ADS1299_DR_250SPS from your C++ code
VREF = 4.5                       # ADS1299 Internal Reference
GAIN = 24.0                   # Matches ADS1299_PGA_GAIN_24

# Scale factor to convert 24-bit raw counts to Microvolts
SCALE_FACTOR_UV = (2.0 * VREF / GAIN) / (2**24 - 1) * 1000000.0

# Protocol Constants
SYNC_BYTE_0 = 0xAA
SYNC_BYTE_1 = 0x55
# uint16_t length, uint32_t chunk_seq (Little Endian)
HEADER_FORMAT = '<HI'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# The C++ code transmits an array of ads1299_sample_t structs.
# Memory layout of ads1299_sample_t on 32-bit ESP32 (Xtensa architecture):
# Total size: 3 + 1 + 32 + 4 + 8 = 48 bytes.
SAMPLE_SIZE_BYTES = 48
SAMPLE_STRUCT_FORMAT = '<3s x 8i 4x q'


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"Listening on UDP port {PORT}")


def setup_lsl():
    """Configures the Lab Streaming Layer (LSL) outlet if available."""
    if not LSL_AVAILABLE:
        print("pylsl not installed. Skipping LSL initialization...")
        return None

    print(f"Creating LSL Stream: {NUM_CHANNELS} channels at {SAMPLE_RATE} SPS")
    info = StreamInfo(
        name='Neurodaq',
        type='EEG',
        channel_count=len(CHANNEL_TO_STREAM),
        nominal_srate=SAMPLE_RATE,
        channel_format='float32',
        source_id='esp32_ads1299_01'
    )

    chns = info.desc().append_child("channels")
    for i in range(len(CHANNEL_TO_STREAM)):
        ch = chns.append_child("channel")
        ch.append_child_value("label", f"Ch{i+1}")
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")

    outlet = StreamOutlet(info)
    return outlet


def parse_byte(data) -> list[int]:
    sync = 0
    if data[sync] != SYNC_BYTE_0:
        return []

    sync += 1

    if data[sync] != SYNC_BYTE_1:
        return []

    sync += 1

    header_bytes = data[sync:sync+HEADER_SIZE]
    if len(header_bytes) < HEADER_SIZE:
        return []
    sync += HEADER_SIZE

    payload_len, chunk_seq = struct.unpack(HEADER_FORMAT, header_bytes)

    payload = data[sync:]

    sync += payload_len

    num_samples = payload_len // SAMPLE_SIZE_BYTES
    selected_data = []
    for i in range(num_samples):
        offset = i * SAMPLE_SIZE_BYTES
        sample_bytes = payload[offset: offset + SAMPLE_SIZE_BYTES]

        status_bytes, * \
            channels_raw, timestamp_us = struct.unpack(
                SAMPLE_STRUCT_FORMAT, sample_bytes)

        # Convert signed raw counts to microvolts
        channels_data = [
            raw_val * SCALE_FACTOR_UV for raw_val in channels_raw]

        selected_data = [channels_data[ch] for ch in CHANNEL_TO_STREAM]

    return selected_data


lsl_outlet = setup_lsl()

while True:
    data, addr = sock.recvfrom(1024*2)
    selected_data = parse_byte(data)

    if lsl_outlet:
        lsl_outlet.push_sample(selected_data)
