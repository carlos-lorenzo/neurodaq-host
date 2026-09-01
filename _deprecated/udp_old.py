import socket
import struct
import time

try:
    from pylsl import StreamInfo, StreamOutlet
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False

# --- Configuration ---
HOST = "0.0.0.0"       # Listen on all network interfaces
PORT = 3333

NUM_CHANNELS = 8
CHANNEL_TO_STREAM = [0]
SAMPLE_RATE = 250
VREF = 4.5
GAIN = 24.0

# Scale factor to convert 24-bit raw counts to Microvolts
SCALE_FACTOR_UV = (2.0 * VREF / GAIN) / (2**24 - 1) * 1000000.0

# Protocol Constants
SYNC_BYTE_0 = 0xAA
SYNC_BYTE_1 = 0x55

# C++ telemetry_header_t layout: uint8_t sync[2], uint16_t length, uint32_t chunk_seq
FULL_HEADER_FORMAT = '<BBHI'
HEADER_SIZE = struct.calcsize(FULL_HEADER_FORMAT)  # 8 bytes

SAMPLE_SIZE_BYTES = 48
SAMPLE_STRUCT_FORMAT = '<3s x 8i 4x q'

# Optional: BrainFlow UDP Forwarding (OpenBCI GUI)
FORWARD_TO_BRAINFLOW = True
BRAINFLOW_UDP_IP = "127.0.0.1"
BRAINFLOW_UDP_PORT = 6677


def setup_lsl():
    """Configures the Lab Streaming Layer (LSL) outlet if available."""
    if not LSL_AVAILABLE:
        print("pylsl not installed. Skipping LSL initialization...")
        return None

    print(
        f"Creating LSL Stream: {len(CHANNEL_TO_STREAM)} channel(s) at {SAMPLE_RATE} SPS")
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

    return StreamOutlet(info)


def setup_brainflow_udp():
    """Initializes standard UDP socket for streaming to OpenBCI GUI."""
    print(
        f"Initializing BrainFlow UDP stream on {BRAINFLOW_UDP_IP}:{BRAINFLOW_UDP_PORT}...")
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def parse_udp_packet(data: bytes):
    """
    Validates sync bytes, checksum, and unpacks payload into lists of samples.
    Returns: (lsl_samples, full_channel_samples)
    """
    if len(data) < HEADER_SIZE:
        return [], []

    # 1. Unpack Header
    sync0, sync1, payload_len, chunk_seq = struct.unpack(
        FULL_HEADER_FORMAT, data[:HEADER_SIZE])

    if sync0 != SYNC_BYTE_0 or sync1 != SYNC_BYTE_1:
        return [], []

    # 2. Validate total packet size (Header + Payload + 1 Checksum byte)
    expected_len = HEADER_SIZE + payload_len + 1
    if len(data) < expected_len:
        print(
            f"Truncated UDP packet seq {chunk_seq}: Got {len(data)} B, expected {expected_len} B")
        return [], []

    # 3. Extract Payload and Checksum
    payload = data[HEADER_SIZE: HEADER_SIZE + payload_len]
    expected_checksum = data[HEADER_SIZE + payload_len]

    # 4. Verify Checksum (XOR over payload bytes)
    computed_checksum = 0
    for b in payload:
        computed_checksum ^= b

    if computed_checksum != expected_checksum:
        print(
            f"Checksum mismatch on seq {chunk_seq}. Expected 0x{expected_checksum:02X}, got 0x{computed_checksum:02X}")
        return [], []

    # 5. Parse Payload Samples
    num_samples = payload_len // SAMPLE_SIZE_BYTES
    lsl_samples = []
    full_channel_samples = []

    for i in range(num_samples):
        offset = i * SAMPLE_SIZE_BYTES
        sample_bytes = payload[offset: offset + SAMPLE_SIZE_BYTES]

        status_bytes, *channels_raw, timestamp_us = struct.unpack(
            SAMPLE_STRUCT_FORMAT, sample_bytes
        )

        # Convert 24-bit raw counts to microvolts
        channels_uv = [raw_val * SCALE_FACTOR_UV for raw_val in channels_raw]
        selected_uv = [channels_uv[ch] for ch in CHANNEL_TO_STREAM]

        lsl_samples.append(selected_uv)
        full_channel_samples.append(channels_uv)

    return lsl_samples, full_channel_samples


def main():
    lsl_outlet = setup_lsl()
    bf_sock = setup_brainflow_udp() if FORWARD_TO_BRAINFLOW else None

    # Server socket listening to ESP32 UDP packet stream
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.bind((HOST, PORT))
    print(f"Listening for UDP telemetry on {HOST}:{PORT}...")

    package_counter = 0.0

    try:
        while True:
            data, _ = rx_sock.recvfrom(4096)
            lsl_samples, full_channel_samples = parse_udp_packet(data)

            if not lsl_samples:
                continue

            # Route A: Push chunk of samples to LSL
            if lsl_outlet:
                lsl_outlet.push_chunk(lsl_samples)

            # Route B: Forward to BrainFlow UDP (Cyton layout for OpenBCI GUI)
            if bf_sock:
                for channels_uv in full_channel_samples:
                    brainflow_packet = [0.0] * 24
                    brainflow_packet[0] = package_counter
                    brainflow_packet[1:9] = channels_uv
                    brainflow_packet[22] = time.time()

                    udp_bytes = struct.pack('<24d', *brainflow_packet)
                    bf_sock.sendto(
                        udp_bytes, (BRAINFLOW_UDP_IP, BRAINFLOW_UDP_PORT))
                    package_counter = (package_counter + 1.0) % 256.0

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        rx_sock.close()
        if bf_sock:
            bf_sock.close()


if __name__ == '__main__':
    main()
