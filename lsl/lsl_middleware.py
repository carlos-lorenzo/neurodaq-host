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

# --- Configuration ---
# Change this to your ESP32's COM port (e.g., 'COM3' on Windows)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200            # Matches standard ESP32 USB JTAG/Serial
NUM_CHANNELS = 8
CHANNEL_TO_STREAM = [0]
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

# --- BrainFlow Streaming Board Configurations ---
# OpenBCI GUI's "Streaming (from external)" expects a BrainFlow stream.
# For Board ID 0 (Cyton), BrainFlow expects 24 double-precision values per packet:
# Row 0: Package Number
# Rows 1-8: EEG Channels
# Rows 9-11: Accelerometer Channels (X, Y, Z)
# Rows 12-18: Other Channels
# Rows 19-21: Analog Channels
# Row 22: Timestamp (double-precision UNIX epoch)
# Row 23: Marker Channel
UDP_IP = "127.0.0.1"
UDP_PORT = 6677


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


def setup_brainflow_udp():
    """Initializes standard UDP socket for streaming to the OpenBCI GUI."""
    print(f"Initializing BrainFlow UDP stream on {UDP_IP}:{UDP_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock


def command_thread(ser):
    """Background thread to forward terminal commands to the ESP32."""
    print("Command thread active. Type 'b' to start, 's' to stop stream on ESP32.")
    try:
        while True:
            cmd = sys.stdin.readline()
            if cmd:
                ser.write(cmd.encode('ascii'))
                ser.flush()
    except Exception as e:
        print(f"Command thread exiting: {e}")


def main():
    # Initialize LSL Outlet and BrainFlow UDP Sockets
    lsl_outlet = setup_lsl()
    udp_sock = setup_brainflow_udp()

    # Initialize Serial Connection
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to ESP32 on {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"Failed to connect to serial port: {e}")
        return

    # Start terminal command passthrough
    # cmd_thread_inst = threading.Thread(
    #     target=command_thread, args=(ser,), daemon=True)
    # cmd_thread_inst.start()

    print("Listening for telemetry stream and forwarding LSL to localhost...")

    sync_state = 0
    package_counter = 0.0

    try:
        while True:
            # 1. Sync State Machine: Look for 0xAA 0x55
            byte = ser.read(1)
            if not byte:
                continue

            if sync_state == 0 and byte[0] == SYNC_BYTE_0:
                sync_state = 1
                continue
            elif sync_state == 1:
                if byte[0] == SYNC_BYTE_1:
                    sync_state = 2
                else:
                    sync_state = 0

            if sync_state != 2:
                continue

            # 2. Read Header (Length and Sequence)
            header_bytes = ser.read(HEADER_SIZE)
            if len(header_bytes) < HEADER_SIZE:
                sync_state = 0
                continue

            payload_len, chunk_seq = struct.unpack(HEADER_FORMAT, header_bytes)

            # 3. Read Payload
            payload = ser.read(payload_len)
            if len(payload) < payload_len:
                sync_state = 0
                continue

            # 4. Read and Verify Checksum
            checksum_byte = ser.read(1)
            if len(checksum_byte) < 1:
                sync_state = 0
                continue

            expected_checksum = checksum_byte[0]
            computed_checksum = 0
            for b in payload:
                computed_checksum ^= b

            if computed_checksum != expected_checksum:
                print(
                    f"Checksum mismatch on seq {chunk_seq}. Expected {expected_checksum:02X}, got {computed_checksum:02X}")
                sync_state = 0
                continue

            # 5. Parse Payload into Samples
            if payload_len % SAMPLE_SIZE_BYTES != 0:
                print(
                    f"Warning: Payload length {payload_len} is not a multiple of {SAMPLE_SIZE_BYTES}")

            num_samples = payload_len // SAMPLE_SIZE_BYTES

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

                # --- Stream Route A: Lab Streaming Layer ---
                if lsl_outlet:
                    lsl_outlet.push_sample(selected_data)

                # --- Stream Route B: BrainFlow UDP (OpenBCI GUI) ---
                # Build a 24-element list of f64 (doubles) representing standard Cyton data rows
                brainflow_packet = [0.0] * 24

                # Assign fields to match BrainFlow's Cyton layout
                # Row 0: Packet Index
                brainflow_packet[0] = package_counter
                # Rows 1-8: EEG Channels (uV)
                brainflow_packet[1:9] = channels_data
                # Rows 9-11: Accel (leave as 0.0)
                # Rows 12-18: Other (leave as 0.0)
                # Rows 19-21: Analog (leave as 0.0)
                # Row 22: Host UNIX Timestamp
                brainflow_packet[22] = time.time()
                # Row 23: Marker (leave as 0.0)

                # Pack as 24 little-endian doubles (192 bytes total)
                udp_packet_bytes = struct.pack('<24d', *brainflow_packet)
                udp_sock.sendto(udp_packet_bytes, (UDP_IP, UDP_PORT))

                # Increment rolling package counter (limits to 0-255 like a real Cyton)
                package_counter = (package_counter + 1.0) % 256.0

            # Reset sync state for next packet
            sync_state = 0

    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        ser.close()
        udp_sock.close()


if __name__ == '__main__':
    main()
