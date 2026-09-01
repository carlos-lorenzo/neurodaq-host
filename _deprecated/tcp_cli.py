import socket
import json
import argparse
import re
import sys
import questionary


def send_and_receive(sock, req_id, cmd, params=None):
    request = {"id": req_id, "cmd": cmd}
    if params:
        request["params"] = params

    sock.sendall((json.dumps(request) + "\n").encode('utf-8'))

    buf = ""
    while True:
        char = sock.recv(1).decode('utf-8')
        if not char:
            raise ConnectionError("Server closed connection")
        buf += char
        if char == '\n':
            break

    return json.loads(buf)


def main():
    parser = argparse.ArgumentParser(description="NeuroDAQ TCP Control CLI")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Target IP address")
    parser.add_argument("--port", type=int, required=True,
                        help="Target TCP port")
    parser.add_argument("--verify", action="store_true",
                        help="Auto-verify register writes")
    args = parser.parse_args()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.host, args.port))
        print(f"Connected to {args.host}:{args.port}")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    req_id = 1
    commands = [
        "start", "stop", "reset", "standby", "wakeup",
        "read_reg", "write_reg", "config_global",
        "config_leadoff", "config_bias", "config_channel",
        "exit"
    ]

    while True:
        cmd = questionary.select(
            "Select command to send:",
            choices=commands,
            style=questionary.Style([('selected', 'fg:green bold')])
        ).ask()

        if cmd == "exit" or cmd is None:
            break

        params = {}

        # Gather dynamic parameters based on command
        if cmd in ["read_reg", "write_reg"]:
            addr_str = questionary.text("Register address (0-255):").ask()
            params["address"] = int(addr_str) if addr_str.isdigit() else 0

            if cmd == "write_reg":
                val_str = questionary.text("Value to write (0-255):").ask()
                params["value"] = int(val_str) if val_str.isdigit() else 0

        elif cmd.startswith("config_"):
            print(
                "Sending default mock parameters for config command. Edit script to customize.")
            if cmd == "config_global":
                params = {"sample_rate": 6,
                          "srb1_enabled": True, "srb2_enabled": False}
            elif cmd == "config_bias":
                params = {"bias_p": True, "bias_n": True,
                          "sensp": 255, "sensn": 255}
            elif cmd == "config_channel":
                params = {"channel": 1, "power_down": 0, "gain": 6, "mux": 0}
            elif cmd == "config_leadoff":
                params = {"enabled": True, "threshold": 4, "current": 2,
                          "frequency": 1, "sensp": 255, "sensn": 255, "flip": 0}

        # Send Primary Command
        print(f"\n[->] TX: {cmd} | Params: {params}")
        resp = send_and_receive(sock, req_id, cmd, params)
        print(f"[<-] RX: {resp}")
        req_id += 1

        # Verification Logic for write_reg
        if cmd == "write_reg" and args.verify and resp.get("success"):
            print(f"[*] Verifying write to register {params['address']}...")
            verify_resp = send_and_receive(sock, req_id, "read_reg", {
                                           "address": params["address"]})
            req_id += 1

            if verify_resp.get("success"):
                # Parse "Reg[0x<addr>] = <value>" from the C++ backend message
                match = re.search(r"=\s*(\d+)", verify_resp.get("message", ""))
                if match:
                    read_val = int(match.group(1))
                    if read_val == params["value"]:
                        print(
                            f"    [+] VERIFIED: Register reads back {read_val}")
                    else:
                        print(
                            f"    [-] MISMATCH: Wrote {params['value']}, but read {read_val}")
                else:
                    print(
                        "    [!] Could not parse register value from response message.")
            else:
                print(
                    f"    [!] Read verification failed: {verify_resp.get('message')}")
        print("-" * 40)

    sock.close()
    print("Connection closed.")


if __name__ == "__main__":
    print("Running")
    main()
