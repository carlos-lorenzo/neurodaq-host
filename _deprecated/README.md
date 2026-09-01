# `_deprecated/` — the migration playground

These are single-purpose scripts from before `neurodaq_gui.py` existed. The directory
name reflects where they're *headed* — each is being folded into a proper subsystem of
the GUI — but several are still the **only** implementation of a capability the GUI
does not yet have. Don't delete one until its capability has actually landed in the app.

## What still lives only here

| Capability | Only implementation | Target subsystem |
|---|---|---|
| `.npz` recording | `plot_record.py`, `ReceiveAndPlot.py` | recording |
| Packet checksum + dropped-packet detection | `udp_receiver.py` | telemetry ingest |
| Experiment paradigms (motor imagery) | `mi_collector.py`, `data_collector.py` | paradigm runner |
| `standby` / `wakeup` / `read_reg` / `write_reg` control | `tcp_cli.py` | control client |
| Serial transport (pre-Wi-Fi) | `lsl_middleware.py` | (pairs with firmware's stubbed USB JTAG) |
| Eyes-open/closed alpha baseline | `frequency_response.py` | analysis |
| Protocol v0 (`0xAA55`) | `udp_old.py` | retire, not migrate |

## Script catalogue

| Script | Status | Notes |
|---|---|---|
| `udp_receiver.py` | reference | Canonical UDP consumer: magic check, XOR verification, sequence-gap detection. **Start here if you're writing a new consumer.** (Two stale comments — says "824 bytes" and a `Q` sequence field; the packet is 1209 B and the seq is `I`.) |
| `tcp_cli.py` | active | Interactive `questionary` CLI over the TCP JSON control protocol. Reaches `standby`/`wakeup`/`read_reg`/`write_reg`, which the GUI omits — the register-poking tool. Pass `--port 3334`. |
| `mi_collector.py` | active | Motor-imagery cued-trial paradigm (see below). Produced most of `data/`. |
| `plot_record.py` | active | LSL scope **with `.npz` recording** — how you record today, since the GUI can't. Direct ancestor of the GUI. |
| `frequency_response.py` | active | Live PSD with baseline capture — the eyes-open/closed alpha demo. |
| `ReceiveAndPlot.py` | historical | Earlier recorder; produced some of `data/recordings/`. |
| `data_collector.py` | historical | Predecessor of `mi_collector.py`. |
| `lsl_middleware.py` | historical | The only **serial** transport (`/dev/ttyACM0` @ 115200), pre-Wi-Fi. |
| `udp_old.py` | historical | Previous `0xAA55` wire protocol + BrainFlow forwarding. Record of protocol v0. |

## Motor-imagery paradigm (`mi_collector.py`)

3 classes — **Relajado / Lado Izquierdo / Lado Derecho** — 20 trials each. Per-trial
timeline: prep (crosshair) 1 s → cue 1 s → **record 4 s** → relax 2 s. The 4 s record
window at 250 SPS is the 1000-sample epoch length in the schema-B datasets. Raw
(unfiltered) signal is what gets saved.
