# neurodaq-host

The host-side software for [NeuroDAQ](https://github.com/carlos-lorenzo/neurodaq):
receives the EEG stream from the ESP32-S3 over Wi-Fi, visualises it live, and controls
the device over TCP.

> **Status: work in progress.** The project is mid-migration from a collection of
> single-purpose scripts (in [`_deprecated/`](_deprecated/)) into a single application,
> `neurodaq_gui.py`. Several capabilities still live only in those scripts — that's
> what the migration is for. This is a research playground, not a finished product.

## The application — `neurodaq_gui.py`

A PyQt5 + pyqtgraph GUI with four tabs:

1. **Oscilloscope & FFT Spectrum** — 8-channel live scope + spectral view, with DC
   removal, Butterworth bandpass, and notch filtering.
2. **10-20 Brain Map & Band Power** — topographic map plus Delta/Theta/Alpha/Beta/Gamma
   band-power bars.
3. **Channel & Device Settings** — per-channel gain, MUX, power-down, SRB1/2, bias.
4. **Lead-Off Diagnostics & Debug** — per-channel electrode impedance in kΩ, computed
   from an FFT at the lead-off excitation frequency.

Samples are also re-broadcast as an **LSL** outlet named `NeuroDAQ EEG` (float32, 8
channels) for downstream tools.

## Install & run

```bash
uv sync
uv run neurodaq_gui.py
```

Connect: enter the ESP32's IP, set the port to **3334** (the TCP control port — the
field defaults to 3333, which is a known bug), click **Connect TCP**, then **Start**.

The full first-time bring-up (firmware config included) is in
[`neurodaq/docs/GETTING_STARTED.md`](https://github.com/carlos-lorenzo/neurodaq/blob/main/docs/GETTING_STARTED.md).
The wire format is in
[`neurodaq/docs/PROTOCOL.md`](https://github.com/carlos-lorenzo/neurodaq/blob/main/docs/PROTOCOL.md).

## Current GUI limitations

These exist only in the legacy scripts today, which is why the migration matters:

- **No recording.** The GUI does not save to disk — use `_deprecated/plot_record.py`.
- **No checksum validation.** The GUI ignores the packet XOR — `_deprecated/udp_receiver.py`
  verifies it.
- **No dropped-packet detection.** The GUI ignores the sequence number — again,
  `udp_receiver.py` is the reference.

## Repository layout

| Path | What it is |
|---|---|
| `neurodaq_gui.py` | The current application. |
| `_deprecated/` | Single-purpose scripts being migrated into subsystems — see its [README](_deprecated/README.md). Several are still the only implementation of a capability. |
| `data/` | Recorded datasets in two `.npz` schemas — see its [README](data/README.md). |
| `analysis/` | Jupyter notebooks (signal quality, spectra, CNN training). |
| `models/` | EEGNet implementation for the motor-imagery experiments. |

## Licence

MIT. See [`LICENSE`](LICENSE).
