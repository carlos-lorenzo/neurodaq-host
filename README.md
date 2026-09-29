# neurodaq-host

Host-side software for [NeuroDAQ](https://github.com/carlos-lorenzo/neurodaq):
receives the EEG stream from the ESP32-S3 over Wi-Fi, visualises it live, and
controls the device over TCP.

## Run

```bash
uv sync
uv run neurodaq          # or: uv run python -m neurodaq_host
```

Connect: default host `192.168.1.53`, port **3334** (TCP control), click
**Connect**, then **START**. Samples arrive as UDP packets on port 3333 and are
re-broadcast as an **LSL** outlet named `NeuroDAQ EEG` (float32, 8ch, µV).

> `neurodaq_gui.py` is the retired PyQt5 monolith — kept for reference only.
> All new work happens in `neurodaq_host/`.

## Architecture — one bytestream, many views

```
firmware --UDP--> io/udp.py --push--> core/hub.py --snapshot--> ui/tabs/*
                    |                      |
                 seq-gap/XOR          events (P300/CV/keys)
                 lost+crc pill         -> unified Session NPZ
```

- **`core/hub.py`** — the single ring buffer of calibrated µV + host/LSL clocks.
  Only `io/udp` writes; every tab reads. Swapping EEG→EMG needs no ingest change.
- **`core/events.py`** — unified marker bus (`lsl_marker` / `manual` / `paradigm`).
  External software syncs via **LSL marker streams** (auto-ingested) and sees our
  markers via the published `NeuroDAQ Markers` outlet.
- **`core/session.py`** — unified Session NPZ: raw `signals`, `fs`, channel
  labels, electrode map, per-sample `t_host`/`t_lsl`, `events`, impedance, filter
  + device meta. `convert_schema_a/b()` read the legacy `data/` files.
- **`core/profiles.py`** — UI profile at `~/.config/neurodaq/profile.json`:
  enabled tabs + order, per-tab panel toggles (e.g. scope waveform/PSD).

## Tabs (all pluggable — `Tabs…` to add/remove)

| Tab | What |
|---|---|
| Oscilloscope | Inline filter bar + stacked traces + optional embedded PSD (each show/hideable) |
| Spectrum | Full-size PSD (same reusable panel as scope) |
| Brain Map | **Owns the 10-20 electrode assignment**, RBF scalp heatmap, region/band bars |
| Device | Rate/SRB, BIAS + sense masks, per-channel vis/PD/gain/**MUX**, reg poke, standby/wakeup |
| Impedance | Lead-off engine config + per-channel Z/tone/flags (no calibration — it was broken) |
| Events | Marker monitor, manual inject, CSV export |
| Paradigm *(off by default)* | Cued-trial runner (prep→cue→record→relax), saves Schema-B epochs to `data/` |
| P300 *(off by default)* | Row/column speller flasher — calibration sessions saved as labeled ERP epochs (`X` n×1×ch×t, `y` target/nontarget + flash meta), ready for classifier training |
| EMG *(off by default)* | Envelope + RMS view — enable + disable Brain for EMG sessions |

Top bar: `lost` (UDP seq gaps since connect) and `crc` (checksum fails) stay at 0
on a clean link; status pills are fixed-width so messages never shift the layout.

## Tests

```bash
uv run pytest tests/ -q   # protocol, DSP/session, offscreen UI smoke
```

## Layout

| Path | What |
|---|---|
| `neurodaq_host/` | The application (see above) |
| `tests/` | pytest suite |
| `neurodaq_gui.py` | Retired monolith (reference only) |
| `_deprecated/` | Legacy single-purpose scripts (see its README) |
| `data/` | Recorded datasets (two legacy `.npz` schemas + new `paradigm_*`/`neurodaq_session_*`) |
| `analysis/` | Notebooks (signal quality, spectra, CNN training) |
| `models/` | EEGNet for motor-imagery experiments |

## Licence

MIT. See [`LICENSE`](LICENSE).
