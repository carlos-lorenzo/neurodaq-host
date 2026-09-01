# `analysis/` — notebooks

| Notebook | Purpose |
|---|---|
| `analyze_data.ipynb` | Signal-quality / spike investigation. |
| `frequency_spectrum.ipynb` | Noise floor and spectra, using the `data/recordings/` characterization set. |
| `train_cnn.ipynb` | MNE + ICA preprocessing → EEGNet 3-class motor-imagery training. |

## Known issues

- **Relative data paths are broken.** They moved when files were reorganised into
  `data/recordings/` (commit `7fc5636`); notebooks referencing the old locations need
  their load paths updated.
- **Kernel/venv mismatch.** Notebook metadata pins Python 3.14 while `.venv` is 3.13.
- Trained checkpoints (`*.pth`) are produced but not committed.

Install the extras these need: `uv sync --extra analysis --extra ml`.
