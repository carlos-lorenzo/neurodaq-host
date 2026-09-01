# `data/` — recorded datasets

Two incompatible `.npz` schemas live here. They are not versions of each other; one is
continuous, the other is epoched trials.

## Schema A — continuous recordings

Single key `stream_0_Neurodaq`, shape `(n_samples, 8)`, `float64`, microvolts, 250 SPS.
Channel order is CH1..CH8. Produced by the LSL recorders (`plot_record.py`,
`ReceiveAndPlot.py`).

```python
import numpy as np
d = np.load("recordings/shorted.npz")["stream_0_Neurodaq"]   # (3475, 8)
```

## Schema B — epoched trials

Keys `X` shape `(n_trials, 1, 8, 1000)` `float32` and `y` shape `(n_trials,)` — labels
are **unicode strings, not integers**. 1000 samples @ 250 SPS = 4 s epochs (the
`mi_collector.py` record window). The `1` axis is the CNN channel dim for EEGNet.

```python
z = np.load("yo/motor_pristine.npz")
X, y = z["X"], z["y"]        # (60, 1, 8, 1000) float32 ; (60,) <U14
```

> ⚠️ The label sets differ across files. Most are the 3-class motor-imagery set
> (`Relajado`, `Lado Izquierdo`, `Lado Derecho`). But `yo/eyes.npz` is a separate
> **2-class** eyes paradigm (`Abre Ojos`, `Cierra Ojos`). Do **not** `np.concatenate`
> across paradigms.

## Characterization set — `recordings/`

These four schema-A captures are a genuine asset most hobby EEG projects lack. Use them
as references:

| File | Samples | What it is | Use it to… |
|---|---|---|---|
| `shorted.npz` | 3475 | Inputs shorted | Establish your input-referred **noise floor** before trusting any measurement. |
| `test_signal.npz` | 4050 | ADS1299 internal square wave | Verify **gain / scaling**. |
| `open.npz` | 5500 | Eyes open | Alpha-block baseline (low ~10 Hz). |
| `closed.npz` | 2500 | Eyes closed | Alpha-block validation (elevated ~10 Hz). |

See `analysis/frequency_spectrum.ipynb`.

## Catalogue

| File | Schema | Shape | Subject / notes |
|---|---|---|---|
| `dataset_224817.npz`, `dataset_225108.npz`, `noisy_test.npz` | B | `(6,1,8,1000)` | loose at root |
| `recordings/shorted,test_signal,open,closed.npz` | A | `(n,8)` | characterization set |
| `recordings/dataset_191818.npz` | **B** | `(2,1,8,1000)` | **misfiled** among the schema-A recordings |
| `yo/dataset_152626,155619,183625.npz` | B | varies | `yo/` = maintainer |
| `yo/motor_pristine.npz`, `motor_pristine_2.npz` | B | `(60,1,8,1000)` | cleanest MI sets |
| `yo/eyes.npz` | B | `(16,1,8,1000)` | **2-class eyes paradigm** |
| `papi/dataset_102458,103438,104242.npz` | B | up to `(30,…)` | `papi/` = second subject |

Subjects: `yo/` is the maintainer, `papi/` a second subject (informal consent). This is
personal biosignal data — treat accordingly.
