#!/usr/bin/env python
"""
ReceiveAndPlot example for LSL

This example shows data from all found outlets in realtime.
It illustrates the following use cases:
- efficiently pulling data, re-using buffers
- automatically discarding older samples
- online postprocessing
"""

import math
from typing import List, Tuple, TypedDict, cast

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import pylsl

import scipy.signal as signal

# Basic parameters for the plotting window
plot_duration = 1  # how many seconds of data to show
update_interval = 60  # ms between screen updates
pull_interval = 500  # ms between each pull operation
CHANNEL_SPACING_UV = 250.0
TRACE_WIDTH = 1.6
LINE_NOISE_FREQ_HZ = 50.0
BAND_LOW = 2
BAND_HIGH = 150
INVERT_OUTPUT = False


class LineNoiseEstimate(TypedDict):
    channel: int
    tone_rms: float
    overall_rms: float
    relative_db: float


class LineNoiseComparison(TypedDict):
    before: LineNoiseEstimate | None
    after: LineNoiseEstimate | None


class Inlet:
    """Base class to represent a plottable inlet"""

    def __init__(self, info: pylsl.StreamInfo):
        # create an inlet and connect it to the outlet we found earlier.
        # max_buflen is set so data older the plot_duration is discarded
        # automatically and we only pull data new enough to show it

        # Also, perform online clock synchronization so all streams are in the
        # same time domain as the local lsl_clock()
        # (see https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html#_CPPv414proc_clocksync)
        # and dejitter timestamps
        self.inlet = pylsl.StreamInlet(
            info,
            max_buflen=plot_duration,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )
        # store the name and channel count
        self.name = info.name()
        self.channel_count = info.channel_count()

    def pull_and_plot(
        self, plot_time: float, plt: pg.PlotItem
    ) -> LineNoiseComparison | None:
        """Pull data from the inlet and add it to the plot.
        :param plot_time: lowest timestamp that's still visible in the plot
        :param plt: the plot the data should be shown on
        """
        # We don't know what to do with a generic inlet, so we skip it.
        pass


class DataInlet(Inlet):
    """A DataInlet represents an inlet with continuous, multi-channel data that
    should be plotted as multiple lines."""

    dtypes = [[], np.float32, np.float64, None,
              np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, plt: pg.PlotItem, channel_base_index: int = 0):
        super().__init__(info)
        self.channel_base_index = channel_base_index
        # calculate the size for our buffer, i.e. two times the displayed data
        self.max_samples = 2 * math.ceil(info.nominal_srate() * plot_duration)
        bufsize = (self.max_samples, info.channel_count())
        self.input_buffer = np.empty(
            bufsize, dtype=self.dtypes[info.channel_format()])
        self.ts_buffer = np.empty(self.max_samples, dtype=float)
        self.y_buffer = np.empty(
            (self.max_samples, info.channel_count()), dtype=float)
        self.buffer_size = 0
        self.write_index = 0
        empty = np.array([])
        # create one curve object for each channel/line that will handle displaying the data
        self.curves = [
            pg.PlotCurveItem(
                x=empty,
                y=empty,
                autoDownsample=True,
                clipToView=True,
                pen=pg.mkPen(
                    pg.intColor(
                        ch_ix,
                        hues=max(self.channel_count, 1),
                        values=1,
                        maxHue=360,
                        minValue=180,
                        maxValue=255,
                    ),
                    width=TRACE_WIDTH,
                ),
            )
            for ch_ix in range(self.channel_count)
        ]
        for curve in self.curves:
            plt.addItem(curve)

        self.fs = info.nominal_srate()

        # 50 Hz notch
        # iirnotch naturally outputs 2 arrays of length 3 (a single 2nd-order section)
        b_notch, a_notch = signal.iirnotch(LINE_NOISE_FREQ_HZ, 30.0, self.fs)

        # Format the single section into the (1, 6) shape expected by sosfilt
        # shape becomes: [[b0, b1, b2, a0, a1, a2]]
        self.sos_notch = np.hstack((b_notch, a_notch)).reshape(1, 6)

        # Example bandpass: 4-38 Hz
        # Set output="sos" to natively generate the cascade matrix
        self.sos_bp = signal.butter(
            4,
            [BAND_LOW, BAND_HIGH],
            btype="bandpass",
            fs=self.fs,
            output="sos"
        )

        self.notch_state = None
        self.bp_state = None

    def _estimate_line_noise(self, timestamps: np.ndarray, values: np.ndarray):
        sample_count = timestamps.size
        if sample_count < 4:
            return None

        phasor = np.exp(-2j * np.pi * LINE_NOISE_FREQ_HZ * timestamps)
        tiny = np.finfo(float).tiny
        channel_reports = []

        for ch_ix in range(self.channel_count):
            channel_values = values[:, ch_ix].astype(float, copy=False)
            demeaned = channel_values - np.mean(channel_values)
            tone_complex = np.sum(demeaned * phasor)
            tone_amplitude = 2.0 * np.abs(tone_complex) / sample_count
            tone_rms = tone_amplitude / np.sqrt(2.0)
            overall_rms = np.sqrt(np.mean(demeaned ** 2))
            relative_db = 20.0 * np.log10(
                max(tone_rms, tiny) / max(overall_rms, tiny)
            )
            channel_reports.append(
                {
                    "channel": ch_ix + 1,
                    "tone_rms": tone_rms,
                    "overall_rms": overall_rms,
                    "relative_db": relative_db,
                }
            )

        return max(channel_reports, key=lambda report: report["tone_rms"])

    def _append_to_history(self, timestamps: np.ndarray, values: np.ndarray) -> None:
        sample_count = timestamps.size
        if sample_count <= 0:
            return

        if sample_count >= self.max_samples:
            timestamps = timestamps[-self.max_samples:]
            values = values[-self.max_samples:, :]
            sample_count = timestamps.size
            self.ts_buffer[:sample_count] = timestamps
            self.y_buffer[:sample_count, :] = values
            self.buffer_size = sample_count
            self.write_index = sample_count % self.max_samples
            return

        end_index = self.write_index + sample_count
        if end_index <= self.max_samples:
            self.ts_buffer[self.write_index:end_index] = timestamps
            self.y_buffer[self.write_index:end_index, :] = values
        else:
            first_part = self.max_samples - self.write_index
            self.ts_buffer[self.write_index:] = timestamps[:first_part]
            self.y_buffer[self.write_index:] = values[:first_part, :]
            remaining = sample_count - first_part
            self.ts_buffer[:remaining] = timestamps[first_part:]
            self.y_buffer[:remaining, :] = values[first_part:, :]

        self.write_index = (self.write_index + sample_count) % self.max_samples
        self.buffer_size = min(
            self.buffer_size + sample_count, self.max_samples)

    def _ordered_history(self):
        if self.buffer_size == 0:
            return None, None
        if self.buffer_size < self.max_samples:
            return (
                self.ts_buffer[:self.buffer_size],
                self.y_buffer[:self.buffer_size, :],
            )
        if self.write_index == 0:
            return self.ts_buffer, self.y_buffer

        ordered_ts = np.concatenate(
            (self.ts_buffer[self.write_index:],
             self.ts_buffer[: self.write_index])
        )
        ordered_y = np.concatenate(
            (
                self.y_buffer[self.write_index:, :],
                self.y_buffer[: self.write_index, :],
            ),
            axis=0,
        )
        return ordered_ts, ordered_y

    def pull_and_plot(self, plot_time, plt) -> LineNoiseComparison | None:
        # Pull the data
        _, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=self.input_buffer.shape[0], dest_obj=self.input_buffer
        )

        # ts will be empty if no samples were pulled
        if not ts:
            return

        ts = np.asarray(ts)
        y = self.input_buffer[: ts.size, :]
        line_noise_before = self._estimate_line_noise(ts, y)

        # Initialize filter state on first valid chunk using sosfilt_zi
        if self.notch_state is None or self.bp_state is None:
            # self.sos_notch shape is (n_sections, 6)
            # sosfilt_zi expects shape (n_sections, 2) per channel, or (n_sections, 2, n_channels)
            # We broadcast the initial condition across all channels at once
            zi_notch_single = signal.sosfilt_zi(
                self.sos_notch)  # Shape: (n_sections, 2)
            zi_bp_single = signal.sosfilt_zi(
                self.sos_bp)        # Shape: (n_sections, 2)

            # Reshape to (n_sections, 2, channel_count) and scale by initial channel values
            # y[0, :] has shape (channel_count,)
            self.notch_state = zi_notch_single[:, :, np.newaxis] * y[0, :]
            self.bp_state = zi_bp_single[:, :, np.newaxis] * y[0, :]

        # Filter all channels simultaneously (SciPy's sosfilt supports axis parameter)
        # y shape is (samples, channels). We filter along axis=0 (the time axis)
        out_n, self.notch_state = signal.sosfilt(
            self.sos_notch, y, axis=0, zi=self.notch_state
        )
        filtered_y, self.bp_state = signal.sosfilt(
            self.sos_bp, out_n, axis=0, zi=self.bp_state
        )

        self._append_to_history(ts, filtered_y)

        history_x, history_y = self._ordered_history()
        if history_x is None or history_y is None:
            return

        plot_offset = history_x.searchsorted(plot_time)
        if plot_offset >= history_x.size:
            return

        visible_x = history_x[plot_offset:]

        channel_offset = CHANNEL_SPACING_UV

        for ch_ix in range(self.channel_count):
            visible_y = (
                history_y[plot_offset:, ch_ix]
                - (self.channel_base_index + ch_ix) * channel_offset
            )
            if INVERT_OUTPUT:
                visible_y *= -1
            self.curves[ch_ix].setData(visible_x, visible_y)

        line_noise_after = self._estimate_line_noise(ts, filtered_y)
        return {
            "before": line_noise_before,
            "after": line_noise_after,
        }


class MarkerInlet(Inlet):
    """A MarkerInlet shows events that happen sporadically as vertical lines"""

    def __init__(self, info: pylsl.StreamInfo):
        super().__init__(info)
        self.marker_lines = []
        self.max_markers = 200

    def pull_and_plot(self, plot_time, plt):
        # TODO: purge old markers
        strings, timestamps = self.inlet.pull_chunk(0)
        if strings and timestamps:
            for string, ts in zip(strings, timestamps):
                line = pg.InfiniteLine(
                    ts, angle=90, movable=False, label=string[0])
                plt.addItem(line)
                self.marker_lines.append(line)

            while len(self.marker_lines) > self.max_markers:
                old_line = self.marker_lines.pop(0)
                plt.removeItem(old_line)


def main():
    # firstly resolve all streams that could be shown
    inlets: List[Inlet] = []
    print("looking for streams")
    streams = pylsl.resolve_streams()

    # Create the pyqtgraph window
    pg.setConfigOptions(antialias=False)
    pw = cast(pg.PlotWidget, pg.plot(title="LSL Plot"))
    pw.setBackground("#101318")
    plt = cast(pg.PlotItem, pw.getPlotItem())
    plt.setMenuEnabled(False)
    plt.showGrid(x=True, y=True, alpha=0.18)
    plt.setLabel("bottom", "Time", units="s")
    plt.setLabel("left", "Stacked channels", units="µV")
    plt.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    # Scale lock state
    scale_locked = False
    next_channel_base = 0

    # iterate over found streams, creating specialized inlet objects that will
    # handle plotting the data
    for info in streams:
        if info.type() == "Markers":
            if (
                info.nominal_srate() != pylsl.IRREGULAR_RATE
                or info.channel_format() != pylsl.cf_string
            ):
                print("Invalid marker stream " + info.name())
            print("Adding marker inlet: " + info.name())
            inlets.append(MarkerInlet(info))
        elif (
            info.nominal_srate() != pylsl.IRREGULAR_RATE
            and info.channel_format() != pylsl.cf_string
        ):
            print("Adding data inlet: " + info.name())
            inlets.append(
                DataInlet(info, plt, channel_base_index=next_channel_base))
            next_channel_base += info.channel_count()
        else:
            print("Don't know what to do with stream " + info.name())

    print("Press 'L' to toggle scale lock")

    def scroll():
        """Move the view so the data appears to scroll"""
        # We show data only up to a timepoint shortly before the current time
        # so new data doesn't suddenly appear in the middle of the plot
        fudge_factor = pull_interval * 0.002
        plot_time = pylsl.local_clock()
        pw.setXRange(plot_time - plot_duration +
                     fudge_factor, plot_time - fudge_factor)

    def update():
        # Read data from the inlet. Use a timeout of 0.0 so we don't block GUI interaction.
        mintime = pylsl.local_clock() - plot_duration
        # call pull_and_plot for each inlet.
        # Special handling of inlet types (markers, continuous data) is done in
        # the different inlet classes.
        line_noise_reports = []
        for inlet in inlets:
            report = inlet.pull_and_plot(mintime, plt)
            if report is not None:
                line_noise_reports.append((inlet.name, report))

        if line_noise_reports:
            def before_tone_rms(item):
                before_report = item[1]["before"]
                if before_report is None:
                    return 0.0
                return before_report["tone_rms"]

            stream_name, report = max(
                line_noise_reports,
                key=before_tone_rms,
            )
            before_report = report["before"]
            after_report = report["after"]
            if before_report is not None and after_report is not None:
                attenuation_db = before_report["relative_db"] - \
                    after_report["relative_db"]
                plt.setTitle(
                    f"LSL Plot - 50 Hz before: {before_report['tone_rms']:.2f} µV RMS "
                    f"after notch: {after_report['tone_rms']:.2f} µV RMS "
                    f"({attenuation_db:+.1f} dB reduction) on {stream_name} ch {before_report['channel']}"
                )
            elif before_report is not None:
                plt.setTitle(
                    f"LSL Plot - 50 Hz before: {before_report['tone_rms']:.2f} µV RMS "
                    f"after notch: unavailable on {stream_name} ch {before_report['channel']}"
                )
            else:
                plt.setTitle(
                    f"LSL Plot - 50 Hz line noise: waiting for samples on {stream_name}"
                )
        else:
            plt.setTitle(
                "LSL Plot - 50 Hz line noise: waiting for samples"
            )

    def keyPressEvent(ev):
        """Handle keyboard events"""
        nonlocal scale_locked
        if ev.text().lower() == 'l':
            scale_locked = not scale_locked
            plt.getViewBox().enableAutoRange(
                axis=pg.ViewBox.YAxis, enable=not scale_locked
            )
            status = "locked" if scale_locked else "unlocked"
            print(f"Scale {status}")

    pw.keyPressEvent = keyPressEvent

    # create a timer that will move the view every update_interval ms
    update_timer = QtCore.QTimer()
    update_timer.timeout.connect(scroll)
    update_timer.start(update_interval)

    # create a timer that will pull and add new data occasionally
    pull_timer = QtCore.QTimer()
    pull_timer.timeout.connect(update)
    pull_timer.start(pull_interval)

    import sys

    # Start Qt event loop unless running in interactive mode or using pyside.
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        app = QtGui.QGuiApplication.instance()
        if app is not None:
            app.exec_()


if __name__ == "__main__":
    main()
