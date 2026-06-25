"""
SpectrogramCanvas — reusable PyQt6 widget embedding a single matplotlib
spectrogram inside a FigureCanvasQTAgg.

"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from hydroseek.signal_processing import (
    compute_spectrogram,
    compute_mel_spectrogram,
    format_frequency_ticks,
    get_frequency_axis_label,
)

import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"]   = 10


class SpectrogramCanvas(FigureCanvasQTAgg):
    """Single-spectrogram matplotlib canvas for embedding in PyQt6."""

    def __init__(
        self,
        parent=None,
        title: str = "",
        figsize: tuple[float, float] = (4.0, 3.0),
        dpi: int = 100,
        tight: bool = True,
    ) -> None:
        self._fig = Figure(figsize=figsize, dpi=dpi)
        self._fig.patch.set_facecolor("none")
        super().__init__(self._fig)
        self.setParent(parent)

        self._ax    = self._fig.add_subplot(111)
        self._title = title
        self._tight = tight
        self._img   = None        # AxesImage, kept for update_colormap/clim

        self._ax.set_facecolor("#ffffff")

    # Public API


    def render(
        self,
        signal: np.ndarray,
        fs: int,
        window_size: int,
        overlap_samples: int,
        f_min: float,
        f_max: float,
        dr_low: float,
        dr_high: float,
        colormap: str = "inferno",
        use_mel: bool = False,
        marker_times: list[float] | None = None,
        hide_y_labels: bool = False,
    ) -> None:
        """
        Compute and display a spectrogram.

        Colour scaling
        dr_high and dr_low are treated as offsets (in dB) relative to the
        99th-percentile peak of the computed spectrogram.  This matches MATLAB
        where the audio is normalised to ±1 before display so the dynamic range
        values make intuitive sense regardless of absolute signal level.

            vmax = peak_db + dr_high      (e.g. peak - 10 + 10  = peak)
            vmin = peak_db + dr_low       (e.g. peak - 10 + -80 = peak - 90)

        marker_times

        Optional list of time values (seconds) at which to draw vertical black
        lines — used to mark the current frame window on the context plot and
        the frame start position on the large overview spectrogram.
        """
        ax = self._ax
        ax.cla()
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_facecolor("#ffffff")

        if use_mel:
            S_db, freqs, times = compute_mel_spectrogram(
                signal, fs, window_size, overlap_samples
            )
        else:
            S_db, freqs, times = compute_spectrogram(
                signal, fs, window_size, overlap_samples
            )

        # Clip to frequency band of interest
        freq_mask = (freqs >= f_min) & (freqs <= f_max)
        if freq_mask.sum() < 2:
            freq_mask = np.ones(len(freqs), dtype=bool)

        S_plot    = S_db[freq_mask, :]
        freq_plot = freqs[freq_mask]

        # Auto-scale: treat dr_low / dr_high as offsets from the data peak
        finite_vals = S_plot[np.isfinite(S_plot)]
        if len(finite_vals) > 0:
            peak_db = float(np.percentile(finite_vals, 99))
        else:
            peak_db = 0.0

        vmax = peak_db + dr_high   # e.g. peak + 10
        vmin = peak_db + dr_low    # e.g. peak - 80

        t_min = float(times[0])  if len(times) > 0 else 0.0
        t_max = float(times[-1]) if len(times) > 0 else 1.0
        f_lo  = float(freq_plot[0])
        f_hi  = float(freq_plot[-1])

        self._img = ax.imshow(
            S_plot,
            aspect="auto",
            origin="lower",
            extent=[t_min, t_max, f_lo, f_hi],
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_ylim(f_lo, f_hi)
        ax.set_xlim(t_min, t_max)

        # Vertical marker lines (frame boundaries / position indicator)
        if marker_times:
            for mt in marker_times:
                if t_min <= mt <= t_max:
                    ax.axvline(x=mt, color="black", linewidth=1.5,
                               linestyle="-", alpha=0.85)

        # Axis labels and ticks
        self._apply_freq_ticks(ax, f_lo, f_hi, f_max)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(get_frequency_axis_label(f_max), fontsize=8)
        ax.set_title(self._title, fontsize=10, pad=2, color='#3a3a3a')
        ax.tick_params(labelsize=10)

        # The large overview spectrogram is too short (< 100 px) for y-axis
        # tick labels to be readable.  When hide_y_labels=True, suppress them
        # entirely so the extra horizontal space goes to the signal image.
        # All other canvases (CP, A, B, C) call render() without this flag
        # so they are completely unaffected.
        if hide_y_labels:
            ax.set_yticks([])
            ax.set_ylabel("")

        if self._tight:
            self._fig.tight_layout(pad=0.15)

        self.draw_idle()

    def render_waveform(
        self,
        signal: np.ndarray,
        fs: int,
        marker_time: float | None = None,
    ) -> None:
        """
        Display a waveform overview with an optional position marker.

        """
        ax = self._ax
        ax.cla()

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_facecolor("#ffffff")

        duration = len(signal) / fs
        times    = np.linspace(0.0, duration, len(signal))

        ax.plot(times, signal, color="#3a3a3a", linewidth=0.25, rasterized=True)

        peak_val = float(np.percentile(np.abs(signal), 99))
        if peak_val == 0.0:
            peak_val = 1.0
        ax.set_ylim(-peak_val * 1.5, peak_val * 1.5)
        ax.set_xlim(0.0, duration)

        if marker_time is not None:
            ax.axvline(x=marker_time, color="black", linewidth=2.0,
                       linestyle="-", alpha=0.85)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(self._title, fontsize=8, pad=2)

        if self._tight:
            self._fig.tight_layout(pad=0.3)

        self.draw_idle()

    def clear(self) -> None:
        self._ax.cla()
        self._ax.set_facecolor("#ffffff")
        self.draw_idle()

    def update_colormap(self, colormap: str) -> None:
        if self._img is not None:
            self._img.set_cmap(colormap)
            self.draw_idle()

    def update_clim(self, dr_low: float, dr_high: float) -> None:
        if self._img is not None:
            self._img.set_clim(dr_low, dr_high)
            self.draw_idle()

    # Internal helpers


    def _apply_freq_ticks(
        self,
        ax,
        f_lo: float,
        f_hi: float,
        f_max_setting: float,
        n_ticks: int = 8,
    ) -> None:
        tick_positions = np.linspace(f_lo, f_hi, n_ticks)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(format_frequency_ticks(tick_positions), fontsize=10)
