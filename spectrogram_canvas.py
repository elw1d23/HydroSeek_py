"""
SpectrogramCanvas — reusable PyQt6 widget embedding a single matplotlib
spectrogram inside a FigureCanvasQTAgg.

"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from hydroseek.signal_processing import (
    compute_spectrogram,
    compute_mel_spectrogram,
    format_frequency_ticks,
    get_frequency_axis_label,
)

import matplotlib as mpl
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"]   = 10

_BOX_ALPHA   = 0.20   # fill transparency for bounding boxes
_POINT_SIZE  = 80     # scatter marker area in points^2

# Per-colormap annotation colours chosen for maximum visibility.
# Each entry is (stroke_colour, text_colour).  Stroke is used for the
# marker X, rectangle edge and label text; they are always the same here
# but kept separate in case a future theme needs them to differ.
_CMAP_ANNOTATION_COLOURS: dict[str, tuple[str, str]] = {
    "turbo":    ("#ffffff", "#ffffff"),   # white stands out on the rainbow
    "inferno":  ("#00ffcc", "#00ffcc"),   # cyan-green on dark purple/orange
    "viridis":  ("#ffdd00", "#ffdd00"),   # yellow on blue-green
    "plasma":   ("#00ffcc", "#00ffcc"),   # cyan on magenta/purple
    "gray":     ("#e63946", "#e63946"),   # red on grey is standard
    "managua":  ("#ffffff", "#ffffff"),   # white on this diverging map
}
_DEFAULT_ANNOTATION_COLOUR = ("#ffffff", "#ffffff")  # fallback


def _annotation_colour(colormap: str) -> str:
    """Return the stroke/text colour for annotations on the given colormap."""
    return _CMAP_ANNOTATION_COLOURS.get(colormap, _DEFAULT_ANNOTATION_COLOUR)[0]


class SpectrogramCanvas(FigureCanvasQTAgg):
    """
    Single-spectrogram matplotlib canvas for embedding in PyQt6.

    Annotation extension
    --------------------
    Supports three interaction modes set via set_annotation_mode():
        "none"  — no mouse interaction (default, waveform/large-spec canvases)
        "point" — single click places a point label
        "box"   — click-drag places a bounding-box rectangle

    When the user completes an annotation, annotation_callback is called with
    a dict describing the event.  The LabellingTab wires this up so it can
    write the event into AppState without the canvas knowing about state.

    Stored annotation artists are redrawn after every render() call via
    draw_annotations(), which accepts the list of event dicts for the current
    frame from AppState.
    """

    def __init__(
        self,
        parent=None,
        title: str = "",
        figsize: tuple[float, float] = (4.0, 3.0),
        dpi: int = 100,
        tight: bool = True,
        plot_name: str = "",
        annotation_callback=None,
    ) -> None:
        self._fig = Figure(figsize=figsize, dpi=dpi)
        self._fig.patch.set_facecolor("none")
        super().__init__(self._fig)
        self.setParent(parent)

        self._ax    = self._fig.add_subplot(111)
        self._title = title
        self._tight = tight
        self._img   = None        # AxesImage, kept for update_colormap/clim

        # Annotation state
        self._plot_name          = plot_name          # "Context" | "A" | "B" | "C" | ""
        self._annotation_mode    = "none"             # "none" | "point" | "box"
        self._annotation_callback = annotation_callback  # callable(dict) | None
        self._mpl_cids: list     = []                 # mpl event connection IDs
        self._annotation_artists: list = []           # overlaid artists (persistent)
        self._rubber_rect        = None               # preview rectangle during drag
        self._press_xy           = None               # (xdata, ydata) at button_press
        self._current_colormap   = "turbo"            # kept in sync via set_colormap()
        self._time_offset: float = 0.0               # absolute file time at axis x=0

        self._ax.set_facecolor("#ffffff")

    # ------------------------------------------------------------------
    # Annotation public API
    # ------------------------------------------------------------------

    def set_annotation_mode(self, mode: str) -> None:
        """
        Switch between "none", "point" and "box" interaction modes.

        Disconnects any existing mpl event listeners before reconnecting so
        there is never more than one handler wired at a time.
        """
        self._annotation_mode = mode
        self._disconnect_events()
        self._press_xy = None
        self._remove_rubber_rect()

        if mode in ("point", "box") and self._plot_name:
            self._mpl_cids = [
                self._fig.canvas.mpl_connect("button_press_event",   self._on_press),
                self._fig.canvas.mpl_connect("button_release_event", self._on_release),
                self._fig.canvas.mpl_connect("motion_notify_event",  self._on_motion),
            ]

    def set_colormap(self, colormap: str) -> None:
        """
        Notify the canvas which colormap is currently active so annotation
        overlays can pick a contrasting colour automatically.

        Called by LabellingTab._on_colormap() alongside update_colormap().
        """
        self._current_colormap = colormap

    def set_time_offset(self, offset_seconds: float) -> None:
        """
        Set the absolute file time (in seconds) that corresponds to x=0
        on this canvas's time axis.

        Must be called after every render() on the four annotation canvases
        so that click coordinates from mpl events can be converted to
        absolute file times before being stored in the event table.

        For A / B / C this is s.start_time (frame start in file).
        For the Context plot it is chunk_time_offset + ctx_start_sample / fs.
        """
        self._time_offset = offset_seconds

    def draw_annotations(self, events: list[dict]) -> None:
        """
        Redraw stored annotation artists for the given list of event dicts.

        Call this after every render() so persisted annotations are visible
        when returning to a previously labelled frame.  Clears any existing
        overlay artists first.
        """
        self._clear_annotation_artists()
        colour = _annotation_colour(self._current_colormap)

        for ev in events:
            if ev.get("Plot") != self._plot_name:
                continue
            ev_type = ev.get("Type", "")
            # Stored times are absolute; canvas x-axis is relative to time_offset.
            t0_abs   = ev.get("Event_Time_sec")
            t1_abs   = ev.get("Event_Time_End_sec")
            f_lo = ev.get("Freq_Hz_low")
            f_hi = ev.get("Freq_Hz_high")
            lbl  = ev.get("Event_Label", "")

            t0 = t0_abs - self._time_offset if t0_abs is not None else None
            t1 = t1_abs - self._time_offset if t1_abs is not None else None

            if ev_type == "point" and t0 is not None and f_lo is not None:
                sc = self._ax.scatter(
                    [t0], [f_lo],
                    s=_POINT_SIZE, c=colour,
                    marker="x", linewidths=2, zorder=5,
                )
                tx = self._ax.text(
                    t0, f_lo, f"  {lbl}",
                    color=colour, fontsize=7, va="bottom", zorder=6,
                    clip_on=True,
                )
                self._annotation_artists.extend([sc, tx])

            elif ev_type == "box" and None not in (t0, t1, f_lo, f_hi):
                import math
                if math.isnan(t1) or math.isnan(f_hi):
                    continue
                width  = t1 - t0
                height = f_hi - f_lo
                rect = mpatches.Rectangle(
                    (t0, f_lo), width, height,
                    linewidth=1.5, edgecolor=colour,
                    facecolor=colour, alpha=_BOX_ALPHA, zorder=5,
                )
                self._ax.add_patch(rect)
                tx = self._ax.text(
                    t0, f_hi, f"  {lbl}",
                    color=colour, fontsize=7, va="bottom", zorder=6,
                    clip_on=True,
                )
                self._annotation_artists.extend([rect, tx])

        self.draw_idle()

    def clear_annotation_artists(self) -> None:
        """Remove all overlay annotation artists and refresh."""
        self._clear_annotation_artists()
        self.draw_idle()

    # ------------------------------------------------------------------
    # Mouse event handlers (internal)
    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        if event.inaxes is not self._ax:
            return
        if event.button != 1:   # left-click only
            return
        self._press_xy = (event.xdata, event.ydata)

    def _on_motion(self, event) -> None:
        if self._annotation_mode != "box":
            return
        if self._press_xy is None:
            return
        if event.inaxes is not self._ax:
            return

        x0, y0 = self._press_xy
        x1, y1 = event.xdata, event.ydata

        self._remove_rubber_rect()
        width  = x1 - x0
        height = y1 - y0
        colour = _annotation_colour(self._current_colormap)
        self._rubber_rect = mpatches.Rectangle(
            (min(x0, x1), min(y0, y1)),
            abs(width), abs(height),
            linewidth=1.5, edgecolor=colour,
            facecolor=colour, alpha=_BOX_ALPHA,
            linestyle="--", zorder=6,
        )
        self._ax.add_patch(self._rubber_rect)
        self.draw_idle()

    def _on_release(self, event) -> None:
        if event.button != 1:
            return
        if self._press_xy is None:
            return

        x0_rel, y0 = self._press_xy
        self._press_xy = None
        self._remove_rubber_rect()

        # Guard: release outside axes
        if event.inaxes is not self._ax or event.xdata is None:
            return

        x1_rel, y1 = event.xdata, event.ydata

        # Convert canvas-relative x coordinates to absolute file times.
        x0 = x0_rel + self._time_offset
        x1 = x1_rel + self._time_offset

        if self._annotation_callback is None:
            return

        if self._annotation_mode == "point":
            # Use press coordinates for point labels (more precise than release
            # when the user barely moves the mouse).
            self._annotation_callback({
                "type":  "point",
                "plot":  self._plot_name,
                "x0":    x0,
                "y0":    y0,
                "x1":    None,
                "y1":    None,
            })

        elif self._annotation_mode == "box":
            # Normalise so top-left is always (min, min).
            self._annotation_callback({
                "type":  "box",
                "plot":  self._plot_name,
                "x0":    min(x0, x1),
                "y0":    min(y0, y1),
                "x1":    max(x0, x1),
                "y1":    max(y0, y1),
            })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _disconnect_events(self) -> None:
        for cid in self._mpl_cids:
            self._fig.canvas.mpl_disconnect(cid)
        self._mpl_cids = []

    def _remove_rubber_rect(self) -> None:
        if self._rubber_rect is not None:
            try:
                self._rubber_rect.remove()
            except ValueError:
                pass
            self._rubber_rect = None

    def _clear_annotation_artists(self) -> None:
        for artist in self._annotation_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._annotation_artists = []

    # ------------------------------------------------------------------
    # Public API (spectrogram rendering)
    # ------------------------------------------------------------------

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

        # Axis labels and ticks — use the user-configured f_min/f_max as the
        # tick anchor range, not the raw spectrogram bin values (f_lo/f_hi).
        # f_lo/f_hi are the nearest computed bins to f_min/f_max and are never
        # round numbers (e.g. 31 Hz, 109 Hz), which makes axis labels unreadable.
        self._apply_freq_ticks(ax, f_min, f_max, f_max, log_scale=use_mel)
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

    def _apply_freq_ticks(
        self,
        ax,
        f_lo: float,
        f_hi: float,
        f_max_setting: float,
        n_ticks: int = 8,
        log_scale: bool = False,
    ) -> None:
        """
        Place y-axis ticks and format their labels.

        log_scale=True is passed only for the context plot (mel-scale),
        where the wide frequency range (e.g. 10 Hz – 16 kHz) makes linear
        ticks cluster at the low end.  A/B/C are linear STFT plots and
        always use linear tick spacing regardless of their frequency range.

        When log_scale is True, ticks are placed at round Hz/kHz candidate
        values (1, 2, 5, 10, 20, 50 … per decade) filtered to the visible
        range, matching the MATLAB context-plot behaviour.
        """
        if log_scale and f_lo > 0:
            candidates = []
            for decade_exp in range(-1, 6):   # 0.1 Hz … 100 kHz
                base = 10 ** decade_exp
                for mult in (1, 2, 5):
                    candidates.append(base * mult)

            ticks = [c for c in candidates if f_lo <= c <= f_hi]

            # Fallback: if fewer than 3 candidates fall in range use log-linspace.
            if len(ticks) < 3:
                ticks = np.logspace(
                    np.log10(max(f_lo, 1.0)), np.log10(f_hi), n_ticks
                ).tolist()
        else:
            # Linear ticks at sensible round values, always anchored at f_lo.
            #
            # Strategy:
            #   1. Include f_lo as the first tick so low frequencies are never
            #      skipped (e.g. 20 Hz on plot C, 10 Hz on plot A).
            #   2. Choose a clean step size (1, 2, 2.5, 5, 10 × a power of 10)
            #      that produces roughly 6-8 additional ticks up to f_hi.
            #   3. Generate subsequent ticks at the first round multiple of
            #      that step above f_lo, then every step until f_hi.
            span = f_hi - f_lo
            target_intervals = n_ticks - 1     # gaps between n_ticks ticks
            raw_step = span / target_intervals

            nice_multipliers = [1, 2, 2.5, 5, 10]
            magnitude = 10 ** np.floor(np.log10(raw_step))
            nice_step = min(
                (m * magnitude for m in nice_multipliers),
                key=lambda s: abs(s - raw_step),
            )

            # Build ticks: f_lo first, then round multiples of nice_step above it.
            import math as _math
            first_round = _math.ceil(f_lo / nice_step) * nice_step
            ticks = [f_lo]
            t = first_round
            while t <= f_hi + nice_step * 0.01:
                if t > f_lo:          # avoid duplicating f_lo if it's already a multiple
                    ticks.append(t)
                t += nice_step

            # Safety: always fall back to plain linspace if something went wrong.
            if len(ticks) < 2:
                ticks = np.linspace(f_lo, f_hi, n_ticks).tolist()

        tick_arr = np.array(ticks)
        ax.set_yticks(tick_arr)
        ax.set_yticklabels(format_frequency_ticks(tick_arr), fontsize=10)
