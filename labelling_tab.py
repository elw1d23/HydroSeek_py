"""
LabellingTab — PyQt6 widget for HydroSeek

Layout (top to bottom):
    Overview strip  — Waveform + large spectrogram, pinned to a short fixed
                      height so they act as a navigation guide without
                      dominating the screen.
    Context Plot    — Full width, medium height.
    Spectrogram row — A | B | C side by side; takes the majority of height.
    Controls panel  — Three columns:
                        left   : dynamic checkboxes (only user-defined labels)
                        centre : progress, confidence, comments
                        right  : counter (prominent), Listen, Next, Previous,
                                 colormap picker

"""

import math

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QCheckBox, QPlainTextEdit,
    QButtonGroup, QRadioButton, QGroupBox,
    QFrame, QSizePolicy, QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette, QColor

from hydroseek.state import AppState
from spectrogram_canvas import SpectrogramCanvas
from hydroseek.labelling import append_row, remove_last_row, fix_chunk_numbering, export_labels
from hydroseek.event_labelling import (
    create_event_labels_table,
    append_event,
    remove_last_event,
    get_events_for_frame,
    export_event_labels,
    export_event_config,
)
from hydroseek.audio import pad_chunk, resample_audio
from hydroseek.signal_processing import overlap_percent_to_samples



# Visual constants — change these to retheme the whole tab in one place.
# Mimics the matlab opening code, so set everythng now, can be user specific. 

# primary action button.
_ACCENT     = "#2a7fcf"
_ACCENT_HOV = "#1a6fbf"

# Muted text colour used for secondary labels (e.g. "CHUNK" / "FRAME" caps).
_MUTED = "#3a3a3a"

# Border colour for the thin QFrame separator and group-box outlines.
_BORDER = "#3a3a3a"

# A stylesheet applied to the entire tab widget so every QGroupBox, QLabel,
# QPushButton, QCheckBox and QPlainTextEdit inherits a consistent look
# without needing per-widget setStyleSheet calls.
#
_TAB_STYLESHEET = """
    /* ---- Base widget background / text ---- */
    QWidget {
        background-color: #ffffff;
        color: #3a3a3a;
        font-family: "Arial";
        font-size: 16px;
    }

    /* ---- Group boxes ---- */
    /* QGroupBox draws its own frame and title.  We style both here. */
    QGroupBox {
        border: 1px solid #3a3a3a;
        border-radius: 12px;
        margin-top: 18px;       /* leaves room for the title label above the border */
        padding: 6px;
        font-size: 16px;
        font-weight: bold;
        color: #aaaaaa;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 16px;
        padding: 0 4px;
        color: #3a3a3a;
    }

    /* ---- Plain buttons ---- */
    QPushButton {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #4a4a4a;
        border-radius: 4px;
        padding: 3px 8px;
    }
    QPushButton:hover  { background-color: #383838; }
    QPushButton:pressed { background-color: #222222; }
    QPushButton:disabled { color: #666666; border-color: #333333; }

    /* ---- Checkboxes ---- */
    QCheckBox { spacing: 6px; }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #5a5a5a;
        border-radius: 3px;
        background-color: #2d2d2d;
    }
    QCheckBox::indicator:checked {
        background-color: #2a7fcf;
        border-color: #2a7fcf;
    }
    QCheckBox::indicator:hover { border-color: #888888; }

    /* ---- Radio buttons ---- */
    QRadioButton { spacing: 5px; }
    QRadioButton::indicator {
        width: 13px;
        height: 13px;
        border: 1px solid #5a5a5a;
        border-radius: 7px;
        background-color: #2d2d2d;
    }
    QRadioButton::indicator:checked {
        background-color: #2a7fcf;
        border-color: #2a7fcf;
    }

    /* ---- Plain text edit (comments box) ---- */
    QPlainTextEdit {
        background-color: #ffffff;
        border: 1px solid #4a4a4a;
        border-radius: 3px;
        color: #3a3a3a;
        selection-background-color: #2a7fcf;
    }

    /* ---- Thin horizontal separator line ---- */
    QFrame[frameShape="4"] {   /* 4 = HLine */
        color: #cccccc;
        background-color: #cccccc;
    }
"""

COLORMAP_OPTIONS = [
    ("turbo",    "Turbo"),
    ("inferno",  "Inferno"),
    ("viridis",  "Viridis"),
    ("gray",     "Grayscale"),
    ("plasma",   "Plasma"),
    ("managua",   "Managua"),
]


class LabellingTab(QWidget):
    """Main labelling interface."""

    def __init__(self, state: AppState, main_window) -> None:
        super().__init__()
        self._state  = state
        self._mw     = main_window
        self._counter_value: float = float("nan")
        self._big_chunk_idx: int = 0

        # These are populated in on_session_start() / _rebuild_checkbox_panel(),
        # not at __init__ time, because label names come from AppState which
        # isn't fully set until Start is pressed.
        self._checkboxes: list[QCheckBox] = []
        self._active_label_count: int = 0   # how many non-NA labels were built

        self._cmap_buttons: dict[str, QPushButton] = {}

        # Apply the dark stylesheet to this widget and all its children.
        self.setStyleSheet(_TAB_STYLESHEET)

        self._build_ui()

    
    # Public entry point

    def on_session_start(self) -> None:
        """
        Called by MainWindow.switch_to_labelling() after audio is loaded.
        Rebuilds the dynamic checkbox panel, resets navigation state,
        and renders the first frame.
        """
        s = self._state

        s.current_chunk        = s.audio_chunks[0]
        s.downsampled          = resample_audio(s.current_chunk, s.fs, s.target_fs)
        s.current_chunk_index  = 0
        s.current_frame_number = 0
        s.chunk_number         = int(len(s.current_chunk) // s.frame_size)
        s.num_chunks           = s.chunk_number

        self._big_chunk_idx = 0

        # Rebuild checkboxes to match the labels the user actually entered.
        # This must happen before _render_overview/_advance_to_next_frame
        # because _reset_controls() iterates self._checkboxes.
        self._rebuild_checkbox_panel()

        # Sync annotation canvas colormaps to the current state default so
        # overlay colours are correct from the very first frame.
        for canvas in (self._cp_canvas, self._sa_canvas, self._sb_canvas, self._sc_canvas):
            canvas.set_colormap(self._state.colormap)

        self._render_overview()
        self._advance_to_next_frame()

    # UI construction


    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Overview strip — waveform on top, large spectrogram below it,
        # both full-width, stacked in a single container widget that is
        # pinned to a fixed total height.
        #
        # Height split: waveform gets 32 px, large spectrogram fills the
        # rest (~92 px).  A thin 1 px QFrame(HLine) sits between them.

        overview = QWidget()
        overview.setFixedHeight(125)        # total strip height, never grows
        overview_lay = QVBoxLayout(overview)
        overview_lay.setContentsMargins(0, 0, 0, 0)
        overview_lay.setSpacing(0)

        # widget controls the Qt layout allocation; matplotlib scales to fit.
        self._waveform_canvas = SpectrogramCanvas(
            parent=self, title="", figsize=(14, 0.7), dpi=80, tight=True
        )
        self._waveform_canvas.setFixedHeight(32)
        overview_lay.addWidget(self._waveform_canvas)

        # 1 px horizontal divider between waveform and spectrogram.
        #hdiv = QFrame()
        #hdiv.setFrameShape(QFrame.Shape.HLine)
        #hdiv.setFrameShadow(QFrame.Shadow.Plain)
        #hdiv.setFixedHeight(1)
        #overview_lay.addWidget(hdiv)

        # Large spectrogram: takes the remaining height of the strip.
        self._large_spec_canvas = SpectrogramCanvas(
            parent=self, title="", figsize=(14, 1.2), dpi=80, tight=True
        )
        overview_lay.addWidget(self._large_spec_canvas)   # fills remainder

        root.addWidget(overview)

        # Thin separator between overview strip and context plot.
        # A 1 px QFrame line is far lighter visually than a QGroupBox border.
        #sep = QFrame()
        #sep.setFrameShape(QFrame.Shape.HLine)
        #sep.setFrameShadow(QFrame.Shadow.Plain)
        #sep.setFixedHeight(1)
        #root.addWidget(sep)

        # Context Plot — full width, medium height.

        self._cp_canvas = SpectrogramCanvas(
            parent=self, title="Context", figsize=(14, 3.0), dpi=90, tight=True,
            plot_name="Context",
            annotation_callback=self._on_annotation_committed,
        )
        self._cp_canvas.setMinimumHeight(200)
        root.addWidget(self._cp_canvas, stretch=3)

      
        # Spectrogram A / B / C row — the primary analysis surface.
        # stretch=3 gives this the majority of remaining vertical space.
        
        spec_row = QHBoxLayout()
        spec_row.setSpacing(4)
        self._sa_canvas = SpectrogramCanvas(
            parent=self, title="A", figsize=(5, 3.5), dpi=90,
            plot_name="A",
            annotation_callback=self._on_annotation_committed,
        )
        self._sb_canvas = SpectrogramCanvas(
            parent=self, title="B", figsize=(5, 3.5), dpi=90,
            plot_name="B",
            annotation_callback=self._on_annotation_committed,
        )
        self._sc_canvas = SpectrogramCanvas(
            parent=self, title="C", figsize=(5, 3.5), dpi=90,
            plot_name="C",
            annotation_callback=self._on_annotation_committed,
        )
        for canvas in (self._sa_canvas, self._sb_canvas, self._sc_canvas):
            canvas.setMinimumHeight(220)
            spec_row.addWidget(canvas, stretch=1)
        root.addLayout(spec_row, stretch=3)



        # Controls panel — three columns inside a plain QWidget.
       
        controls = self._build_controls_panel()
        # Fixed height: controls never grow/shrink as the window resizes.
        # 185 px fits all elements comfortably on a 1024 px tall monitor.
        controls.setFixedHeight(185)

        controls_wrapper = QHBoxLayout()
        controls_wrapper.setContentsMargins(60, 0, 60, 0)
        controls_wrapper.addWidget(controls)
        root.addLayout(controls_wrapper)

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        outer = QHBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # Placeholder container for checkboxes — populated dynamically in
        # _rebuild_checkbox_panel() when a session starts.
        self._checkbox_container = QGroupBox("Select labels")
        self._checkbox_container.setLayout(QGridLayout())
        # stretch=3 gives checkboxes the most width; they can have many labels.
        outer.addWidget(self._checkbox_container, stretch=3)

        outer.addWidget(self._build_centre_panel(),      stretch=2)
        outer.addWidget(self._build_event_annot_panel(), stretch=2)
        outer.addWidget(self._build_right_panel(),       stretch=2)

        return panel

    def _build_event_annot_panel(self) -> QGroupBox:
        """
        Event Annotation tools. Conatined for easy editing

        Layout (compact):
          MODE label + None / Point / Box radio buttons (one row) - user selects the event type before adding label.
          Event Label: text entry, stays input so that you can just draw multiple event boxes in one go. 
          [X]  Events labelled: N   (clear button + count on one row) - press clear button to undo last event label (continuously works)
          User gets a config.csv saved with the event labels file for reconstructing boundng boxes with the same fft settings.
        """
        grp = QGroupBox("Event Annotations")
        vlay = QVBoxLayout(grp)
        vlay.setSpacing(5)
        vlay.setContentsMargins(6, 16, 6, 6)

        # Mode label
        mode_lbl = QLabel("MODE")
        mode_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_MUTED};")
        vlay.addWidget(mode_lbl)

        # Mode radio buttons on one row
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self._annot_mode_group = QButtonGroup(self)
        self._annot_mode_btns: dict[str, QRadioButton] = {}
        for key, display in (("none", "None"), ("point", "Point"), ("box", "Box")):
            rb = QRadioButton(display)
            if key == "none":
                rb.setChecked(True)
            self._annot_mode_group.addButton(rb)
            self._annot_mode_btns[key] = rb
            mode_row.addWidget(rb)
        mode_row.addStretch(1)
        vlay.addLayout(mode_row)

        self._annot_mode_btns["none"].toggled.connect(
            lambda checked: self._on_annot_mode_changed("none") if checked else None
        )
        self._annot_mode_btns["point"].toggled.connect(
            lambda checked: self._on_annot_mode_changed("point") if checked else None
        )
        self._annot_mode_btns["box"].toggled.connect(
            lambda checked: self._on_annot_mode_changed("box") if checked else None
        )

        # Event label free-text entry
        from PyQt6.QtWidgets import QLineEdit
        label_row = QHBoxLayout()
        label_lbl = QLabel("Event Label:")
        label_lbl.setStyleSheet(f"color: {_MUTED};")
        label_row.addWidget(label_lbl)

        self._event_label_edit = QLineEdit()
        self._event_label_edit.setFixedHeight(28)
        self._event_label_edit.setPlaceholderText("e.g. click, whistle...")
        self._event_label_edit.setStyleSheet(
            "QLineEdit { background-color: #ffffff; border: 1px solid #4a4a4a; "
            "border-radius: 3px; color: #3a3a3a; padding: 2px 4px; font-size: 14px; }"
            "QLineEdit:focus { border-color: #2a7fcf; }"
        )
        label_row.addWidget(self._event_label_edit)
        vlay.addLayout(label_row)

        # Event confidence
        event_conf_row = QHBoxLayout()
        event_conf_lbl = QLabel("Confidence:")
        event_conf_lbl.setStyleSheet(f"color: {_MUTED};")
        event_conf_row.addWidget(event_conf_lbl)
        self._event_conf_group = QButtonGroup(self)
        self._event_conf_btns: list[QRadioButton] = []
        for val in (1, 2, 3):
            rb = QRadioButton(str(val))
            if val == 1:
                rb.setChecked(True)
            self._event_conf_group.addButton(rb, val)
            event_conf_row.addWidget(rb)
            self._event_conf_btns.append(rb)
        event_conf_row.addStretch(1)
        vlay.addLayout(event_conf_row)

        # Clear button (X) and event count on one compact row
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)

        self._clear_last_event_btn = QPushButton("X")
        self._clear_last_event_btn.setFixedSize(28, 28)
        self._clear_last_event_btn.setToolTip("Clear last placed event annotation")
        self._clear_last_event_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; "
            "border: 1px solid #4a4a4a; border-radius: 4px; "
            "background: #2d2d2d; color: #e0e0e0; }"
            "QPushButton:hover { background: #383838; }"
            "QPushButton:pressed { background: #222222; }"
        )
        self._clear_last_event_btn.clicked.connect(self._on_clear_last_event)
        bottom_row.addWidget(self._clear_last_event_btn)

        count_lbl = QLabel("Events labelled:")
        count_lbl.setStyleSheet(f"color: {_MUTED}; font-size: 13px;")
        self._event_count_display = QLabel("0")
        self._event_count_display.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #3a3a3a;"
        )
        bottom_row.addWidget(count_lbl)
        bottom_row.addWidget(self._event_count_display)
        bottom_row.addStretch(1)
        vlay.addLayout(bottom_row)

        vlay.addStretch(1)
        return grp

    # Dynamic checkbox panel
    # adds the checkboxes that the labeller set rather than just havign all 18 like in the matlab veriosn


    def _rebuild_checkbox_panel(self) -> None:
        """
        Build (or rebuild) the checkbox grid from the currently active labels.
        
        The user may change label names and press Start again.  
        Rebuilding from scratch is safer than trying to diff the old list.

        Any entry in state.labels that is non-empty AND does not start with
        "NA_".  

        IMPORTANT — checkbox_values contract:
        apppend_row expects 1 value per label column in labels_table. So its only created with the number of active labels
        set by the user as labels column. The checkboxes are in the same order and count, no padding used. 
        """
        # Remove all existing widgets from the container's layout.
        layout = self._checkbox_container.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._checkboxes = []

        # Cast to QGridLayout so we can call addWidget with row/col args.
        grid = self._checkbox_container.layout()
        assert isinstance(grid, QGridLayout)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(3)
        grid.setContentsMargins(6, 14, 6, 6)

        # Collect only the labels the user actually named.
        active_labels = [
            (i, lbl)
            for i, lbl in enumerate(self._state.labels)
            if lbl and not lbl.startswith("NA_")
        ]
        self._active_label_count = len(active_labels)

        if not active_labels:
            # Edge case: user left all labels blank.  Show one placeholder.
            placeholder = QLabel("No labels configured — go to Set Up tab.")
            placeholder.setStyleSheet(f"color: {_MUTED};")
            grid.addWidget(placeholder, 0, 0)
            return

        # Arrange into two columns of up to 9 rows each.
        # If the user has <= 9 labels, only the left column is populated.
        # The layout naturally collapses to a single column when n <= 9.
        col_size = math.ceil(len(active_labels) / 2) if len(active_labels) > 9 else 9
        # With <= 9 labels ust use a single column so they don't look
        # stretched across an empty right side.
        n_cols = 1 if len(active_labels) <= 9 else 2

        small_font = QFont()
        small_font.setPointSize(12)

        for seq, (_, lbl_text) in enumerate(active_labels):
            # QCheckBox with the label text built in — one widget per label.
            
            cb = QCheckBox(lbl_text)
            cb.setFont(small_font)
 

            col = seq // col_size
            row = seq % col_size
            grid.addWidget(cb, row, col)
            self._checkboxes.append(cb)

    def _build_centre_panel(self) -> QGroupBox:
        grp = QGroupBox("Annotations")
        vlay = QVBoxLayout(grp)
        vlay.setSpacing(5)
        vlay.setContentsMargins(6, 16, 6, 6)

       
        # Counter
        #
        # The value is shown in a large bold label flanked by +/- buttons.
        # Starting invisible 
        counter_cap = QLabel("COUNT")
        counter_cap.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {_MUTED};")
        vlay.addWidget(counter_cap)

        counter_outer = QHBoxLayout()
        counter_outer.setSpacing(4)

        self._counter_minus_btn = QPushButton("-")
        self._counter_minus_btn.setFixedSize(32, 32)
        self._counter_minus_btn.setStyleSheet(
            "QPushButton { font-size: 16px; font-weight: bold; "
            "border: 1px solid #4a4a4a; border-radius: 4px; background: #2d2d2d; color: #e0e0e0; }"
            "QPushButton:hover { background: #383838; color: #e0e0e0; }"
        )
        self._counter_minus_btn.clicked.connect(self._on_counter_decrement)

        self._counter_display = QLabel("-")
        self._counter_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._counter_display.setMinimumWidth(60)
        self._counter_display.setStyleSheet(
            "font-size: 30px; font-weight: bold; color: #3a3a3a;"
            "font-family: 'Courier New', monospace;"
        )
        #self._counter_display.setVisible(False)

        self._counter_plus_btn = QPushButton("+")
        self._counter_plus_btn.setFixedSize(32, 32)
        self._counter_plus_btn.setStyleSheet(
            "QPushButton { font-size: 16px; font-weight: bold; "
            "border: 1px solid #4a4a4a; border-radius: 4px; background: #2d2d2d; color: #e0e0e0; }"
            "QPushButton:hover { background: #383838; color: #e0e0e0; }"
        )
        self._counter_plus_btn.clicked.connect(self._on_counter_increment)

        counter_outer.addWidget(self._counter_minus_btn)
        counter_outer.addWidget(self._counter_display, stretch=1)
        counter_outer.addWidget(self._counter_plus_btn)
        vlay.addLayout(counter_outer)

        # Thin separator between counter and annotation fields.
        #sep = QFrame()
        #sep.setFrameShape(QFrame.Shape.HLine)
        #sep.setFrameShadow(QFrame.Shadow.Plain)
        #vlay.addWidget(sep)

        # Confidence radio buttons
        conf_row = QHBoxLayout()
        conf_lbl = QLabel("Confidence:")
        conf_lbl.setStyleSheet(f"color: {_MUTED};")
        conf_row.addWidget(conf_lbl)
        self._conf_group = QButtonGroup(self)
        self._conf_btns: list[QRadioButton] = []
        for val in (1, 2, 3):
            rb = QRadioButton(str(val))
            if val == 1:
                rb.setChecked(True)
            self._conf_group.addButton(rb, val)
            conf_row.addWidget(rb)
            self._conf_btns.append(rb)
        conf_row.addStretch(1)
        vlay.addLayout(conf_row)

        # Comments box
        
        comments_row = QHBoxLayout()
        comments_lbl = QLabel("Notes:")
        comments_lbl.setStyleSheet(f"color: {_MUTED};")
        comments_row.addWidget(comments_lbl)
        self._comments_edit = QPlainTextEdit()
        self._comments_edit.setFixedHeight(40)
        self._comments_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._comments_edit.setPlaceholderText("Optional annotation...")
        comments_row.addWidget(self._comments_edit)
        vlay.addLayout(comments_row)

        vlay.addStretch(1)
        return grp

    def _build_right_panel(self) -> QGroupBox:
        """
        Right column: Chunk/Frame progress display, Listen, Next Frame,
        Previous, colormap picker.


        """
        grp = QGroupBox("Navigation")
        vlay = QVBoxLayout(grp)
        vlay.setSpacing(5)
        vlay.setContentsMargins(6, 16, 6, 6)

       
        progress_row = QHBoxLayout()
        progress_row.setSpacing(16)

        for attr, caption in (("_chunk_display", "CHUNK"), ("_frame_display", "FRAME")):
            col = QVBoxLayout()
            col.setSpacing(0)
            cap = QLabel(caption)
            cap.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {_MUTED};")
            val = QLabel("- of -")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet("font-size: 16px; font-weight: bold;")
            col.addWidget(cap)
            col.addWidget(val)
            setattr(self, attr, val)
            progress_row.addLayout(col)

        #progress_row.addStretch(1)
        vlay.addLayout(progress_row)

        # Thin separator before the action buttons.
        #sep = QFrame()
        #sep.setFrameShape(QFrame.Shape.HLine)
        #sep.setFrameShadow(QFrame.Shadow.Plain)
        #vlay.addWidget(sep)

        # Action buttons row: Listen | Next Frame | Previous

        action_row = QHBoxLayout()
        action_row.setSpacing(4)

        self._listen_btn = QPushButton("Listen")
        self._listen_btn.setFixedHeight(28)
        self._listen_btn.clicked.connect(self._on_listen)
        action_row.addWidget(self._listen_btn)

        # "Load Next Frame" is the primary action
        self._next_btn = QPushButton("Next Frame")
        self._next_btn.setFixedHeight(28)
        bold_font = QFont()
        bold_font.setBold(True)
        bold_font.setPointSize(9)
        self._next_btn.setFont(bold_font)
        self._next_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_ACCENT}; color: white; "
            f"border: none; border-radius: 4px; }} "
            f"QPushButton:hover {{ background-color: {_ACCENT_HOV}; }} "
            f"QPushButton:disabled {{ background-color: #444; color: #888; }}"
        )
        self._next_btn.clicked.connect(self._on_next_frame)
        action_row.addWidget(self._next_btn, stretch=1)

        self._prev_btn = QPushButton("Previous")
        self._prev_btn.setFixedHeight(28)
        self._prev_btn.clicked.connect(self._on_previous)
        action_row.addWidget(self._prev_btn)

        vlay.addLayout(action_row)

   
        cmap_cap = QLabel("COLORMAP")
        cmap_cap.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {_MUTED};")
        vlay.addWidget(cmap_cap)

        cmap_grid = QGridLayout()
        cmap_grid.setSpacing(3)
        for i, (cmap, label) in enumerate(COLORMAP_OPTIONS):
            btn = QPushButton(label)
            btn.setFixedHeight(26)
            btn.setCheckable(True)
            btn.setStyleSheet(
                f"QPushButton {{ font-size: 11px; padding: 1px 3px; "
                f"border: 1px solid #4a4a4a; border-radius: 3px; }} "
                f"QPushButton:checked {{ background-color: {_ACCENT}; "
                f"color: white; border-color: transparent; }}"
            )
            btn.clicked.connect(lambda checked=False, cm=cmap: self._on_colormap(cm))
            self._cmap_buttons[cmap] = btn
            cmap_grid.addWidget(btn, i // 3, i % 3)  # row 0: buttons 0,1,2 — row 1: buttons 3,4,5

        # Mark the default colormap as active.
        default_cmap = self._state.colormap
        if default_cmap in self._cmap_buttons:
            self._cmap_buttons[default_cmap].setChecked(True)

        vlay.addLayout(cmap_grid)
        vlay.addStretch(1)
        return grp


    # Event annotation callbacks

    def _on_annot_mode_changed(self, mode: str) -> None:
        """Propagate mode change to all four annotation canvases."""
        self._state.annotation_mode = mode
        for canvas in (self._cp_canvas, self._sa_canvas, self._sb_canvas, self._sc_canvas):
            canvas.set_annotation_mode(mode)

    def _on_annotation_committed(self, data: dict) -> None:
        """
        Called by a SpectrogramCanvas when the user completes a point click
        or finishes dragging a box.

        Reads current frame context from AppState, increments the event ID
        counter, appends the row to event_labels_table, redraws overlays on
        all canvases, and updates the event count display.
        """
        s = self._state
        if s.event_labels_table is None:
            return

        label   = self._event_label_edit.text().strip()
        event_conf = self._event_conf_group.checkedId()
        comment = self._comments_edit.toPlainText().strip()

        s.event_id_counter += 1

        ev_type = data["type"]
        x0 = data["x0"]
        y0 = data["y0"]
        x1 = data.get("x1")
        y1 = data.get("y1")

        s.event_labels_table = append_event(
            table          = s.event_labels_table,
            event_id       = s.event_id_counter,
            event_type     = ev_type,
            plot           = data["plot"],
            chunk_idx  = self._big_chunk_idx,
            frame_no       = s.current_chunk_index,
            frame_start    = s.start_time,
            frame_end      = s.end_time,
            event_time     = x0,
            event_time_end = x1,
            freq_low       = y0,
            freq_high      = y1,
            label          = label,
            confidence     = event_conf,
            comment        = comment,
        )

        self._refresh_annotation_overlays()
        self._update_event_count_display()

    def _on_clear_last_event(self) -> None:
        """Remove the most recently placed event and refresh overlays."""
        s = self._state
        if s.event_labels_table is None or len(s.event_labels_table) == 0:
            return
        s.event_id_counter = max(0, s.event_id_counter - 1)
        s.event_labels_table = remove_last_event(s.event_labels_table)
        self._refresh_annotation_overlays()
        self._update_event_count_display()

    def _refresh_annotation_overlays(self) -> None:
        """
        Redraw stored event annotations on all four labelling canvases for
        the current frame.  Called after every append, clear, or frame change.
        """
        s = self._state
        if s.event_labels_table is None:
            return
        events = get_events_for_frame(
            s.event_labels_table, self._big_chunk_idx, s.current_chunk_index
        )
        for canvas in (self._cp_canvas, self._sa_canvas, self._sb_canvas, self._sc_canvas):
            canvas.draw_annotations(events)

    def _update_event_count_display(self) -> None:
        """Update the events-this-frame counter label."""
        s = self._state
        if s.event_labels_table is None:
            self._event_count_display.setText("0")
            return
        events = get_events_for_frame(
            s.event_labels_table, self._big_chunk_idx, s.current_chunk_index
        )
        self._event_count_display.setText(str(len(events)))

    # ALL Button callbacks

    def _on_next_frame(self) -> None:
        s = self._state

        # Reconstruct label checkbox list.
        #
        checkbox_values = [1 if cb.isChecked() else 0 for cb in self._checkboxes]

        confidence = self._conf_group.checkedId()
        comment    = self._comments_edit.toPlainText().strip()
        count      = self._counter_value

        s.labels_table = append_row(
            s.labels_table,
            chunk_no        = s.current_chunk_index,
            start_time      = s.start_time,
            end_time        = s.end_time,
            checkbox_values = checkbox_values,
            confidence      = confidence,
            comment         = comment,
            count           = count,
        )

        # End of this big chunk?
        if s.current_chunk_index >= s.num_chunks:
            self._big_chunk_idx += 1

            if self._big_chunk_idx >= len(s.audio_chunks):
                self._finish_labelling()
                return

            s.current_chunk       = s.audio_chunks[self._big_chunk_idx]
            s.downsampled         = resample_audio(s.current_chunk, s.fs, s.target_fs)
            s.current_chunk_index = 0
            s.chunk_number        = int(len(s.current_chunk) // s.frame_size)
            s.num_chunks          = s.chunk_number

            self._render_overview()

        self._advance_to_next_frame()

    def _on_previous(self) -> None:
        s = self._state

        if s.current_chunk_index <= 1:
            QMessageBox.information(self, "HydroSeek", "Already at the first frame.")
            return

        s.labels_table = remove_last_row(s.labels_table)
        s.current_chunk_index -= 1

        self._set_frame_indices(s.current_chunk_index)
        self._reset_controls()
        self._render_spectrograms()
        self._update_progress_display()

    def _on_counter_increment(self) -> None:
        if math.isnan(self._counter_value):
            self._counter_value = 0
        self._counter_value += 1
        self._counter_display.setText(str(int(self._counter_value)))
        #self._counter_display.setVisible(True)

    def _on_counter_decrement(self) -> None:
        if math.isnan(self._counter_value):
            self._counter_value = 0
        self._counter_value -= 1
        self._counter_display.setText(str(int(self._counter_value)))
        #self._counter_display.setVisible(True)

    def _on_listen(self) -> None:
        s = self._state
        if s.chunk is None:
            return
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(s.chunk, samplerate=s.fs)
        except Exception as exc:
            QMessageBox.warning(self, "Playback Error", str(exc))

    def _on_colormap(self, cmap: str) -> None:
        """Switch colormap on all canvases and update button highlight."""
        self._state.colormap = cmap
        for key, btn in self._cmap_buttons.items():
            btn.setChecked(key == cmap)
        for canvas in (
            self._cp_canvas, self._sa_canvas,
            self._sb_canvas, self._sc_canvas,
            self._large_spec_canvas,
        ):
            canvas.update_colormap(cmap)
        # Keep annotation canvases aware of the new colormap so overlay
        # colours update to remain visible against the new palette.
        for canvas in (self._cp_canvas, self._sa_canvas, self._sb_canvas, self._sc_canvas):
            canvas.set_colormap(cmap)
        # Refresh overlays immediately so existing annotations redraw in the new colour.
        self._refresh_annotation_overlays()


    # Frame rendering


    def _advance_to_next_frame(self) -> None:
        s = self._state
        s.current_chunk_index  += 1
        s.current_frame_number += 1

        self._set_frame_indices(s.current_chunk_index)
        self._reset_controls()
        self._render_spectrograms()
        self._update_progress_display()

    def _set_frame_indices(self, frame_idx: int) -> None:
        s = self._state

        s.start_index = (frame_idx - 1) * s.frame_size
        is_final_frame = (
            frame_idx == s.num_chunks
            and self._big_chunk_idx == len(s.audio_chunks) - 1
        )
        if is_final_frame:
            s.end_index = len(s.current_chunk) - 1
        else:
            s.end_index = min(
                s.start_index + s.frame_size - 1, len(s.current_chunk) - 1
            )

        # Cumulative time offset: sum the duration of all preceding big chunks
        # so that start_time / end_time are absolute positions in the file,
        # not relative to the start of the current chunk.
        chunk_time_offset = sum(
            len(s.audio_chunks[i]) / s.fs
            for i in range(self._big_chunk_idx)
        )

        s.start_time = round(s.start_index / s.fs + chunk_time_offset)
        s.end_time   = round(s.end_index   / s.fs + chunk_time_offset)

        ds_frame_size      = int(s.label_frame_length * s.target_fs)
        s.down_start_index = (frame_idx - 1) * ds_frame_size
        s.down_end_index   = min(
            s.down_start_index + ds_frame_size - 1,
            len(s.downsampled) - 1
        )

        raw = s.current_chunk[s.start_index : s.end_index + 1]
        s.chunk = pad_chunk(raw, s.windowsize_sa, s.overlap_samples_sa)

        ds_raw = s.downsampled[s.down_start_index : s.down_end_index + 1]
        s.ds_chunk = pad_chunk(ds_raw, s.windowsize_sc, s.overlap_samples_sc)

        s.current_plot_chunk = self._build_context_window(frame_idx)

    def _build_context_window(self, frame_idx: int) -> np.ndarray:
        s  = self._state
        fs = s.frame_size

        start = s.start_index
        end   = s.end_index
        n     = len(s.current_chunk)

        if frame_idx == 1:
            ctx_start = start
            ctx_end   = min(start + 3 * fs - 1, n - 1)
        elif frame_idx == 2:
            ctx_start = start - fs
            ctx_end   = min(start + 2 * fs - 1, n - 1)
        elif frame_idx == s.num_chunks:
            ctx_start = max(0, start - 2 * fs)
            ctx_end   = end
        else:
            ctx_start = max(0, start - fs)
            ctx_end   = min(end + fs, n - 1)

        chunk = s.current_chunk[ctx_start : ctx_end + 1]

        if len(chunk) < s.windowsize_cp:
            chunk = np.concatenate(
                [chunk, np.zeros(s.windowsize_cp - len(chunk), dtype=chunk.dtype)]
            )
        return chunk

    def _render_overview(self) -> None:
        """
        Render the full-chunk waveform and large overview spectrogram.
        Called once per big chunk (not per frame).  Markers are updated
        per-frame by _render_spectrograms().
        """
        s = self._state

        self._waveform_canvas.render_waveform(
            s.current_chunk, s.fs,
            marker_time=float(s.start_time) if s.start_time else None
        )

        self._large_spec_canvas.render(
            signal          = s.current_chunk,
            fs              = s.fs,
            window_size     = 2048,
            overlap_samples = 1024,
            f_min           = 0.0,
            f_max           = s.fs / 2.0,
            dr_low          = s.dynamic_range_l,
            dr_high         = s.dynamic_range_u,
            colormap        = s.colormap,
            use_mel         = False,
            marker_times    = [float(s.start_time)] if s.start_time else None,
            hide_y_labels   = True,
        )

    def _render_spectrograms(self) -> None:
        """
        Render CP, A, B, C for the current frame and update the
        waveform and large-spectrogram position markers.
        """
        s = self._state

        frame_idx = s.current_chunk_index
        fs        = s.frame_size
        n         = len(s.current_chunk)

        if frame_idx == 1:
            ctx_start_sample = s.start_index
        elif frame_idx == 2:
            ctx_start_sample = s.start_index - fs
        elif frame_idx == s.num_chunks:
            ctx_start_sample = max(0, s.start_index - 2 * fs)
        else:
            ctx_start_sample = max(0, s.start_index - fs)

        ctx_start_sample = max(0, ctx_start_sample)
        frame_t0 = (s.start_index - ctx_start_sample) / s.fs
        frame_t1 = frame_t0 + s.label_frame_length

        # Compute the absolute file time at x=0 for each canvas so that
        # click coordinates can be converted to absolute times when stored.
        chunk_time_offset = sum(
            len(s.audio_chunks[i]) / s.fs
            for i in range(self._big_chunk_idx)
        )
        cp_time_offset  = chunk_time_offset + ctx_start_sample / s.fs
        abc_time_offset = float(s.start_time)   # already absolute after fix

        self._cp_canvas.set_time_offset(cp_time_offset)
        self._sa_canvas.set_time_offset(abc_time_offset)
        self._sb_canvas.set_time_offset(abc_time_offset)
        self._sc_canvas.set_time_offset(abc_time_offset)

        self._cp_canvas.render(
            signal          = s.current_plot_chunk,
            fs              = s.fs,
            window_size     = s.windowsize_cp,
            overlap_samples = s.overlap_samples_cp,
            f_min           = s.min_f_cp,
            f_max           = s.max_f_cp,
            dr_low          = s.dynamic_range_l,
            dr_high         = s.dynamic_range_u,
            colormap        = s.colormap,
            use_mel         = True,
            marker_times    = [frame_t0, frame_t1],
        )

        self._sa_canvas.render(
            signal          = s.chunk,
            fs              = s.fs,
            window_size     = s.windowsize_sa,
            overlap_samples = s.overlap_samples_sa,
            f_min           = s.min_f_sa,
            f_max           = s.max_f_sa,
            dr_low          = s.dynamic_range_l,
            dr_high         = s.dynamic_range_u,
            colormap        = s.colormap,
            use_mel         = s.mel_sa,
        )

        self._sb_canvas.render(
            signal          = s.chunk,
            fs              = s.fs,
            window_size     = s.windowsize_sb,
            overlap_samples = s.overlap_samples_sb,
            f_min           = s.min_f_sb,
            f_max           = s.max_f_sb,
            dr_low          = s.dynamic_range_l,
            dr_high         = s.dynamic_range_u,
            colormap        = s.colormap,
            use_mel         = s.mel_sb,
        )

        self._sc_canvas.render(
            signal          = s.ds_chunk,
            fs              = s.target_fs,
            window_size     = s.windowsize_sc,
            overlap_samples = s.overlap_samples_sc,
            f_min           = s.min_f_sc,
            f_max           = s.max_f_sc,
            dr_low          = s.dynamic_range_l,
            dr_high         = s.dynamic_range_u,
            colormap        = s.colormap,
            use_mel         = False,
        )

        start_t = float(s.start_time)
        self._waveform_canvas.render_waveform(
            s.current_chunk, s.fs, marker_time=start_t
        )
        self._large_spec_canvas.render(
            signal          = s.current_chunk,
            fs              = s.fs,
            window_size     = 2048,
            overlap_samples = 1024,
            f_min           = 0.0,
            f_max           = s.fs / 2.0,
            dr_low          = s.dynamic_range_l,
            dr_high         = s.dynamic_range_u,
            colormap        = s.colormap,
            use_mel         = False,
            marker_times    = [start_t],
            hide_y_labels   = True,
        )

        # Redraw persisted event annotations for the newly displayed frame.
        self._refresh_annotation_overlays()
        self._update_event_count_display()

    # UI state helpers


    def _reset_controls(self) -> None:
        for cb in self._checkboxes:
            cb.setChecked(False)
        if self._conf_btns:
            self._conf_btns[0].setChecked(True)
        self._comments_edit.clear()
        self._counter_value = float("nan")
        self._counter_display.setText("")
        # Annotation mode intentionally preserved across frame navigation —
        # the user keeps their selected tool (None / Point / Box) active.

    def _update_progress_display(self) -> None:
        s = self._state
        total_big_chunks  = len(s.audio_chunks)
        current_big_chunk = self._big_chunk_idx + 1  # 1-based

        self._chunk_display.setText(f"{current_big_chunk} of {total_big_chunks}")
        self._frame_display.setText(f"{s.current_chunk_index} of {s.num_chunks}")


    # Finishing label state


    def _finish_labelling(self) -> None:
        s = self._state

        try:
            fixed    = fix_chunk_numbering(s.labels_table, s.label_frame_length)
            csv_path = export_labels(fixed, s.audio_file_path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error", f"Failed to write labels CSV:\n{exc}"
            )
            return

        # Export event labels (point + box annotations).
        event_csv_path = None
        if s.event_labels_table is not None:
            try:
                event_csv_path = export_event_labels(s.event_labels_table, s.audio_file_path)
            except Exception as exc:
                QMessageBox.warning(
                    self, "Export Warning",
                    f"Frame labels saved, but failed to write event labels:\n{exc}"
                )

        msg = f"All frames labelled.\nLabels saved to:\n{csv_path}"
        if event_csv_path:
            msg += f"\nEvent labels saved to:\n{event_csv_path}"

        QMessageBox.information(self, "Labelling Complete", msg)

        s.reset_navigation()
        self._next_btn.setEnabled(True)
