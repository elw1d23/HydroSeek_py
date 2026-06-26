"""
SetupTab — PyQt6 widget for the HydroSeek Set Up tab.

Mirrors the MATLAB HydroSeekSetUpTab exactly:
    - Audio file picker (wav / mp3 / flac)
    - Load config CSV button / Export config CSV button
    - Spectrogram parameters: Context Plot, Spec A, Spec B, Spec C
    - Acoustic settings: chunks, frame length, downsample fs, dynamic range
    - 18 label name fields
    - Start button

On Start:
    1. Reads all fields into AppState
    2. Calls audio.load_audio + audio.chunk_audio
    3. Calls audio.resample_audio for the downsampled signal
    4. Calls labelling.create_labels_table
    5. Computes overlap sample counts for all four spectrograms
    6. Calls main_window.switch_to_labelling()
"""

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox,
    QDoubleSpinBox, QSpinBox, QGroupBox, QScrollArea,
    QSizePolicy, QMessageBox, QFileDialog, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap

from hydroseek.state import AppState
from hydroseek.audio import load_audio, chunk_audio, resample_audio, check_existing_labels, get_audio_info
from hydroseek.config import load_config, export_config, config_to_state_kwargs
from hydroseek.labelling import create_labels_table
from hydroseek.event_labelling import create_event_labels_table, export_event_config
from hydroseek.signal_processing import overlap_percent_to_samples


# Visual constants

_BODY_PX  = 16   # px, not pt — used in stylesheet strings directly
_SMALL_PX = 14
_CTRL_H   = 28   # minimum height for all interactive controls

# Stylesheet
#

_SETUP_STYLESHEET = (
    # QWidget base
    "QWidget {"
    "  font-family: Arial;"
    f" font-size: {_BODY_PX}px;"
    "  color: #3a3a3a;"
    "  background-color: #ffffff;"
    "}"

    # Group boxes
    "QGroupBox {"
    "  border: 1px solid #cccccc;"
    "  border-radius: 6px;"
    "  margin-top: 22px;"
    "  padding: 6px;"
    "  background-color: #ffffff;"
    f" font-size: {_BODY_PX}px;"
    "  font-weight: bold;"
    "  color: #3a3a3a;"
    "}"
    "QGroupBox::title {"
    "  subcontrol-origin: margin;"
    "  subcontrol-position: top left;"
    "  left: 10px;"
    "  padding: 0 4px;"
    "  color: #3a3a3a;"
    "}"

    # Plain buttons 
    "QPushButton {"
    "  background-color: #ffffff;"
    "  color: #3a3a3a;"
    "  border: 1px solid #bbbbbb;"
    "  border-radius: 5px;"
    "  padding: 4px 14px;"
    f" font-size: {_BODY_PX}px;"
    f" min-height: {_CTRL_H}px;"
    "}"
    "QPushButton:hover  { background-color: #d5d5d5; }"
    "QPushButton:pressed { background-color: #c0c0c0; }"
    "QPushButton:disabled { color: #999999; border-color: #dddddd; }"

    # Spin boxes and line edits
    "QSpinBox, QDoubleSpinBox, QLineEdit {"
    "  background-color: #ffffff;"
    "  border: 1px solid #bbbbbb;"
    "  border-radius: 4px;"
    "  padding: 2px 6px;"
    f" font-size: {_BODY_PX}px;"
    f" min-height: {_CTRL_H}px;"
    "}"
    "QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {"
    "  border-color: #2a7fcf;"
    "}"

    # Checkboxes
    "QCheckBox {"
    "  spacing: 6px;"
    f" font-size: {_BODY_PX}px;"
    "}"
    "QCheckBox::indicator {"
    "  width: 16px;"
    "  height: 16px;"
    "  border: 1px solid #aaaaaa;"
    "  border-radius: 3px;"
    "  background-color: #ffffff;"
    "}"
    "QCheckBox::indicator:checked {"
    "  background-color: #ffffff;"
    "  border-color: #2a7fcf;"
    "}"

    # Scroll area 
    "QScrollArea { background-color: transparent; border: none; }"
    "QScrollArea > QWidget > QWidget { background-color: transparent; }"
)

# Start button override 
_START_BTN_STYLESHEET = (
    "QPushButton#StartButton {"
    "  background-color: #2a7fcf;"
    "  color: white;"
    "  border: none;"
    "  border-radius: 6px;"
    "  font-size: 13pt;"
    "  font-weight: bold;"
    "}"
    "QPushButton#StartButton:hover  { background-color: #1a6fbf; }"
    "QPushButton#StartButton:pressed { background-color: #1060af; }"
    "QPushButton#StartButton:disabled { background-color: #aaaaaa; color: #eeeeee; }"
)

# Spin-box helpers


def _float_spin(
    value: float,
    lo: float,
    hi: float,
    step: float = 1.0,
    decimals: int = 1,
    width: int = 110,
) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setFixedWidth(width)
    sb.setMinimumHeight(_CTRL_H)
    return sb


def _int_spin(value: int, lo: int, hi: int, width: int = 110) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    sb.setFixedWidth(width)
    sb.setMinimumHeight(_CTRL_H)
    return sb


class SetupTab(QWidget):
    """Setup / configuration tab."""

    def __init__(self, state: AppState, main_window) -> None:
        super().__init__()
        self._state = state
        self._mw    = main_window
        self.setStyleSheet(_SETUP_STYLESHEET)
        self._build_ui()

    
    # UI construction


    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)

        # Header — logo 
        # Structure:
        #   QHBoxLayout (header_row)
        #     QLabel (logo, 140x140)
        #     QVBoxLayout (right_col)
        #       QLabel ("HydroSeek" title)
        #       QFrame (HLine separator)
        #       QHBoxLayout (audio file row: label + path edit + Browse button)
        #       QLabel (audio info — Fs, duration, channels)

        header_row = QHBoxLayout()
        header_row.setSpacing(20)
        header_row.setContentsMargins(0, 0, 0, 0)

        # Logo: 140x140 
        self._logo_label = QLabel()
        self._logo_label.setFixedSize(160, 160)
        self._logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
   
        self._logo_label.setStyleSheet("background-color: #ffffff;")
        self._try_load_logo()
        header_row.addWidget(self._logo_label)

        # Right column: title + audio picker stacked vertically.
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        right_col.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel("HydroSeek")
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        # 
        # QLabel inherits background from QWidget (#ffffff).
        title_lbl.setStyleSheet(
            "font-size: 32pt; font-weight: bold; color: #3a3a3a;"
            " background-color: transparent;"
        )
        right_col.addWidget(title_lbl)

        # Thin rule below the title.
        title_sep = QFrame()
        title_sep.setFrameShape(QFrame.Shape.HLine)
        title_sep.setFrameShadow(QFrame.Shadow.Plain)
        title_sep.setStyleSheet("color: #dddddd; background-color: #dddddd;")
        title_sep.setFixedHeight(1)
        right_col.addWidget(title_sep)

        # Audio file row: descriptive label + path field + Browse button
        audio_row = QHBoxLayout()
        audio_row.setSpacing(8)

        audio_lbl = QLabel("Audio file (wav, mp3, flac):")
        audio_lbl.setStyleSheet("background-color: transparent;")
        audio_row.addWidget(audio_lbl)

        self._audio_path_edit = QLineEdit()
        self._audio_path_edit.setReadOnly(True)
        self._audio_path_edit.setPlaceholderText("No file selected")
        audio_row.addWidget(self._audio_path_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(110)
        browse_btn.clicked.connect(self._on_browse_audio)
        audio_row.addWidget(browse_btn)

        right_col.addLayout(audio_row)

        # Audio info line — appears below the file row once a file is loaded.
        # Starts empty; _on_browse_audio populates it.
        self._audio_info_label = QLabel("")
        self._audio_info_label.setStyleSheet(
            f"color: #3a3a3a; font-size: {_BODY_PX}px; background-color: transparent;"
        )
        right_col.addWidget(self._audio_info_label)

        right_col.addStretch(1)   # pushes content to the top within the column

        header_row.addLayout(right_col, stretch=1)
        root.addLayout(header_row)

        # Separator between header and scrollable body.
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: #cccccc; background-color: #cccccc;")
        sep.setFixedHeight(1)
        root.addWidget(sep)

        # Scrollable body 
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setStyleSheet("background-color: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setSpacing(12)
        body_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        body_layout.addWidget(self._build_config_group())

        mid_row = QHBoxLayout()
        mid_row.setSpacing(12)
        mid_row.addWidget(self._build_acoustic_group(),    stretch=1)
        mid_row.addWidget(self._build_spectrogram_group(), stretch=2)
        body_layout.addLayout(mid_row)

        body_layout.addWidget(self._build_labels_group())
        body_layout.addStretch(1)


        # Footer — attribution (left) + Start button (right).
        # Always visible below the scroll area.
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        sep2.setStyleSheet("color: #cccccc; background-color: #cccccc;")
        sep2.setFixedHeight(1)
        root.addWidget(sep2)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 6, 0, 0)
        footer.setSpacing(12)

        attribution = QLabel("White, 2026")
        attribution.setStyleSheet(
            f"color: #3a3a3a; font-size: {_BODY_PX}px; font-style: italic;"
            " background-color: transparent;"
        )
        attribution.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        footer.addWidget(attribution)

        footer.addStretch(1)

        self._start_btn = QPushButton("  Start  ")
        self._start_btn.setFixedSize(200, 48)
        start_font = QFont()
        start_font.setPointSize(13)
        start_font.setBold(True)
        self._start_btn.setFont(start_font)
        self._start_btn.setObjectName("StartButton")
        self._start_btn.setStyleSheet(_START_BTN_STYLESHEET)
        self._start_btn.clicked.connect(self.on_start)
        footer.addWidget(self._start_btn)

        root.addLayout(footer)

    # Logo loader

    def _try_load_logo(self) -> None:
        """
        Load logo.png from the same directory as this file, if it exists.

        Place a file called 'logo.png' alongside setup_tab.py.
        Any PNG or JPEG is accepted.  If the file is absent, the label
        stays blank — no error is raised.

        WHY os.path.abspath(__file__)?
        Resolves the correct directory regardless of which working directory
        the user launches the app from.
        """
        here      = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(here, "logo.png")

        if not os.path.isfile(logo_path):
            return

        pixmap = QPixmap(logo_path)
        if pixmap.isNull():
            return

        # Scale to fit the label's fixed size, preserving aspect ratio.
        # SmoothTransformation applies antialiasing so the image looks
        # sharp even when scaled down significantly.
        scaled = pixmap.scaled(
            self._logo_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._logo_label.setPixmap(scaled)

    # Panel builders

    def _build_config_group(self) -> QGroupBox:
        grp = QGroupBox("Configuration File")
        lay = QHBoxLayout(grp)
        lay.setSpacing(10)

        lbl = QLabel("Load a saved config CSV, or set parameters manually below:")
        lbl.setStyleSheet("background-color: transparent;")
        lay.addWidget(lbl)
        lay.addStretch(1)

        load_btn = QPushButton("Load Config File")
        load_btn.setFixedWidth(160)
        load_btn.clicked.connect(self._on_load_config)
        lay.addWidget(load_btn)

        export_btn = QPushButton("Export Config File")
        export_btn.setFixedWidth(160)
        export_btn.clicked.connect(self._on_export_config)
        lay.addWidget(export_btn)

        return grp

    def _build_acoustic_group(self) -> QGroupBox:
        grp = QGroupBox("Acoustic Settings")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.setContentsMargins(10, 20, 10, 10)

        s = self._state
        self._chunks_spin    = _int_spin(s.file_chunk_number, 1, 100)
        self._frame_len_spin = _float_spin(s.label_frame_length, 0.5, 60.0, step=0.5, decimals=1)
        self._ds_fs_spin     = _int_spin(s.target_fs, 100, 96000)
        self._dr_low_spin    = _float_spin(s.dynamic_range_l, -200.0, 0.0, step=5.0, decimals=1)
        self._dr_high_spin   = _float_spin(s.dynamic_range_u, -50.0, 100.0, step=5.0, decimals=1)

        rows = [
            ("Split file into chunks:",   self._chunks_spin),
            ("Frame length (s):",         self._frame_len_spin),
            ("Downsample Fs (Hz):",       self._ds_fs_spin),
            ("Dynamic range lower (dB):", self._dr_low_spin),
            ("Dynamic range upper (dB):", self._dr_high_spin),
        ]
        for r, (label_text, widget) in enumerate(rows):
            lbl = QLabel(label_text)
            lbl.setStyleSheet("background-color: transparent;")
            grid.addWidget(lbl, r, 0)
            grid.addWidget(widget, r, 1)

        grid.setColumnStretch(0, 1)
        grid.setRowStretch(len(rows), 1)
        return grp

    def _build_spectrogram_group(self) -> QGroupBox:
        grp = QGroupBox("Spectrogram Settings")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setContentsMargins(10, 20, 10, 10)

        header_font = QFont()
        header_font.setBold(True)
        for col, text in enumerate(["", "Window", "Overlap %", "F1 (Hz)", "F2 (Hz)", "Mel"]):
            lbl = QLabel(text)
            lbl.setFont(header_font)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"font-size: {_SMALL_PX}px; color: #555555; background-color: transparent;"
            )
            grid.addWidget(lbl, 0, col)

        s = self._state
        specs = [
            ("Context Plot",       "_cp", False, s.windowsize_cp, s.overlap_cp, s.min_f_cp, s.max_f_cp, False),
            ("Spectrogram A",      "_sa", True,  s.windowsize_sa, s.overlap_sa, s.min_f_sa, s.max_f_sa, s.mel_sa),
            ("Spectrogram B",      "_sb", True,  s.windowsize_sb, s.overlap_sb, s.min_f_sb, s.max_f_sb, s.mel_sb),
            ("Spectrogram C (ds)", "_sc", False, s.windowsize_sc, s.overlap_sc, s.min_f_sc, s.max_f_sc, False),
        ]

        for r, (label, prefix, has_mel, win_v, ovlp_v, f1_v, f2_v, mel_v) in enumerate(specs, start=1):
            row_lbl = QLabel(label)
            row_lbl.setStyleSheet("background-color: transparent;")
            grid.addWidget(row_lbl, r, 0)

            win_w  = _int_spin(win_v,  64, 65536, width=100)
            ovlp_w = _float_spin(ovlp_v, 0.0, 99.0, step=5.0, width=90)
            f1_w   = _float_spin(f1_v,   0.0, 100000.0, step=10.0,  decimals=1, width=100)
            f2_w   = _float_spin(f2_v,   1.0, 200000.0, step=100.0, decimals=1, width=100)

            grid.addWidget(win_w,  r, 1)
            grid.addWidget(ovlp_w, r, 2)
            grid.addWidget(f1_w,   r, 3)
            grid.addWidget(f2_w,   r, 4)

            setattr(self, f"{prefix}_win",  win_w)
            setattr(self, f"{prefix}_ovlp", ovlp_w)
            setattr(self, f"{prefix}_f1",   f1_w)
            setattr(self, f"{prefix}_f2",   f2_w)

            if has_mel:
                mel_cb = QCheckBox()
                mel_cb.setChecked(mel_v)
                mel_cb.setToolTip("Use Mel scale")
                grid.addWidget(mel_cb, r, 5, Qt.AlignmentFlag.AlignCenter)
                setattr(self, f"{prefix}_mel", mel_cb)

        grid.setColumnStretch(0, 1)
        return grp

    def _build_labels_group(self) -> QGroupBox:
        grp = QGroupBox("Assign Label Names (up to 18)")
        grid = QGridLayout(grp)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setContentsMargins(10, 20, 10, 10)

        self._label_edits: list[QLineEdit] = []

        for i in range(18):
            row = i // 6
            col = (i % 6) * 2

            num_lbl = QLabel(f"{i + 1}.")
            num_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            num_lbl.setStyleSheet(
                f"color: #666666; font-size: {_BODY_PX}px; background-color: transparent;"
            )
            grid.addWidget(num_lbl, row, col)

            edit = QLineEdit()
            edit.setPlaceholderText(f"label_{i + 1}")
            current = self._state.labels[i]
            edit.setText("" if current.startswith("NA_") else current)
            edit.setMinimumWidth(110)
            edit.setMinimumHeight(_CTRL_H)
            grid.addWidget(edit, row, col + 1)
            self._label_edits.append(edit)

        return grp

    # Callbacks


    def _on_browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)",
        )
        if not path:
            return

        self._audio_path_edit.setText(path)
        self._audio_duration_seconds: float = 0.0   # reset on each new file

        try:
            info = get_audio_info(path)
            self._audio_duration_seconds = info["duration_seconds"]
            self._audio_info_label.setText(
                f"Fs = {info['sample_rate']} Hz     "
                f"{info['duration_minutes']:.1f} min     "
                f"{info['channels']} ch"
            )
        except Exception as exc:
            self._audio_info_label.setText(f"Could not read info: {exc}")

        # Warn immediately if a labels file already exists for this audio.
        # This gives the user a chance to pick a different file or directory
        # before they finish configuring all the parameters.
        
        if check_existing_labels(path):
            QMessageBox.warning(
                self,
                "Labels File Already Exists",
                f"A labels CSV already exists for this file:\n"
                f"{path}\n\n"
                "If you continue, starting a session will overwrite it.\n"
                "To keep the existing labels, either select a different audio "
                "file or move the labels CSV to another directory before "
                "pressing Start.",
            )

    def _on_load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Config CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            cfg    = load_config(path)
            kwargs = config_to_state_kwargs(cfg)
            for key, val in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, val)
        except Exception as exc:
            QMessageBox.warning(self, "Config Load Error", str(exc))
            return

        self._populate_fields_from_state()

    def _on_export_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config CSV", "hydroseek_config.csv", "CSV Files (*.csv)"
        )
        if not path:
            return

        self._read_fields_into_state()

        s = self._state
        values = {
            "File_Chunk_No":       s.file_chunk_number,
            "Frame_Length":        s.label_frame_length,
            "Downsample_fs":       s.target_fs,
            "Dynamic_Range_Lower": s.dynamic_range_l,
            "Dynamic_Range_Upper": s.dynamic_range_u,
            "CP_WindowSize":       s.windowsize_cp,
            "CP_Overlap":          s.overlap_cp,
            "CP_F1":               s.min_f_cp,
            "CP_F2":               s.max_f_cp,
            "SA_WindowSize":       s.windowsize_sa,
            "SA_Overlap":          s.overlap_sa,
            "SA_F1":               s.min_f_sa,
            "SA_F2":               s.max_f_sa,
            "SB_WindowSize":       s.windowsize_sb,
            "SB_Overlap":          s.overlap_sb,
            "SB_F1":               s.min_f_sb,
            "SB_F2":               s.max_f_sb,
            "SC_WindowSize":       s.windowsize_sc,
            "SC_Overlap":          s.overlap_sc,
            "SC_F1":               s.min_f_sc,
            "SC_F2":               s.max_f_sc,
        }
        for i, lbl in enumerate(s.labels, 1):
            values[f"label_{i}"] = lbl

        try:
            export_config(path, values)
            QMessageBox.information(self, "Config Saved", f"Config written to:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Config Export Error", str(exc))

    def on_start(self) -> None:
        """Validate inputs, load audio, and hand off to the labelling tab."""
        audio_path = self._audio_path_edit.text().strip()
        if not audio_path:
            QMessageBox.warning(self, "No Audio File", "Please select an audio file first.")
            return
        if not os.path.isfile(audio_path):
            QMessageBox.warning(self, "File Not Found", f"Cannot find:\n{audio_path}")
            return

        self._read_fields_into_state()

        # Safety-net labels check (in case the file appeared after browse).
        if check_existing_labels(audio_path):
            ret = QMessageBox.question(
                self,
                "Existing Labels File",
                "A labels CSV already exists for this audio file.\n"
                "Starting a new session will overwrite it.\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        # Truncation warning — must run after _read_fields_into_state()
        # so we have the current chunk count and frame length values.
        if not self._check_truncation_and_warn():
            return   # user chose to cancel and adjust parameters

        self._start_btn.setEnabled(False)
        self._start_btn.setText("Loading...")

        try:
            self._load_audio_and_initialise(audio_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load audio:\n{exc}")
            self._start_btn.setEnabled(True)
            self._start_btn.setText("  Start  ")
            return

        self._start_btn.setEnabled(True)
        self._start_btn.setText("  Start  ")
        self._mw.switch_to_labelling()

    # Internal helpers
   

    def _check_truncation_and_warn(self) -> bool:
        """
        Calculate whether the current duration / chunk / frame settings
        will result in audio being excluded from labelling, and if so
        show an informative warning.

        Returns True if the user wants to proceed, False if they cancel.

        Two truncation effects are checked:

        1. Sub-frame tail  — audio that is shorter than one full frame
           at the very end of the file.  e.g. 9.7 min file with 60 s
           frames: 9*60 = 540 s used, 9.7*60 - 540 = 42 s tail.
           This tail ends up inside the last chunk as trailing samples
           that pad_chunk() will zero-pad for spectrogram display but
           which will NOT receive their own label row.

        2. Uneven chunk division — if total_complete_frames is not
           divisible by file_chunk_number, the last chunk gets more
           frames than the others.  e.g. 11 frames across 4 chunks
           gives 2+2+2+5. This is not an error but can surprise users
           who expect equal chunk sizes.
        """
        duration = getattr(self, "_audio_duration_seconds", 0.0)
        if duration <= 0.0:
            # Duration not known (file not yet read or read failed) — skip.
            return True

        s = self._state
        frame_len   = s.label_frame_length
        n_chunks    = s.file_chunk_number

        total_complete_frames = int(duration // frame_len)
        labelled_duration     = total_complete_frames * frame_len
        tail_seconds          = duration - labelled_duration

        frames_per_chunk      = total_complete_frames // n_chunks
        frames_in_last_chunk  = total_complete_frames - frames_per_chunk * (n_chunks - 1)
        uneven                = (frames_in_last_chunk != frames_per_chunk)

        # Build warning text only if something worth flagging exists.
        issues = []

        if tail_seconds >= 0.5:
            # Only flag if the tail is >= 0.5 s — smaller remainders are
            # negligible and not worth alarming the user about.
            issues.append(
                f"  Unlabelled tail: {tail_seconds:.1f} s at the end of the file\n"
                f"  ({total_complete_frames} complete {frame_len:.0f} s frames "
                f"= {labelled_duration:.1f} s labelled out of {duration:.1f} s total)\n"
                "  The tail will be retained in the last chunk for spectrogram\n"
                "  display but will not receive its own label row."
            )

        if uneven and n_chunks > 1:
            issues.append(
                f"  Uneven chunk sizes: {n_chunks - 1} chunk(s) will have "
                f"{frames_per_chunk} frame(s),\n"
                f"  but the last chunk will have {frames_in_last_chunk} frame(s).\n"
                "  This is normal behaviour — it does not affect labelling accuracy."
            )

        if not issues:
            return True   # nothing to warn about

        msg = (
            "The current settings will result in the following:\n\n"
            + "\n\n".join(issues)
            + "\n\nDo you want to continue with these settings?"
        )

        ret = QMessageBox.question(
            self,
            "Audio Coverage Warning",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return ret == QMessageBox.StandardButton.Yes

    def _load_audio_and_initialise(self, audio_path: str) -> None:
        s = self._state

        audio_data, fs = load_audio(audio_path)
        chunks, frames_per_chunk, frames_in_last_chunk, frame_size = chunk_audio(
            audio_data, fs, s.file_chunk_number, s.label_frame_length
        )

        downsampled = resample_audio(audio_data, fs, s.target_fs)

        s.overlap_samples_cp = overlap_percent_to_samples(s.overlap_cp, s.windowsize_cp)
        s.overlap_samples_sa = overlap_percent_to_samples(s.overlap_sa, s.windowsize_sa)
        s.overlap_samples_sb = overlap_percent_to_samples(s.overlap_sb, s.windowsize_sb)
        s.overlap_samples_sc = overlap_percent_to_samples(s.overlap_sc, s.windowsize_sc)

        s.audio_data           = audio_data
        s.fs                   = fs
        s.audio_chunks         = chunks
        s.audio_file_path      = audio_path
        s.audio_length         = len(audio_data) / fs
        s.frame_size           = frame_size
        s.num_frames_to_label  = int(s.audio_length // s.label_frame_length)
        s.num_chunks_per_split = frames_per_chunk
        s.frames_in_last_chunk = frames_in_last_chunk
        s.downsampled          = downsampled

        s.current_chunk_index  = 0
        s.current_frame_number = 0
        s.chunk_number         = 0
        s.num_chunks           = 0

        s.labels_table = create_labels_table(s.labels)

        # Initialise event labels table and write the config CSV immediately
        # so the spectrogram settings are recorded even if the session is aborted.
        s.event_labels_table = create_event_labels_table()
        s.event_id_counter   = 0
        s.annotation_mode    = "none"
        try:
            export_event_config(s, audio_path)
        except Exception as exc:
            print(f"Warning: could not write event config CSV: {exc}")

        print(
            f"Audio loaded: {fs} Hz, {s.audio_length:.1f} s, "
            f"{len(chunks)} chunk(s), {s.num_frames_to_label} frames total"
        )

    def _read_fields_into_state(self) -> None:
        """Copy all UI field values into AppState."""
        s = self._state

        s.file_chunk_number  = self._chunks_spin.value()
        s.label_frame_length = self._frame_len_spin.value()
        s.target_fs          = self._ds_fs_spin.value()
        s.dynamic_range_l    = self._dr_low_spin.value()
        s.dynamic_range_u    = self._dr_high_spin.value()

        s.windowsize_cp = self._cp_win.value()
        s.overlap_cp    = self._cp_ovlp.value()
        s.min_f_cp      = self._cp_f1.value()
        s.max_f_cp      = self._cp_f2.value()

        s.windowsize_sa = self._sa_win.value()
        s.overlap_sa    = self._sa_ovlp.value()
        s.min_f_sa      = self._sa_f1.value()
        s.max_f_sa      = self._sa_f2.value()
        s.mel_sa        = self._sa_mel.isChecked()

        s.windowsize_sb = self._sb_win.value()
        s.overlap_sb    = self._sb_ovlp.value()
        s.min_f_sb      = self._sb_f1.value()
        s.max_f_sb      = self._sb_f2.value()
        s.mel_sb        = self._sb_mel.isChecked()

        s.windowsize_sc = self._sc_win.value()
        s.overlap_sc    = self._sc_ovlp.value()
        s.min_f_sc      = self._sc_f1.value()
        s.max_f_sc      = self._sc_f2.value()

        s.labels = [edit.text().strip() for edit in self._label_edits]

    def _populate_fields_from_state(self) -> None:
        """Push AppState values back into all UI widgets (called after config load)."""
        s = self._state

        self._chunks_spin.setValue(s.file_chunk_number)
        self._frame_len_spin.setValue(s.label_frame_length)
        self._ds_fs_spin.setValue(s.target_fs)
        self._dr_low_spin.setValue(s.dynamic_range_l)
        self._dr_high_spin.setValue(s.dynamic_range_u)

        self._cp_win.setValue(s.windowsize_cp)
        self._cp_ovlp.setValue(s.overlap_cp)
        self._cp_f1.setValue(s.min_f_cp)
        self._cp_f2.setValue(s.max_f_cp)

        self._sa_win.setValue(s.windowsize_sa)
        self._sa_ovlp.setValue(s.overlap_sa)
        self._sa_f1.setValue(s.min_f_sa)
        self._sa_f2.setValue(s.max_f_sa)
        self._sa_mel.setChecked(s.mel_sa)

        self._sb_win.setValue(s.windowsize_sb)
        self._sb_ovlp.setValue(s.overlap_sb)
        self._sb_f1.setValue(s.min_f_sb)
        self._sb_f2.setValue(s.max_f_sb)
        self._sb_mel.setChecked(s.mel_sb)

        self._sc_win.setValue(s.windowsize_sc)
        self._sc_ovlp.setValue(s.overlap_sc)
        self._sc_f1.setValue(s.min_f_sc)
        self._sc_f2.setValue(s.max_f_sc)

        for i, lbl in enumerate(s.labels):
            self._label_edits[i].setText("" if lbl.startswith("NA_") else lbl)
