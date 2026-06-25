"""
MainWindow — the top-level QMainWindow for HydroSeek.

Owns a QTabWidget with two tabs:
    Tab 0 — SetupTab   (config, file picker, spectrogram parameters, labels)
    Tab 1 — LabellingTab  (spectrograms, checkboxes, controls)

Also owns the single AppState instance shared between the two tabs.
Neither tab holds state directly; they read from and write to state.

The labelling tab is initially disabled and only becomes active after the
user successfully presses Start on the setup tab, at which point the main
window switches to it automatically.
"""

from PyQt6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QWidget,
)
from PyQt6.QtCore import Qt

from hydroseek.state import AppState
from setup_tab import SetupTab
from labelling_tab import LabellingTab


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()

        self.state = AppState()

        self.setWindowTitle("HydroSeek")
        self.resize(1280, 960)
        self.setMinimumSize(1024, 700)

        self._build_ui()

    # UI construction

    def _build_ui(self) -> None:
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(False)
        self._tabs.setDocumentMode(True)                          # use Qt painter, not native
        self._tabs.setStyleSheet(                                 # white pane background
            "QTabWidget::pane { background-color: #ffffff; border: none; }"
            "QWidget { background-color: #ffffff; }"
    )

        self._setup_tab    = SetupTab(state=self.state, main_window=self)
        self._labelling_tab = LabellingTab(state=self.state, main_window=self)

        self._tabs.addTab(self._setup_tab,    "HydroSeek Set Up")
        self._tabs.addTab(self._labelling_tab, "Labelling")

        # Labelling tab is locked until Start is pressed
        self._tabs.setTabEnabled(1, False)

        self.setCentralWidget(self._tabs)

    # Public API called by SetupTab after a successful Start

    def switch_to_labelling(self) -> None:
        """
        Enable the labelling tab and switch to it.

        Called by SetupTab.on_start() once audio is loaded and the labels
        table is initialised.
        """
        self._tabs.setTabEnabled(1, True)
        self._tabs.setCurrentIndex(1)
        self._labelling_tab.on_session_start()
