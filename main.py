"""
HydroSeek — PyQt6 desktop application for annotating passive acoustic files.

Entry point. 

Run with:
    python main.py
"""

import sys

from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("HydroSeek")
    app.setOrganizationName("HydroSeek")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
