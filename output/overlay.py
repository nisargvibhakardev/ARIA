from __future__ import annotations
from typing import Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from config import OverlayConfig


class Overlay(QWidget):
    dismissed = pyqtSignal()

    def __init__(
        self,
        config: OverlayConfig,
        on_engage: Callable | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._on_engage = on_engage
        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.timeout.connect(self.hide_message)
        self._setup_ui()
        self.hide()

    def _setup_ui(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        self._msg_label = QLabel()
        self._msg_label.setWordWrap(True)
        self._msg_label.setStyleSheet("color: #e2e8f0; font-size: 13px;")
        layout.addWidget(self._msg_label)

        self._reason_label = QLabel()
        self._reason_label.setWordWrap(True)
        self._reason_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        layout.addWidget(self._reason_label)

        btn_row = QHBoxLayout()
        self._got_it_btn = QPushButton("Got it")
        self._got_it_btn.setStyleSheet(
            "QPushButton { background:#1e40af; color:white; border-radius:4px; padding:4px 12px; }"
            "QPushButton:hover { background:#2563eb; }"
        )
        self._got_it_btn.clicked.connect(self._on_got_it)
        btn_row.addStretch()
        btn_row.addWidget(self._got_it_btn)
        layout.addLayout(btn_row)

        self.setStyleSheet(
            "QWidget { background: rgba(15,23,42,0.95); border: 1px solid rgba(99,102,241,0.4); border-radius: 8px; }"
        )
        self._position_window()

    def _position_window(self) -> None:
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        self.move(geo.right() - self.width() - 20, geo.bottom() - 200)

    def show_message(self, message: str, importance: str, reason: str) -> None:
        self._msg_label.setText(message)
        self._reason_label.setText(f"Why: {reason}")
        self._got_it_btn.setVisible(importance == "high")
        self.adjustSize()
        self.show()
        if importance != "high":
            self._auto_timer.start(self._config.auto_dismiss_seconds * 1000)

    def hide_message(self) -> None:
        self._auto_timer.stop()
        self.hide()
        self.dismissed.emit()

    def _on_got_it(self) -> None:
        if self._on_engage:
            self._on_engage()
        self.hide_message()
