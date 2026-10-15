"""Temporary, non-physical vehicle motion controls for viewport inspection."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFormLayout, QLabel, QPushButton, QSlider, QVBoxLayout


class VehiclePreviewDialog(QDialog):
    def __init__(self, viewport, vehicle_system: dict, parent=None) -> None:
        super().__init__(parent)
        self.viewport = viewport
        self.vehicle_system = vehicle_system
        self.setWindowTitle("Vehicle Preview")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMinimumWidth(360)
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Prueba visual temporal; no cambia ni guarda las transformaciones."))
        form = QFormLayout()
        root.addLayout(form)
        steering_max = float(vehicle_system.get("steering", {}).get("max_angle", 35.0))
        suspension = float(vehicle_system.get("suspension", {}).get("travel", 0.25))
        self.steering_value = QLabel("0°")
        self.steering_slider = self._slider(-100, 100)
        form.addRow("Dirección", self.steering_slider)
        form.addRow("Ángulo", self.steering_value)
        self.rotation_value = QLabel("0°")
        self.rotation_slider = self._slider(-360, 360)
        form.addRow("Giro de ruedas", self.rotation_slider)
        form.addRow("Rotación", self.rotation_value)
        self.suspension_value = QLabel("0.000 m")
        self.suspension_slider = self._slider(-100, 100)
        form.addRow("Suspensión", self.suspension_slider)
        form.addRow("Desplazamiento", self.suspension_value)
        self.steering_slider.valueChanged.connect(lambda value: self._set("steering", value, steering_max))
        self.rotation_slider.valueChanged.connect(lambda value: self._set("wheel_rotation", value, 1.0))
        self.suspension_slider.valueChanged.connect(lambda value: self._set("suspension", value, suspension))
        buttons = QFormLayout()
        root.addLayout(buttons)
        reset = QPushButton("Restablecer prueba")
        reset.clicked.connect(self.reset_preview)
        buttons.addRow(reset)
        self._set_preview()

    @staticmethod
    def _slider(minimum: int, maximum: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(0)
        return slider

    def _set(self, channel: str, value: int, scale: float) -> None:
        normalized = value / 100.0 if channel != "wheel_rotation" else float(value)
        setattr(self, f"_{channel}", normalized)
        label = {"steering": self.steering_value, "wheel_rotation": self.rotation_value,
                 "suspension": self.suspension_value}[channel]
        suffix = "°" if channel != "suspension" else " m"
        label.setText(f"{normalized * scale:.3f}{suffix}" if channel == "suspension" else f"{normalized * scale:.1f}{suffix}")
        self._set_preview()

    def _set_preview(self) -> None:
        self.viewport.set_vehicle_preview(
            True,
            steering=float(getattr(self, "_steering", 0.0)),
            wheel_rotation=float(getattr(self, "_wheel_rotation", 0.0)),
            suspension=float(getattr(self, "_suspension", 0.0)),
        )

    def reset_preview(self) -> None:
        for slider in (self.steering_slider, self.rotation_slider, self.suspension_slider):
            slider.setValue(0)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.viewport.set_vehicle_preview(False)
        super().closeEvent(event)
