"""Interactive 2D UV island editor for a selected scene object."""

from __future__ import annotations

import base64

import numpy as np
from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)

from .scene import auto_unwrap_geometry, ensure_editable_geometry, planar_project_geometry, transform_face_uvs


class UVCanvas(QWidget):
    selectionChanged = Signal(object)
    geometryChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)
        self.geometry: dict | None = None
        self.selected_faces: set[int] = set()
        self.background_image: QImage | None = None
        self._drag_start: np.ndarray | None = None
        self._drag_uvs: dict[int, list] = {}
        self._moved = False

    def set_geometry(self, geometry: dict | None, selected_faces=None) -> None:
        self.geometry = geometry
        self.selected_faces = set(selected_faces or ())
        self._drag_start = None
        self._drag_uvs = {}
        self.update()

    def set_texture(self, image: QImage | None) -> None:
        self.background_image = image
        self.update()

    def set_selection(self, selected_faces) -> None:
        self.selected_faces = set(selected_faces or ())
        self.update()

    def _tile_rect(self):
        size = max(1, min(self.width() - 44, self.height() - 44))
        left = (self.width() - size) * 0.5
        top = (self.height() - size) * 0.5
        return left, top, size

    def _to_screen(self, uv) -> QPointF:
        left, top, size = self._tile_rect()
        return QPointF(left + float(uv[0]) * size, top + (1.0 - float(uv[1])) * size)

    def _to_uv(self, point) -> np.ndarray:
        left, top, size = self._tile_rect()
        return np.array([(point.x() - left) / size, 1.0 - (point.y() - top) / size], dtype=float)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111318"))
        left, top, size = self._tile_rect()
        cell = size / 12.0
        for y in range(12):
            for x in range(12):
                shade = QColor("#282c33" if (x + y) % 2 else "#20242a")
                painter.fillRect(int(left + x * cell), int(top + y * cell), int(cell + 1), int(cell + 1), shade)
        target = (int(left), int(top), int(size), int(size))
        if self.background_image and not self.background_image.isNull():
            painter.setOpacity(0.84)
            painter.drawImage(QRect(*target), self.background_image)
            painter.setOpacity(1.0)
        painter.setPen(QPen(QColor("#8b93a0"), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(*target)
        for index in range(1, 10):
            coordinate = index / 10.0
            x = left + size * coordinate
            y = top + size * coordinate
            painter.setPen(QPen(QColor(125, 133, 147, 55), 1))
            painter.drawLine(int(x), int(top), int(x), int(top + size))
            painter.drawLine(int(left), int(y), int(left + size), int(y))
        if not self.geometry:
            painter.end()
            return
        faces = self.geometry.get("faces", [])
        face_uvs = self.geometry.get("face_uvs", [])
        palette = ("#56a0ef", "#df9e4a", "#8bc985", "#c18ee5", "#56c1b4", "#df7b83")
        for face_id, uv_loop in enumerate(face_uvs):
            if face_id >= len(faces) or len(uv_loop) < 3:
                continue
            polygon = QPolygonF([self._to_screen(uv) for uv in uv_loop])
            selected = face_id in self.selected_faces
            painter.setPen(QPen(QColor("#e0ff53" if selected else palette[face_id % len(palette)]),
                                2.2 if selected else 1.3))
            painter.setBrush(QColor(224, 255, 83, 58) if selected else QColor(40, 90, 145, 52))
            painter.drawPolygon(polygon)
            if polygon.boundingRect().width() > 18 and polygon.boundingRect().height() > 14:
                painter.setPen(QColor("#f3f4f7"))
                painter.drawText(polygon.boundingRect(), Qt.AlignmentFlag.AlignCenter, str(face_id))
        painter.end()

    def _face_at(self, point) -> int | None:
        if not self.geometry:
            return None
        faces = self.geometry.get("faces", [])
        face_uvs = self.geometry.get("face_uvs", [])
        limit = min(len(faces), len(face_uvs))
        ordered = sorted((face_id for face_id in self.selected_faces if 0 <= face_id < limit), reverse=True)
        ordered.extend(face_id for face_id in range(limit - 1, -1, -1) if face_id not in self.selected_faces)
        for face_id in ordered:
            if len(face_uvs[face_id]) < 3:
                continue
            polygon = QPolygonF([self._to_screen(uv) for uv in face_uvs[face_id]])
            if polygon.containsPoint(point, Qt.FillRule.OddEvenFill):
                return face_id
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() != Qt.MouseButton.LeftButton or not self.geometry:
            return super().mousePressEvent(event)
        face_id = self._face_at(event.position())
        if face_id is None:
            self.selected_faces.clear()
            self.selectionChanged.emit(set())
            self.update()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if face_id in self.selected_faces:
                self.selected_faces.remove(face_id)
            else:
                self.selected_faces.add(face_id)
        elif face_id not in self.selected_faces or len(self.selected_faces) != 1:
            self.selected_faces = {face_id}
        self.selectionChanged.emit(set(self.selected_faces))
        if face_id in self.selected_faces:
            self._drag_start = self._to_uv(event.position())
            face_uvs = self.geometry.get("face_uvs", [])
            self._drag_uvs = {index: [list(uv) for uv in face_uvs[index]] for index in self.selected_faces}
            self._moved = False
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._drag_start is None or not self.geometry or not (event.buttons() & Qt.MouseButton.LeftButton):
            return super().mouseMoveEvent(event)
        delta = self._to_uv(event.position()) - self._drag_start
        self._moved |= float(np.linalg.norm(delta)) > 1e-6
        face_uvs = self.geometry.setdefault("face_uvs", [])
        for face_id, uvs in self._drag_uvs.items():
            face_uvs[face_id] = (np.asarray(uvs, dtype=float) + delta).tolist()
        self.update()
        self.geometryChanged.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            if self._moved:
                self.geometryChanged.emit()
            self._drag_start = None
            self._drag_uvs = {}
            self._moved = False
            return
        super().mouseReleaseEvent(event)

    def apply_transform(self, scale: float, rotation: float, offset_u: float, offset_v: float) -> None:
        if not self.geometry or not self.selected_faces:
            return
        transformed = transform_face_uvs(self.geometry, self.selected_faces, (offset_u, offset_v), scale, rotation)
        self.geometry.update(transformed)
        self.update()
        self.geometryChanged.emit()

    def unwrap(self, method: str) -> None:
        if not self.geometry:
            return
        selected = self.selected_faces or None
        if method == "auto":
            transformed = auto_unwrap_geometry(self.geometry, selected)
        else:
            axis = {"planar_x": 0, "planar_y": 1, "planar_z": 2}[method]
            transformed = planar_project_geometry(self.geometry, axis, selected)
        self.geometry.update(transformed)
        self.update()
        self.geometryChanged.emit()


class UVEditorDialog(QDialog):
    """Modeless UV editor synchronized with the active 3D object's face selection."""

    faceSelectionChanged = Signal(object)
    uvChanged = Signal()

    def __init__(self, project_data: dict, node: dict | None, parent=None) -> None:
        super().__init__(parent)
        self.project_data = project_data
        self.node = node
        self.setWindowTitle("Editor UV — Puki3D")
        self.resize(760, 620)
        root = QVBoxLayout(self)
        self.object_label = QLabel()
        self.object_label.setObjectName("panelHeading")
        root.addWidget(self.object_label)
        self.canvas = UVCanvas()
        self.canvas.selectionChanged.connect(self._on_canvas_selection_changed)
        self.canvas.geometryChanged.connect(self.uvChanged)
        root.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        transform = QFormLayout()
        self.scale = self._spin(0.01, 100.0, 1.0, 0.1)
        self.rotation = self._spin(-3600.0, 3600.0, 0.0, 5.0)
        self.offset_u = self._spin(-1000.0, 1000.0, 0.0, 0.05)
        self.offset_v = self._spin(-1000.0, 1000.0, 0.0, 0.05)
        transform.addRow("Escala", self.scale)
        transform.addRow("Rotación (°)", self.rotation)
        transform.addRow("Mover U", self.offset_u)
        transform.addRow("Mover V", self.offset_v)
        controls.addLayout(transform)
        actions = QVBoxLayout()
        self.apply_button = QPushButton("Aplicar transformación")
        self.apply_button.clicked.connect(self.apply_transform)
        self.apply_button.setEnabled(False)
        actions.addWidget(self.apply_button)
        unwrap_row = QHBoxLayout()
        self.unwrap_method = QComboBox()
        self.unwrap_method.addItem("Automático", "auto")
        self.unwrap_method.addItem("Planar X", "planar_x")
        self.unwrap_method.addItem("Planar Y", "planar_y")
        self.unwrap_method.addItem("Planar Z", "planar_z")
        unwrap_row.addWidget(self.unwrap_method, 1)
        unwrap_button = QPushButton("Proyectar")
        unwrap_button.clicked.connect(lambda: self.canvas.unwrap(self.unwrap_method.currentData()))
        unwrap_row.addWidget(unwrap_button)
        actions.addLayout(unwrap_row)
        actions.addStretch(1)
        controls.addLayout(actions, 1)
        root.addLayout(controls)
        self.set_node(node)

    @staticmethod
    def _spin(minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(3)
        field.setSingleStep(step)
        field.setValue(value)
        field.setKeyboardTracking(False)
        return field

    def set_node(self, node: dict | None, selected_faces=None) -> None:
        self.node = node
        self.object_label.setText(node.get("name", "Sin objeto activo") if node else "Sin objeto activo")
        geometry = ensure_editable_geometry(node.setdefault("geometry", {})) if node else None
        self.canvas.set_geometry(geometry, selected_faces)
        self.canvas.setEnabled(node is not None)
        self.apply_button.setEnabled(bool(self.canvas.selected_faces))
        self._refresh_texture()

    def set_selected_faces(self, selected_faces) -> None:
        self.canvas.set_selection(selected_faces)
        self.apply_button.setEnabled(bool(self.canvas.selected_faces))
        self._refresh_texture()

    def _on_canvas_selection_changed(self, selected_faces) -> None:
        self.apply_button.setEnabled(bool(selected_faces))
        self.faceSelectionChanged.emit(selected_faces)
        self._refresh_texture()

    def _refresh_texture(self) -> None:
        if not self.node:
            self.canvas.set_texture(None)
            return
        geometry = self.node.get("geometry", {})
        face_materials = geometry.get("face_materials", [])
        material_ids = [face_materials[index] if index < len(face_materials) else None
                        for index in self.canvas.selected_faces]
        material_ids.append(self.node.get("material_id"))
        materials = {material.get("id"): material for material in self.project_data.get("materials", [])}
        material = next((materials.get(material_id) for material_id in material_ids
                         if material_id in materials and materials[material_id].get("texture_id")), None)
        texture_id = material.get("texture_id") if material else None
        texture = next((asset for asset in self.project_data.get("textures", [])
                        if asset.get("id") == texture_id), None)
        try:
            raw = base64.b64decode(texture.get("data_base64", ""), validate=True) if texture else b""
            image = QImage.fromData(raw) if raw else None
        except (ValueError, TypeError):
            image = None
        self.canvas.set_texture(image)

    def apply_transform(self) -> None:
        self.canvas.apply_transform(self.scale.value(), self.rotation.value(), self.offset_u.value(), self.offset_v.value())
        self.scale.setValue(1.0)
        self.rotation.setValue(0.0)
        self.offset_u.setValue(0.0)
        self.offset_v.setValue(0.0)
