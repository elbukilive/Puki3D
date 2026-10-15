"""Project material library and compact material editor."""

from __future__ import annotations

import base64
from copy import deepcopy
import mimetypes
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import (
    QAbstractItemView, QColorDialog, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QDoubleSpinBox, QVBoxLayout, QWidget,
)


DEFAULT_MATERIALS = (
    ("Pintura automotriz", "#1967C9", 0.82, 0.24, 1.0),
    ("Vidrio oscuro", "#182A38", 0.12, 0.12, 0.62),
    ("Caucho", "#17191C", 0.0, 0.88, 1.0),
    ("Metal", "#929AA4", 0.72, 0.34, 1.0),
    ("Plástico", "#30343A", 0.0, 0.62, 1.0),
    ("Cromado", "#D7DDE4", 1.0, 0.12, 1.0),
)


def _new_material(name: str, color: str, metallic: float, roughness: float, opacity: float) -> dict:
    return {
        "id": str(uuid4()), "name": name, "base_color": color,
        "metallic": metallic, "roughness": roughness, "opacity": opacity,
        "texture_id": None,
        "uv_transform": {"scale": [1.0, 1.0], "rotation": 0.0, "offset": [0.0, 0.0]},
    }


def ensure_material_library(project_data: dict) -> list[dict]:
    """Add starter swatches to new or legacy projects without replacing user data."""
    materials = project_data.setdefault("materials", [])
    if not isinstance(materials, list):
        materials = project_data["materials"] = []
    if not materials:
        materials.extend(_new_material(*values) for values in DEFAULT_MATERIALS)
    project_data.setdefault("textures", [])
    if not isinstance(project_data["textures"], list):
        project_data["textures"] = []
    return materials


def material_map(project_data: dict) -> dict[str, dict]:
    return {material.get("id"): material for material in ensure_material_library(project_data)
            if material.get("id")}


def material_color(material: dict | None, fallback=(0.55, 0.61, 0.69)) -> tuple[float, float, float]:
    if not material:
        return fallback
    color = QColor(str(material.get("base_color", "#8C9BAF")))
    if not color.isValid():
        return fallback
    return color.redF(), color.greenF(), color.blueF()


def import_texture(project_data: dict, path: Path) -> dict:
    """Embed an imported raster in the .puki3d JSON so it cannot lose its path."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Importa una imagen PNG o JPG.")
    image = QImage(str(path))
    if image.isNull():
        raise ValueError("No se pudo leer la imagen seleccionada.")
    if image.width() > 8192 or image.height() > 8192:
        raise ValueError("La textura no puede superar 8192 × 8192 píxeles.")
    mime = mimetypes.guess_type(path.name)[0] or ("image/png" if suffix == ".png" else "image/jpeg")
    raw = path.read_bytes()
    if len(raw) > 24 * 1024 * 1024:
        raise ValueError("La textura no puede superar 24 MB.")
    asset = {"id": str(uuid4()), "name": path.name, "mime_type": mime,
             "data_base64": base64.b64encode(raw).decode("ascii")}
    project_data.setdefault("textures", []).append(asset)
    return asset


class MaterialEditorDialog(QDialog):
    """Edit the project library; accepted edits are shared by every assigned node."""

    def __init__(self, project_data: dict, parent=None) -> None:
        super().__init__(parent)
        self.project_data = project_data
        ensure_material_library(project_data)
        self._original = deepcopy(project_data.get("materials", []))
        self._original_textures = deepcopy(project_data.get("textures", []))
        self._deleted_ids: set[str] = set()
        self.current: dict | None = None
        self.setWindowTitle("Biblioteca de materiales")
        self.resize(770, 520)
        root = QVBoxLayout(self)
        body = QHBoxLayout()
        root.addLayout(body, 1)
        self.items = QListWidget()
        self.items.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.items.currentRowChanged.connect(self._select_material)
        body.addWidget(self.items, 1)
        editor = QWidget()
        form = QFormLayout(editor)
        body.addWidget(editor, 2)
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._name_changed)
        form.addRow("Nombre", self.name)
        self.color_button = QPushButton("Elegir color…")
        self.color_button.clicked.connect(self._choose_color)
        form.addRow("Color base", self.color_button)
        self.metallic = self._spin(0.0, 1.0, 0.05)
        self.roughness = self._spin(0.04, 1.0, 0.05)
        self.opacity = self._spin(0.0, 1.0, 0.05)
        form.addRow("Metallic", self.metallic)
        form.addRow("Roughness", self.roughness)
        form.addRow("Opacidad", self.opacity)
        self.texture_label = QLabel("Sin textura")
        self.texture_button = QPushButton("Importar PNG/JPG…")
        self.texture_button.clicked.connect(self._import_texture)
        form.addRow("Textura", self.texture_label)
        form.addRow("", self.texture_button)
        self.uv_scale_u = self._spin(0.01, 1000.0, 0.1)
        self.uv_scale_v = self._spin(0.01, 1000.0, 0.1)
        self.uv_rotation = self._spin(-3600.0, 3600.0, 5.0)
        self.uv_offset_u = self._spin(-1000.0, 1000.0, 0.05)
        self.uv_offset_v = self._spin(-1000.0, 1000.0, 0.05)
        form.addRow("Escala U", self.uv_scale_u)
        form.addRow("Escala V", self.uv_scale_v)
        form.addRow("Rotación UV", self.uv_rotation)
        form.addRow("Posición U", self.uv_offset_u)
        form.addRow("Posición V", self.uv_offset_v)
        for field, key in ((self.metallic, "metallic"), (self.roughness, "roughness"), (self.opacity, "opacity")):
            field.valueChanged.connect(lambda value, field_key=key: self._set_value(field_key, value))
        for field, group, axis in ((self.uv_scale_u, "scale", 0), (self.uv_scale_v, "scale", 1),
                                   (self.uv_offset_u, "offset", 0), (self.uv_offset_v, "offset", 1)):
            field.valueChanged.connect(lambda value, g=group, a=axis: self._set_uv(g, a, value))
        self.uv_rotation.valueChanged.connect(lambda value: self._set_uv_scalar("rotation", value))
        buttons = QHBoxLayout()
        root.addLayout(buttons)
        for title, callback in (("Nuevo", self.add_material), ("Duplicar", self.duplicate_material),
                                ("Eliminar", self.delete_material)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        buttons.addWidget(box)
        self._refresh_list()

    @staticmethod
    def _spin(minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(3)
        field.setSingleStep(step)
        field.setKeyboardTracking(False)
        return field

    def _refresh_list(self, selected_id: str | None = None) -> None:
        materials = ensure_material_library(self.project_data)
        self.items.blockSignals(True)
        self.items.clear()
        row = 0
        for index, material in enumerate(materials):
            item = QListWidgetItem(material.get("name", "Material"))
            item.setData(Qt.ItemDataRole.UserRole, material.get("id"))
            self.items.addItem(item)
            if material.get("id") == selected_id:
                row = index
        self.items.blockSignals(False)
        if materials:
            self.items.setCurrentRow(row)

    def _select_material(self, row: int) -> None:
        materials = ensure_material_library(self.project_data)
        self.current = materials[row] if 0 <= row < len(materials) else None
        self._populate()

    def _populate(self) -> None:
        enabled = self.current is not None
        for widget in (self.name, self.color_button, self.metallic, self.roughness, self.opacity,
                       self.texture_button, self.uv_scale_u, self.uv_scale_v, self.uv_rotation,
                       self.uv_offset_u, self.uv_offset_v):
            widget.setEnabled(enabled)
        if not enabled:
            return
        material = self.current
        self.name.setText(material.get("name", "Material"))
        color = QColor(str(material.get("base_color", "#8C9BAF")))
        self.color_button.setStyleSheet(f"background-color: {color.name()};")
        for field, key in ((self.metallic, "metallic"), (self.roughness, "roughness"), (self.opacity, "opacity")):
            field.blockSignals(True); field.setValue(float(material.get(key, 0.0))); field.blockSignals(False)
        texture = next((asset for asset in self.project_data.get("textures", [])
                        if asset.get("id") == material.get("texture_id")), None)
        self.texture_label.setText(texture.get("name", "Textura") if texture else "Sin textura")
        transform = material.setdefault("uv_transform", {})
        for field, group, axis in ((self.uv_scale_u, "scale", 0), (self.uv_scale_v, "scale", 1),
                                   (self.uv_offset_u, "offset", 0), (self.uv_offset_v, "offset", 1)):
            value = transform.get(group, [1.0, 1.0] if group == "scale" else [0.0, 0.0])[axis]
            field.blockSignals(True); field.setValue(float(value)); field.blockSignals(False)
        self.uv_rotation.blockSignals(True); self.uv_rotation.setValue(float(transform.get("rotation", 0.0))); self.uv_rotation.blockSignals(False)

    def _name_changed(self) -> None:
        if self.current:
            self.current["name"] = self.name.text().strip() or "Material"
            item = self.items.currentItem()
            if item:
                item.setText(self.current["name"])

    def _set_value(self, key: str, value: float) -> None:
        if self.current:
            self.current[key] = float(value)

    def _set_uv(self, group: str, axis: int, value: float) -> None:
        if self.current:
            transform = self.current.setdefault("uv_transform", {})
            values = list(transform.get(group, [1.0, 1.0] if group == "scale" else [0.0, 0.0]))
            values[axis] = float(value)
            transform[group] = values

    def _set_uv_scalar(self, key: str, value: float) -> None:
        if self.current:
            self.current.setdefault("uv_transform", {})[key] = float(value)

    def _choose_color(self) -> None:
        if not self.current:
            return
        initial = QColor(str(self.current.get("base_color", "#8C9BAF")))
        color = QColorDialog.getColor(initial, self, "Color base del material")
        if color.isValid():
            self.current["base_color"] = color.name().upper()
            self._populate()

    def _import_texture(self) -> None:
        if not self.current:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Importar textura", "", "Imágenes (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            asset = import_texture(self.project_data, Path(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "No se pudo importar la textura", str(exc))
            return
        self.current["texture_id"] = asset["id"]
        self._populate()

    def add_material(self) -> None:
        material = _new_material("Nuevo material", "#8C9BAF", 0.0, 0.5, 1.0)
        ensure_material_library(self.project_data).append(material)
        self._refresh_list(material["id"])

    def duplicate_material(self) -> None:
        if not self.current:
            return
        clone = deepcopy(self.current)
        clone["id"] = str(uuid4())
        clone["name"] = f"{clone.get('name', 'Material')} copia"
        ensure_material_library(self.project_data).append(clone)
        self._refresh_list(clone["id"])

    def delete_material(self) -> None:
        if not self.current:
            return
        material_id = self.current.get("id")
        self.project_data["materials"] = [item for item in self.project_data["materials"]
                                           if item.get("id") != material_id]
        if material_id:
            self._deleted_ids.add(material_id)
        self.current = None
        if not self.project_data["materials"]:
            self.add_material()
        else:
            self._refresh_list()

    def reject(self) -> None:
        self.project_data["materials"] = self._original
        self.project_data["textures"] = self._original_textures
        super().reject()

    def accept(self) -> None:
        for scene in self.project_data.get("scenes", []):
            for node in scene.get("nodes", []):
                if node.get("material_id") in self._deleted_ids:
                    node.pop("material_id", None)
                geometry = node.get("geometry", {})
                geometry["face_materials"] = [None if value in self._deleted_ids else value
                                               for value in geometry.get("face_materials", [])]
        used_textures = {material.get("texture_id") for material in self.project_data.get("materials", [])}
        self.project_data["textures"] = [texture for texture in self.project_data.get("textures", [])
                                          if texture.get("id") in used_textures]
        super().accept()

