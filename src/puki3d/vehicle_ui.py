"""Focused editor for a project's vehicle structure."""

from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .scene import node_matrix
from .vehicle import (
    COMPONENTS, MOUNT_TYPES, ROLES, VEHICLE_TYPES, auto_classify,
    axle_dimensions, create_mount_point, ensure_vehicle_system, set_node_classification,
    vehicle_diagnostics,
)


class VehicleSystemDialog(QDialog):
    def __init__(self, project_data: dict, nodes: list[dict], selected_node_id: str | None = None,
                 selected_mount_id: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sistema de vehículo")
        self.resize(1180, 720)
        self.project_data = copy.deepcopy(project_data)
        self.nodes = copy.deepcopy(nodes)
        self.meta = ensure_vehicle_system(self.project_data, self.nodes)
        self.selected_node_id = selected_node_id
        self.selected_mount_id = selected_mount_id
        root = QVBoxLayout(self)
        title = QLabel("Estructura funcional del vehículo")
        title.setObjectName("dialogTitle")
        root.addWidget(title)
        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        summary = QWidget()
        form = QFormLayout(summary)
        self.name_field = QLineEdit(str(self.meta.get("name", "")))
        form.addRow("Nombre del vehículo", self.name_field)
        self.type_field = QComboBox()
        self.type_field.addItems(VEHICLE_TYPES)
        self._set_combo(self.type_field, self.meta.get("type", "Pickup"))
        form.addRow("Tipo", self.type_field)
        self.chassis_field = QComboBox()
        self.chassis_field.addItem("Sin asignar", None)
        for node in self.nodes:
            self.chassis_field.addItem(node.get("name", "Objeto"), node.get("id"))
        self._set_combo_data(self.chassis_field, self.meta.get("chassis_node_id"))
        form.addRow("Referencia de chasis", self.chassis_field)
        self.max_steering = self._number(float(self.meta["steering"].get("max_angle", 35.0)), 0.0, 80.0)
        self.max_steering.setValue(float(self.meta["steering"].get("max_angle", 35.0)))
        self.max_steering.setSuffix("°")
        form.addRow("Ángulo máximo de dirección", self.max_steering)
        self.suspension_travel = self._number(float(self.meta["suspension"].get("travel", 0.25)), 0.0, 2.0)
        self.suspension_travel.setValue(float(self.meta["suspension"].get("travel", 0.25)))
        self.suspension_travel.setSuffix(" m")
        form.addRow("Recorrido de suspensión", self.suspension_travel)
        form.addRow(QLabel("Obligatorios: Body, Cabin, Chassis y cuatro Wheels."))
        form.addRow(QLabel("La referencia de chasis puede compartir la pieza con el componente Chassis."))
        tabs.addTab(summary, "Vehículo")

        parts_tab = QWidget()
        parts_layout = QVBoxLayout(parts_tab)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filtrar piezas"))
        self.filter_field = QLineEdit()
        self.filter_field.setPlaceholderText("Nombre de pieza")
        filter_row.addWidget(self.filter_field, 1)
        self.suggest_button = QPushButton("Sugerir por nombre")
        self.suggest_button.setToolTip("Clasifica los nombres comunes; puedes corregir cualquier fila.")
        filter_row.addWidget(self.suggest_button)
        parts_layout.addLayout(filter_row)
        self.parts_table = QTableWidget(0, 3)
        self.parts_table.setHorizontalHeaderLabels(["Pieza", "Componente", "Rol"])
        self.parts_table.setColumnWidth(0, 390)
        self.parts_table.setColumnWidth(1, 170)
        self.parts_table.setColumnWidth(2, 220)
        self.parts_table.setAlternatingRowColors(True)
        self.parts_table.setSortingEnabled(False)
        parts_layout.addWidget(self.parts_table, 1)
        tabs.addTab(parts_tab, "Piezas")

        wheels_tab = QWidget()
        wheels_layout = QVBoxLayout(wheels_tab)
        wheels_layout.addWidget(QLabel("Selecciona el rol wheel en Piezas para registrar un objeto como rueda."))
        self.wheels_table = QTableWidget(0, 13)
        self.wheels_table.setHorizontalHeaderLabels(["Rueda", "X", "Y", "Z", "Rot X", "Rot Y", "Rot Z", "Lado", "Eje", "Radio", "Ancho", "Dirige", "Motriz"])
        self.wheels_table.setColumnWidth(0, 190)
        for col in range(1, 13):
            self.wheels_table.setColumnWidth(col, 72)
        self.wheels_table.setAlternatingRowColors(True)
        wheels_layout.addWidget(self.wheels_table, 1)
        self.axle_metrics = QLabel()
        wheels_layout.addWidget(self.axle_metrics)
        tabs.addTab(wheels_tab, "Ruedas")

        mounts_tab = QWidget()
        mounts_layout = QVBoxLayout(mounts_tab)
        mounts_layout.addWidget(QLabel("Coordenadas en metros; al vincular una pieza pasan a ser locales a esa pieza."))
        self.mounts_table = QTableWidget(0, 6)
        self.mounts_table.setHorizontalHeaderLabels(["Nombre", "Tipo", "Pieza vinculada", "X", "Y", "Z"])
        self.mounts_table.setColumnWidth(0, 230)
        self.mounts_table.setColumnWidth(1, 125)
        self.mounts_table.setColumnWidth(2, 230)
        for col in range(3, 6):
            self.mounts_table.setColumnWidth(col, 110)
        self.mounts_table.itemChanged.connect(self._mount_name_changed)
        mounts_layout.addWidget(self.mounts_table, 1)
        mount_buttons = QHBoxLayout()
        add_mount = QPushButton("Añadir mount point")
        add_mount.clicked.connect(self.add_mount_point)
        remove_mount = QPushButton("Eliminar seleccionado")
        remove_mount.clicked.connect(self.remove_mount_point)
        mount_buttons.addWidget(add_mount)
        mount_buttons.addWidget(remove_mount)
        mount_buttons.addStretch()
        mounts_layout.addLayout(mount_buttons)
        tabs.addTab(mounts_tab, "Mount points")

        diagnostics_tab = QWidget()
        diagnostic_layout = QVBoxLayout(diagnostics_tab)
        diagnostic_layout.addWidget(QLabel("Diagnósticos de estructura"))
        self.diagnostics_list = QListWidget()
        diagnostic_layout.addWidget(self.diagnostics_list, 1)
        tabs.addTab(diagnostics_tab, "Validación")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Aplicar")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._build_parts_table()
        self._build_wheels_table()
        self._build_mounts_table()
        self.filter_field.textChanged.connect(self._filter_parts)
        self.suggest_button.clicked.connect(self._suggest_classifications)
        self.name_field.textChanged.connect(lambda _value: self._update_diagnostics())
        self.type_field.currentTextChanged.connect(lambda _value: self._update_diagnostics())
        self.chassis_field.currentIndexChanged.connect(lambda _value: self._update_diagnostics())
        self.max_steering.valueChanged.connect(lambda value: self._set_vehicle_measure("steering", "max_angle", value))
        self.suspension_travel.valueChanged.connect(lambda value: self._set_vehicle_measure("suspension", "travel", value))
        self._update_diagnostics()

    def _set_combo(self, combo: QComboBox, value: str) -> None:
        index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_combo_data(self, combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _build_parts_table(self) -> None:
        self.parts_table.setRowCount(len(self.nodes))
        for row, node in enumerate(self.nodes):
            item = QTableWidgetItem(node.get("name", "Objeto"))
            item.setData(Qt.ItemDataRole.UserRole, node.get("id"))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.parts_table.setItem(row, 0, item)
            info = node.get("vehicle", {})
            for column, options, current in (
                (1, COMPONENTS, info.get("component")),
                (2, ROLES, info.get("role")),
            ):
                combo = QComboBox()
                combo.addItem("Sin clasificar", None)
                for option in options:
                    combo.addItem(option, option)
                self._set_combo_data(combo, current)
                combo.currentIndexChanged.connect(lambda _index, row=row: self._part_changed(row))
                self.parts_table.setCellWidget(row, column, combo)

    def _node_by_id(self, node_id: str) -> dict | None:
        return next((node for node in self.nodes if node.get("id") == node_id), None)

    def _part_changed(self, row: int) -> None:
        node_id = self.parts_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        node = self._node_by_id(node_id)
        if not node:
            return
        component_combo = self.parts_table.cellWidget(row, 1)
        role_combo = self.parts_table.cellWidget(row, 2)
        set_node_classification(node, component_combo.currentData(), role_combo.currentData())
        if node.get("vehicle", {}).get("role") == "wheel":
            self._build_wheels_table()
        else:
            self._build_wheels_table()
        if self.chassis_field.currentData() == node_id:
            if node.get("vehicle", {}).get("role") != "chassis_part":
                self.chassis_field.setCurrentIndex(0)
        self._update_diagnostics()

    def _suggest_classifications(self) -> None:
        auto_classify(self.nodes)
        self._build_parts_table()
        self._build_wheels_table()
        chassis = next((node for node in self.nodes if node.get("vehicle", {}).get("component") == "Chassis"), None)
        if chassis:
            self._set_combo_data(self.chassis_field, chassis.get("id"))
        self._update_diagnostics()

    def _filter_parts(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self.parts_table.rowCount()):
            self.parts_table.setRowHidden(row, needle not in self.parts_table.item(row, 0).text().casefold())

    @staticmethod
    def _number(value: float, minimum: float = -100000.0, maximum: float = 100000.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        spin.setKeyboardTracking(False)
        return spin

    def _build_wheels_table(self) -> None:
        wheels = [node for node in self.nodes if node.get("vehicle", {}).get("role") == "wheel"]
        self.wheels_table.setRowCount(len(wheels))
        for row, node in enumerate(wheels):
            item = QTableWidgetItem(node.get("name", "Rueda"))
            item.setData(Qt.ItemDataRole.UserRole, node.get("id"))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.wheels_table.setItem(row, 0, item)
            info = node["vehicle"].setdefault("wheel", {})
            transform = node.setdefault("transform", {})
            transform.setdefault("position", [0.0, 0.0, 0.0])
            transform.setdefault("rotation", [0.0, 0.0, 0.0])
            info["position"] = list(transform["position"])
            info["rotation"] = list(transform["rotation"])
            for col, value in enumerate(info["position"], start=1):
                spin = self._number(float(value))
                spin.setValue(float(value))
                spin.valueChanged.connect(lambda new_value, row=row, axis=col - 1: self._wheel_position_changed(row, axis, new_value))
                self.wheels_table.setCellWidget(row, col, spin)
            for col, value in enumerate(info["rotation"], start=4):
                spin = self._number(float(value), -360.0, 360.0)
                spin.setValue(float(value))
                spin.setSuffix("°")
                spin.valueChanged.connect(lambda new_value, row=row, axis=col - 4: self._wheel_rotation_changed(row, axis, new_value))
                self.wheels_table.setCellWidget(row, col, spin)
            side = QComboBox()
            side.addItem("Izquierda", "left")
            side.addItem("Derecha", "right")
            self._set_combo_data(side, info.get("side", "left"))
            side.currentIndexChanged.connect(lambda _value, row=row: self._wheel_field_changed(row, "side", self.wheels_table.cellWidget(row, 7).currentData()))
            self.wheels_table.setCellWidget(row, 7, side)
            axle = QComboBox()
            axle.addItem("Delantero", "front")
            axle.addItem("Trasero", "rear")
            self._set_combo_data(axle, info.get("axle", "front"))
            axle.currentIndexChanged.connect(lambda _value, row=row: self._wheel_field_changed(row, "axle", self.wheels_table.cellWidget(row, 8).currentData()))
            self.wheels_table.setCellWidget(row, 8, axle)
            for col, key in ((9, "radius"), (10, "width")):
                spin = self._number(float(info.get(key, 0.1)), 0.001, 1000.0)
                spin.setValue(float(info.get(key, 0.1)))
                spin.valueChanged.connect(lambda value, row=row, key=key: self._wheel_field_changed(row, key, value))
                self.wheels_table.setCellWidget(row, col, spin)
            for col, key in ((11, "steerable"), (12, "driven")):
                check = QCheckBox()
                check.setChecked(bool(info.get(key, False)))
                check.setToolTip("Permite usar esta rueda en la dirección" if key == "steerable" else "Marca esta rueda como motriz")
                check.stateChanged.connect(lambda state, row=row, key=key: self._wheel_field_changed(row, key, bool(state)))
                self.wheels_table.setCellWidget(row, col, check)
        dimensions = axle_dimensions(self.nodes)
        self._update_axle_metrics(dimensions)

    def _update_axle_metrics(self, dimensions: dict | None = None) -> None:
        dimensions = dimensions or axle_dimensions(self.nodes)
        fmt = lambda value: "—" if value is None else f"{value:.3f} m"
        self.axle_metrics.setText(
            f"Distancia entre ejes: {fmt(dimensions['wheelbase'])}  ·  "
            f"Vía delantera: {fmt(dimensions['front_track'])}  ·  "
            f"Vía trasera: {fmt(dimensions['rear_track'])}"
        )

    def _wheel_node_for_row(self, row: int) -> dict | None:
        item = self.wheels_table.item(row, 0)
        return self._node_by_id(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def _wheel_position_changed(self, row: int, axis: int, value: float) -> None:
        node = self._wheel_node_for_row(row)
        if node:
            position = node.setdefault("transform", {}).setdefault("position", [0.0, 0.0, 0.0])
            position[axis] = value
            wheel = node["vehicle"].setdefault("wheel", {})
            wheel["position"] = list(position)
            wheel["rest_position"] = list(position)
            self._update_axle_metrics()

    def _wheel_rotation_changed(self, row: int, axis: int, value: float) -> None:
        node = self._wheel_node_for_row(row)
        if node:
            rotation = node.setdefault("transform", {}).setdefault("rotation", [0.0, 0.0, 0.0])
            rotation[axis] = value
            node["vehicle"].setdefault("wheel", {})["rotation"] = list(rotation)

    def _wheel_field_changed(self, row: int, key: str, value) -> None:
        node = self._wheel_node_for_row(row)
        if node:
            node["vehicle"].setdefault("wheel", {})[key] = value
            if key in {"side", "axle"}:
                self._update_axle_metrics()
            self._update_diagnostics()

    def _set_vehicle_measure(self, group: str, key: str, value: float) -> None:
        self.meta.setdefault(group, {})[key] = value

    def _build_mounts_table(self) -> None:
        mounts = self.meta.setdefault("mount_points", [])
        self.mounts_table.setRowCount(len(mounts))
        for row, point in enumerate(mounts):
            name = QTableWidgetItem(str(point.get("name", "Mount Point")))
            name.setData(Qt.ItemDataRole.UserRole, point.get("id"))
            self.mounts_table.setItem(row, 0, name)
            kind = QComboBox()
            kind.setEditable(True)
            kind.addItems(MOUNT_TYPES)
            self._set_combo(kind, point.get("type", "Generic"))
            kind.currentTextChanged.connect(lambda value, row=row: self._mount_field_changed(row, "type", value))
            self.mounts_table.setCellWidget(row, 1, kind)
            position = point.setdefault("position", [0.0, 0.0, 0.0])
            target = QComboBox()
            target.addItem("Vehículo", None)
            for node in self.nodes:
                target.addItem(node.get("name", "Objeto"), node.get("id"))
            self._set_combo_data(target, point.get("attached_node_id"))
            target.currentIndexChanged.connect(lambda _value, row=row: self._mount_attachment_changed(row, self.mounts_table.cellWidget(row, 2).currentData()))
            self.mounts_table.setCellWidget(row, 2, target)
            for col, value in enumerate(position, start=3):
                spin = self._number(float(value))
                spin.setValue(float(value))
                spin.valueChanged.connect(lambda new_value, row=row, axis=col - 3: self._mount_position_changed(row, axis, new_value))
                self.mounts_table.setCellWidget(row, col, spin)
            if point.get("id") == self.selected_mount_id:
                self.mounts_table.selectRow(row)

    def _mount_for_row(self, row: int) -> dict | None:
        if row < 0 or row >= len(self.meta.get("mount_points", [])):
            return None
        return self.meta["mount_points"][row]

    def _mount_name_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            point = self._mount_for_row(item.row())
            if point:
                point["name"] = item.text().strip()
                self._update_diagnostics()

    def _mount_field_changed(self, row: int, key: str, value) -> None:
        point = self._mount_for_row(row)
        if point:
            point[key] = value

    def _mount_attachment_changed(self, row: int, node_id: str | None) -> None:
        point = self._mount_for_row(row)
        if not point:
            return
        world = point.setdefault("position", [0.0, 0.0, 0.0])
        old_id = point.get("attached_node_id")
        old_node = self._node_by_id(old_id) if old_id else None
        if old_node:
            import numpy as np
            world = (node_matrix(old_node) @ np.append(world, 1.0))[:3]
        if node_id:
            target = self._node_by_id(node_id)
            if target:
                import numpy as np
                local = (np.linalg.inv(node_matrix(target)) @ np.append(world, 1.0))[:3]
                point["position"] = local.tolist()
        else:
            point["position"] = list(world)
        point["attached_node_id"] = node_id
        for axis, value in enumerate(point["position"]):
            spin = self.mounts_table.cellWidget(row, axis + 3)
            if spin:
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)

    def _mount_position_changed(self, row: int, axis: int, value: float) -> None:
        point = self._mount_for_row(row)
        if point:
            point.setdefault("position", [0.0, 0.0, 0.0])[axis] = value

    def add_mount_point(self) -> None:
        selected = self._node_by_id(self.selected_node_id) if self.selected_node_id else None
        point = create_mount_point(self.meta["mount_points"])
        if selected:
            point["attached_node_id"] = selected.get("id")
        self.meta["mount_points"].append(point)
        self._build_mounts_table()
        self.mounts_table.selectRow(self.mounts_table.rowCount() - 1)
        self._update_diagnostics()

    def remove_mount_point(self) -> None:
        row = self.mounts_table.currentRow()
        if 0 <= row < len(self.meta["mount_points"]):
            del self.meta["mount_points"][row]
            self._build_mounts_table()
            self._update_diagnostics()

    def _update_diagnostics(self) -> None:
        self.meta["name"] = self.name_field.text().strip()
        self.meta["type"] = self.type_field.currentText()
        self.meta["chassis_node_id"] = self.chassis_field.currentData()
        self.diagnostics_list.clear()
        messages = vehicle_diagnostics(self.project_data, self.nodes)
        if not messages:
            self.diagnostics_list.addItem("Estructura válida: componentes obligatorios y cuatro ruedas registrados.")
        else:
            for message in messages:
                self.diagnostics_list.addItem(message)

    def accept(self) -> None:
        self.meta["name"] = self.name_field.text().strip()
        self.meta["type"] = self.type_field.currentText()
        self.meta["chassis_node_id"] = self.chassis_field.currentData()
        super().accept()
