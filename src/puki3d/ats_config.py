"""ATS player-truck preparation metadata and validation (not an exporter)."""

from __future__ import annotations

import copy
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QTabWidget, QVBoxLayout, QWidget,
)

from .vehicle import axle_dimensions, vehicle_diagnostics


ATS_REQUIRED_FALLBACKS = (
    "chassis", "cabin", "engine", "transmission", "f_tire", "f_disc", "f_hub",
    "f_nuts", "r_tire", "r_disc", "r_hub", "r_nuts", "head_light", "paint_job", "interior",
)
SUSPENSION_TYPES = ("simple", "wishbone", "macpherson", "solidaxle")
ENGINE_TYPES = ("diesel", "petrol", "electric")
CAMERA_FIELDS = (
    ("behind_camera", "Cámara exterior"),
    ("interior_camera", "Cámara interior"),
    ("bumper_camera", "Cámara de defensa"),
    ("wheel_camera", "Cámara de rueda"),
)
_UNIT_ID = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_DEPENDENCY_ID = re.compile(r"^[a-z0-9_]+$")


def ensure_ats_config(project_data: dict, nodes: list[dict]) -> dict:
    """Create draft ATS configuration in project data, retaining unknown future fields."""
    config = project_data.setdefault("ats_config", {})
    config.setdefault("status", "prepared_unverified")
    config.setdefault("vehicle_type", "player_truck")
    identity = config.setdefault("identity", {})
    name = project_data.get("vehicle_system", {}).get("name") or project_data.get("project", {}).get("name", "")
    identity.setdefault("brand_id", "puki")
    identity.setdefault("model_id", "puki.pickup")
    identity.setdefault("display_name", name)
    identity.setdefault("brand_name", "Puki3D")
    chassis = config.setdefault("chassis", {})
    dims = axle_dimensions(nodes)
    wheel_axles = sorted({
        node.get("vehicle", {}).get("wheel", {}).get("axle")
        for node in nodes if node.get("vehicle", {}).get("role") == "wheel"
        and node.get("vehicle", {}).get("wheel", {}).get("axle") in {"front", "rear"}
    })
    chassis.setdefault("drive_layout", "4x2")
    chassis["axle_count"] = len(wheel_axles)
    chassis["wheelbase_m"] = dims["wheelbase"]
    chassis["front_track_m"] = dims["front_track"]
    chassis["rear_track_m"] = dims["rear_track"]
    chassis.setdefault("max_steering_angle_deg", 40.0)
    chassis.setdefault("collision_path", "")
    chassis.setdefault("suspension_types", ["wishbone", "solidaxle"][:max(1, len(wheel_axles))])
    chassis.setdefault("residual_travel_m", [0.2] * len(wheel_axles))
    chassis.setdefault("kerb_weight_kg", [0.0] * len(wheel_axles))
    power = config.setdefault("engine", {})
    power.setdefault("type", "petrol")
    power.setdefault("volume_l", 0.0)
    power.setdefault("torque_nm", 0.0)
    power.setdefault("torque_peak_rpm", 0)
    power.setdefault("power_kw", 0.0)
    power.setdefault("torque_curve", [])
    power.setdefault("rpm_idle", 600)
    power.setdefault("rpm_limit", 4500)
    transmission = config.setdefault("transmission", {})
    transmission.setdefault("differential_ratio", 0.0)
    transmission.setdefault("shift_time", 1.0)
    transmission.setdefault("ratios_forward", [])
    transmission.setdefault("ratios_reverse", [])
    lights = config.setdefault("lights", {})
    lights.setdefault("headlight_unit", "")
    cameras = config.setdefault("cameras", {})
    cameras.setdefault("behind_camera", "camera.behind.basic")
    cameras.setdefault("interior_camera", "")
    cameras.setdefault("bumper_camera", "camera.bumper.basic")
    cameras.setdefault("wheel_camera", "camera.wheel.basic")
    config.setdefault("dlc_dependencies", [])
    config.setdefault("external_references", [])
    config.setdefault("sounds", {"exterior": "", "interior": ""})
    config.setdefault("fallbacks", {})
    return config


def _message(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def ats_diagnostics(project_data: dict, nodes: list[dict], installed_dlcs: set[str] | None = None) -> list[dict[str, str]]:
    """Report actual SCS truck-data requirements and mark unverified Puki3D data."""
    config = project_data.get("ats_config", {})
    identity = config.get("identity", {})
    chassis = config.get("chassis", {})
    engine = config.get("engine", {})
    transmission = config.get("transmission", {})
    cameras = config.get("cameras", {})
    lights = config.get("lights", {})
    messages: list[dict[str, str]] = []

    if config.get("vehicle_type") != "player_truck":
        messages.append(_message("error", "vehicle_type", "Para conducirlo como vehículo de jugador, ATS requiere una configuración de camión de jugador."))
    if not str(identity.get("brand_name", "")).strip() or not str(identity.get("display_name", "")).strip():
        messages.append(_message("error", "identity", "Falta el nombre de marca o modelo para la identidad del camión."))
    unit_id = f"{identity.get('brand_id', '')}.{identity.get('model_id', '').split('.')[-1]}"
    if not _UNIT_ID.fullmatch(unit_id):
        messages.append(_message("error", "unit_id", "El identificador de unidad debe tener formato brand.model en minúsculas (SII)."))

    system = project_data.get("vehicle_system")
    if not system:
        messages.append(_message("error", "vehicle_system", "Configura primero VEHICLE SYSTEM: ATS requiere piezas y ruedas registradas."))
    else:
        for issue in vehicle_diagnostics(project_data, nodes):
            messages.append(_message("error", "vehicle_system", issue))
    wheel_nodes = [node for node in nodes if node.get("vehicle", {}).get("role") == "wheel"]
    axle_counts = {axle: sum(n.get("vehicle", {}).get("wheel", {}).get("axle") == axle for n in wheel_nodes)
                   for axle in ("front", "rear")}
    if len(wheel_nodes) != 4 or axle_counts != {"front": 2, "rear": 2}:
        messages.append(_message("error", "wheels", "Un chasis 4 ruedas requiere dos ruedas por cada uno de los dos ejes registrados."))
    if int(chassis.get("axle_count", 0) or 0) != 2:
        messages.append(_message("error", "axles", "La configuración de pickup de esta etapa espera dos ejes; ajusta el número de ejes."))
    max_steer = float(chassis.get("max_steering_angle_deg", 0) or 0)
    if not 0 < max_steer <= 70:
        messages.append(_message("error", "steering", "El ángulo máximo de dirección debe estar entre 0° y 70° para el chasis de camión."))
    for key, values, allowed in (
        ("suspension_types", chassis.get("suspension_types", []), set(SUSPENSION_TYPES)),
        ("residual_travel_m", chassis.get("residual_travel_m", []), None),
        ("kerb_weight_kg", chassis.get("kerb_weight_kg", []), None),
    ):
        if len(values) != 2:
            messages.append(_message("error", key, f"{key} necesita un valor por cada uno de los 2 ejes."))
        if allowed is not None and any(value not in allowed for value in values):
            messages.append(_message("error", key, "Tipo de suspensión no reconocido por accessory_chassis_data."))
        if key == "residual_travel_m" and any(float(v) < 0 for v in values):
            messages.append(_message("error", key, "El recorrido de suspensión no puede ser negativo."))
        if key == "kerb_weight_kg" and any(float(v) <= 0 for v in values):
            messages.append(_message("error", key, "Asigna el peso estático (kerb_weight) de cada eje en kg."))

    if engine.get("type") in {"diesel", "petrol"} and float(engine.get("volume_l", 0) or 0) < 0.8:
        messages.append(_message("error", "engine_volume", "Motor incompleto: accessory_engine_data exige volumen mínimo de 0.8 L."))
    if float(engine.get("torque_nm", 0) or 0) <= 0:
        messages.append(_message("error", "engine_torque", "Falta el par máximo del motor en N·m."))
    if int(engine.get("torque_peak_rpm", 0) or 0) <= 0:
        messages.append(_message("error", "engine_rpm", "Falta el régimen al que se alcanza el par máximo."))
    if engine.get("type") not in ENGINE_TYPES:
        messages.append(_message("error", "engine_type", "Tipo de motor inválido; usa diesel, petrol o electric."))
    if engine.get("type") == "electric" and float(engine.get("power_kw", 0) or 0) <= 0 and not engine.get("torque_curve"):
        messages.append(_message("error", "electric_power", "El motor eléctrico necesita power positivo o torque_curve."))
    curve = engine.get("torque_curve", [])
    if curve:
        if len(curve) < 2 or any(float(rpm) <= 0 or not 0 <= float(factor) <= 1 for rpm, factor in curve):
            messages.append(_message("error", "torque_curve", "Puntos (rpm, factor) requieren rpm > 0 y factor entre 0 y 1."))
        elif any(float(right[0]) <= float(left[0]) for left, right in zip(curve, curve[1:])):
            messages.append(_message("error", "torque_curve_order", "Los puntos de torque_curve deben aumentar en rpm."))
    elif engine.get("type") in {"diesel", "petrol"}:
        messages.append(_message("warning", "torque_curve_default", "Sin torque_curve: ATS usará la curva genérica predeterminada."))
    idle, limit = int(engine.get("rpm_idle", 0) or 0), int(engine.get("rpm_limit", 0) or 0)
    if not 500 <= idle <= 850 or limit <= idle:
        messages.append(_message("error", "engine_rpm_range", "El ralentí debe estar entre 500 y 850 rpm y el límite por encima del ralentí."))
    forward = transmission.get("ratios_forward", [])
    reverse = transmission.get("ratios_reverse", [])
    if any(float(a) <= float(b) for a, b in zip(forward, forward[1:])):
        messages.append(_message("error", "forward_ratio_order", "Las relaciones hacia delante deben ir en orden de mayor a menor."))
    if any(float(a) >= float(b) for a, b in zip(reverse, reverse[1:])):
        messages.append(_message("error", "reverse_ratio_order", "Las relaciones de reversa deben ir en orden de reducciÃ³n decreciente (ej. -11, -8)."))
    if not forward or any(float(v) <= 0 for v in forward):
        messages.append(_message("error", "forward_ratios", "La transmisión requiere relaciones hacia delante positivas."))
    if not reverse or any(float(v) >= 0 for v in reverse):
        messages.append(_message("error", "reverse_ratios", "La transmisión requiere al menos una relación de reversa negativa."))
    if float(transmission.get("differential_ratio", 0) or 0) <= 0:
        messages.append(_message("error", "differential_ratio", "Falta una relación final de diferencial positiva."))
    expected_driven = {"4x2": {"rear"}, "4x4": {"front", "rear"}}.get(chassis.get("drive_layout"))
    actual_driven = {node.get("vehicle", {}).get("wheel", {}).get("axle") for node in wheel_nodes
                     if node.get("vehicle", {}).get("wheel", {}).get("driven")}
    if expected_driven is not None and actual_driven != expected_driven:
        messages.append(_message("error", "drive_layout", "La tracción elegida no coincide con los ejes motrices de VEHICLE SYSTEM."))

    fallbacks = config.get("fallbacks", {})
    missing_fallbacks = [name for name in ATS_REQUIRED_FALLBACKS if not str(fallbacks.get(name, "")).strip()]
    if missing_fallbacks:
        messages.append(_message("error", "fallbacks", "Faltan definiciones fallback obligatorias de truck_data: " + ", ".join(missing_fallbacks) + "."))
    else:
        messages.append(_message("warning", "fallbacks_unresolved", "Las rutas fallback están declaradas, pero sus archivos .sii todavía no existen en un paquete y no se pueden resolver."))
    invalid_fallbacks = [name for name in ATS_REQUIRED_FALLBACKS if fallbacks.get(name)
                         and ("|" not in str(fallbacks[name]) or not str(fallbacks[name]).lower().endswith(".sii")
                              or ".." in str(fallbacks[name]) or "\\" in str(fallbacks[name]))]
    if invalid_fallbacks:
        messages.append(_message("error", "fallback_format", "Fallback debe usar tipo|ruta relativa .sii, sin '..' ni barras Windows: " + ", ".join(invalid_fallbacks) + "."))
    if not str(lights.get("headlight_unit", "")).strip():
        messages.append(_message("error", "head_light", "Falta enlazar la unidad de faros seleccionada para el camión."))
    for field, label in CAMERA_FIELDS:
        if not str(cameras.get(field, "")).strip():
            messages.append(_message("warning", f"camera_{field}", f"Falta la referencia opcional de {label} en truck_data."))
    light_mounts = [point for point in system.get("mount_points", []) if point.get("type") == "Light"] if system else []
    if not light_mounts:
        messages.append(_message("warning", "light_locators", "No hay mount points de luces en VEHICLE SYSTEM; la ubicación de faros no está preparada."))
    if not any(node.get("vehicle", {}).get("role") == "collision_helper" for node in nodes):
        messages.append(_message("warning", "collision", "No hay geometría auxiliar de colisión; faltan descriptores PMC/colisión convexa del juego."))
    if not str(chassis.get("collision_path", "")).strip():
        messages.append(_message("warning", "collision_path", "Falta ruta collision (.pmc); todavía no hay geometría de colisión ATS."))
    elif ".." in str(chassis["collision_path"]) or "\\" in str(chassis["collision_path"]):
        messages.append(_message("error", "collision_path_format", "La ruta collision debe usar '/', sin '..' ni separadores Windows."))
    sounds = config.get("sounds", {})
    for key, label in (("exterior", "exterior"), ("interior", "interior")):
        if not str(sounds.get(key, "")).strip():
            messages.append(_message("error", f"sound_{key}", f"Falta el accesorio de sonido {label}; ATS requiere unidades exterior e interior."))
    if project_data.get("materials"):
        messages.append(_message("warning", "materials", "Los materiales PBR de Puki3D aún no declaran shaders SCS; compatibilidad de materiales no verificada."))
    dlcs = config.get("dlc_dependencies", [])
    if any(not _DEPENDENCY_ID.fullmatch(str(item)) for item in dlcs):
        messages.append(_message("error", "dlc_dependency_format", "Las dependencias DLC deben ser identificadores internos sin rutas ni espacios."))
    if installed_dlcs is not None:
        missing = [item for item in dlcs if item not in installed_dlcs]
        if missing:
            messages.append(_message("warning", "dlc_unresolved", "DLC no detectados en esta instalación: " + ", ".join(missing) + "."))
    if dlcs:
        messages.append(_message("warning", "dlc_unverified", "Las DLC declaradas aún no se han resuelto contra un paquete de mod."))
    if config.get("external_references"):
        messages.append(_message("warning", "external_references", "Las referencias externas no pueden resolverse hasta preparar el paquete de mod."))
    invalid_references = [item for item in config.get("external_references", [])
                          if "\\" in str(item) or ".." in str(item) or not str(item).startswith("/")]
    if invalid_references:
        messages.append(_message("error", "external_reference_format", "Las rutas SCS deben empezar con '/' y usar separadores '/', sin '..'."))
    messages.append(_message("info", "verification", "Preparación ATS en Puki3D solamente. No se han generado unidades SII/PIX ni probado el vehículo en el juego."))
    return messages


class ATSConfigDialog(QDialog):
    """Edit SCS-documented truck metadata while keeping build/export out of scope."""

    def __init__(self, project_data: dict, nodes: list[dict], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ATS Config · Preparación del vehículo")
        self.resize(980, 680)
        self.project_data = copy.deepcopy(project_data)
        self.nodes = copy.deepcopy(nodes)
        self.config = ensure_ats_config(self.project_data, self.nodes)
        outer = QVBoxLayout(self)
        banner = QLabel("Preparación ATS · compatibilidad en juego sin verificar")
        banner.setObjectName("dialogTitle")
        outer.addWidget(banner)
        tabs = QTabWidget()
        outer.addWidget(tabs, 1)

        self.identity_fields: dict[str, QLineEdit] = {}
        identity_tab = QWidget(); identity_form = QFormLayout(identity_tab)
        identity = self.config["identity"]
        for key, label in (("brand_name", "Marca mostrada"), ("display_name", "Modelo mostrado"),
                           ("brand_id", "Unidad SII de marca"), ("model_id", "Unidad SII del modelo")):
            field = QLineEdit(str(identity.get(key, "")))
            self.identity_fields[key] = field
            identity_form.addRow(label, field)
        self.vehicle_type = QComboBox(); self.vehicle_type.addItem("Camión de jugador (conducible)", "player_truck")
        identity_form.addRow("Tipo ATS", self.vehicle_type)
        identity_form.addRow(QLabel("Una pickup conducible se define mediante las unidades de jugador truck; ATS no tiene una categoría player-pickup separada."))
        tabs.addTab(identity_tab, "Identidad")

        chassis_tab = QWidget(); chassis_form = QFormLayout(chassis_tab)
        self.layout_combo = QComboBox()
        for item in ("4x2", "4x4"):
            self.layout_combo.addItem(item, item)
        self.layout_combo.setCurrentIndex(max(0, self.layout_combo.findData(self.config["chassis"].get("drive_layout", "4x2"))))
        chassis_form.addRow("Tracción / ruedas motrices", self.layout_combo)
        dims = axle_dimensions(self.nodes)
        metrics = QLabel(f"Ruedabase: {self._fmt(dims['wheelbase'])} m · Vía delantera: {self._fmt(dims['front_track'])} m · Vía trasera: {self._fmt(dims['rear_track'])} m")
        chassis_form.addRow("Medidas desde Vehicle System", metrics)
        wheels = [node.get("vehicle", {}).get("wheel", {}) for node in self.nodes
                  if node.get("vehicle", {}).get("role") == "wheel"]
        wheel_summary = QLabel(f"{len(wheels)} ruedas registradas · radios " + ", ".join(
            f"{float(wheel.get('radius', 0)):.3f} m" for wheel in wheels) + " · anchos " + ", ".join(
            f"{float(wheel.get('width', 0)):.3f} m" for wheel in wheels))
        chassis_form.addRow("Medidas de ruedas", wheel_summary)
        ch = self.config["chassis"]
        self.steering = self._spin(ch.get("max_steering_angle_deg", 40), 0, 70, "°")
        chassis_form.addRow("max_steering_angle (0–70°)", self.steering)
        self.axle_count = QLabel(str(ch.get("axle_count", 0)))
        chassis_form.addRow("Ejes detectados", self.axle_count)
        self.suspension_types = QLineEdit(", ".join(ch.get("suspension_types", [])))
        chassis_form.addRow("axle_suspension_type (por eje)", self.suspension_types)
        self.collision_path = QLineEdit(ch.get("collision_path", ""))
        self.collision_path.setPlaceholderText("/vehicle/truck/puki.pickup/chassis/chassis.pmc")
        chassis_form.addRow("collision (.pmc)", self.collision_path)
        self.travel = QLineEdit(", ".join(str(v) for v in ch.get("residual_travel_m", [])))
        chassis_form.addRow("residual_travel (m por eje)", self.travel)
        self.weights = QLineEdit(", ".join(str(v) for v in ch.get("kerb_weight_kg", [])))
        chassis_form.addRow("kerb_weight (kg por eje)", self.weights)
        chassis_form.addRow(QLabel("Ruedas y orientación se leen de VEHICLE SYSTEM; el rol motriz se agrupa por eje."))
        tabs.addTab(chassis_tab, "Chasis y ruedas")

        engine_tab = QWidget(); engine_form = QFormLayout(engine_tab)
        self.engine_type = QComboBox()
        for item in ENGINE_TYPES: self.engine_type.addItem(item, item)
        self.engine_type.setCurrentIndex(max(0, self.engine_type.findData(self.config["engine"].get("type", "petrol"))))
        engine_form.addRow("type", self.engine_type)
        self.engine_fields: dict[str, QDoubleSpinBox] = {}
        for key, label, value, minimum, maximum, suffix, decimals in (
            ("volume_l", "volumen (L; mínimo SCS 0.8)", "volume_l", 0, 100, " L", 2),
            ("torque_nm", "par máximo (N·m)", "torque_nm", 0, 10000, " N·m", 0),
            ("torque_peak_rpm", "régimen de par máximo", "torque_peak_rpm", 0, 10000, " rpm", 0),
            ("rpm_idle", "rpm_idle", "rpm_idle", 0, 3000, " rpm", 0),
            ("rpm_limit", "rpm_limit", "rpm_limit", 0, 12000, " rpm", 0),
        ):
            field = self._spin(self.config["engine"].get(value, 0), minimum, maximum, suffix, decimals)
            self.engine_fields[key] = field
            engine_form.addRow(label, field)
        self.engine_fields["power_kw"] = self._spin(self.config["engine"].get("power_kw", 0), 0, 5000, " kW", 1)
        engine_form.addRow("power (solo eléctrico, ATS 1.60+)", self.engine_fields["power_kw"])
        self.torque_curve = QLineEdit(", ".join(f"{rpm}:{factor}" for rpm, factor in self.config["engine"].get("torque_curve", [])))
        self.torque_curve.setPlaceholderText("Opcional: 300:0, 440:0.5, 1000:1 (rpm:factor)")
        engine_form.addRow("torque_curve", self.torque_curve)
        engine_form.addRow(QLabel("Los valores describen accessory_engine_data; el par requiere curva torque_curve antes de una simulación final."))
        tabs.addTab(engine_tab, "Motor")

        trans_tab = QWidget(); trans_form = QFormLayout(trans_tab)
        t = self.config["transmission"]
        self.forward_ratios = QLineEdit(", ".join(str(v) for v in t.get("ratios_forward", [])))
        self.reverse_ratios = QLineEdit(", ".join(str(v) for v in t.get("ratios_reverse", [])))
        self.diff_ratio = self._spin(t.get("differential_ratio", 0), 0, 30, "", 3)
        self.shift_time = self._spin(t.get("shift_time", 1.0), 0.1, 10, " s", 2)
        trans_form.addRow("ratios_forward (positivas, de mayor a menor)", self.forward_ratios)
        trans_form.addRow("ratios_reverse (negativas)", self.reverse_ratios)
        trans_form.addRow("differential_ratio", self.diff_ratio)
        trans_form.addRow("shift_time", self.shift_time)
        tabs.addTab(trans_tab, "Transmisión")

        lighting_tab = QWidget(); lighting_form = QFormLayout(lighting_tab)
        self.headlight_unit = QLineEdit(self.config["lights"].get("headlight_unit", ""))
        lighting_form.addRow("Unidad de faros (.sii)", self.headlight_unit)
        self.camera_fields: dict[str, QLineEdit] = {}
        for field, label in CAMERA_FIELDS:
            line = QLineEdit(self.config["cameras"].get(field, "")); self.camera_fields[field] = line
            lighting_form.addRow(label + " (unidad SCS)", line)
        light_count = sum(point.get("type") == "Light" for point in self.configured_mounts())
        lighting_form.addRow("Mount points de luces registrados", QLabel(str(light_count)))
        tabs.addTab(lighting_tab, "Luces y cámaras")

        sound_tab = QWidget(); sound_form = QFormLayout(sound_tab)
        sounds = self.config["sounds"]
        self.exterior_sound = QLineEdit(sounds.get("exterior", ""))
        self.interior_sound = QLineEdit(sounds.get("interior", ""))
        sound_form.addRow("Accesorio sonido exterior (.sii)", self.exterior_sound)
        sound_form.addRow("Accesorio sonido interior (.sii)", self.interior_sound)
        tabs.addTab(sound_tab, "Sonidos")

        deps_tab = QWidget(); deps_form = QFormLayout(deps_tab)
        self.dlc_deps = QLineEdit(", ".join(self.config.get("dlc_dependencies", [])))
        self.external_refs = QLineEdit(", ".join(self.config.get("external_references", [])))
        deps_form.addRow("dlc_dependencies (IDs separados por coma)", self.dlc_deps)
        deps_form.addRow("Referencias externas a recursos/unidades", self.external_refs)
        deps_form.addRow(QLabel("SCS usa dlc_dependencies en Workshop packages; la resolución de referencias se hará al preparar el paquete."))
        tabs.addTab(deps_tab, "Dependencias")

        fallback_tab = QWidget(); fallback_form = QFormLayout(fallback_tab)
        self.fallback_fields: dict[str, QLineEdit] = {}
        for fallback in ATS_REQUIRED_FALLBACKS:
            field = QLineEdit(str(self.config["fallbacks"].get(fallback, "")))
            field.setPlaceholderText(f"{fallback}|<archivo>.sii")
            self.fallback_fields[fallback] = field
            fallback_form.addRow(fallback, field)
        tabs.addTab(fallback_tab, "Truck fallbacks")

        diagnostic_tab = QWidget(); diagnostic_layout = QVBoxLayout(diagnostic_tab)
        self.diagnostics = QListWidget(); diagnostic_layout.addWidget(self.diagnostics)
        self.diagnostic_summary = QLabel(); diagnostic_layout.addWidget(self.diagnostic_summary)
        tabs.addTab(diagnostic_tab, "Validador ATS")

        self._inputs = [*self.identity_fields.values(), self.layout_combo, self.steering, self.suspension_types,
                        self.travel, self.weights, self.collision_path, self.engine_type, *self.engine_fields.values(),
                        self.torque_curve,
                        self.forward_ratios, self.reverse_ratios, self.diff_ratio, self.shift_time,
                        self.headlight_unit, *self.camera_fields.values(), self.exterior_sound, self.interior_sound,
                        self.dlc_deps, self.external_refs]
        self._inputs.extend(self.fallback_fields.values())
        for widget in self._inputs:
            if isinstance(widget, QComboBox): widget.currentIndexChanged.connect(self._refresh_diagnostics)
            elif isinstance(widget, QDoubleSpinBox): widget.valueChanged.connect(self._refresh_diagnostics)
            else: widget.textChanged.connect(self._refresh_diagnostics)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Guardar configuración")
        buttons.accepted.connect(self._commit); buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._refresh_diagnostics()

    def configured_mounts(self) -> list[dict]:
        return self.project_data.get("vehicle_system", {}).get("mount_points", [])

    @staticmethod
    def _fmt(value) -> str:
        return "—" if value is None else f"{value:.3f}"

    @staticmethod
    def _spin(value, minimum, maximum, suffix="", decimals=3) -> QDoubleSpinBox:
        field = QDoubleSpinBox(); field.setRange(minimum, maximum); field.setDecimals(decimals)
        field.setKeyboardTracking(False); field.setSuffix(suffix); field.setValue(float(value or 0))
        return field

    @staticmethod
    def _csv_numbers(value: str) -> list[float]:
        if not value.strip(): return []
        try:
            return [float(part.strip()) for part in value.split(",")]
        except ValueError:
            return []

    @staticmethod
    def _csv_text(value: str) -> list[str]:
        return [part.strip() for part in value.split(",") if part.strip()]

    @staticmethod
    def _parse_torque_curve(value: str) -> list[list[float]]:
        if not value.strip():
            return []
        try:
            curve = []
            for item in value.split(","):
                parts = item.split(":")
                if len(parts) != 2:
                    return []
                curve.append([float(parts[0].strip()), float(parts[1].strip())])
            return curve
        except ValueError:
            return []

    def _refresh_diagnostics(self, *_args) -> None:
        if not hasattr(self, "diagnostics"): return
        draft = copy.deepcopy(self.project_data)
        cfg = ensure_ats_config(draft, self.nodes)
        cfg["vehicle_type"] = self.vehicle_type.currentData()
        cfg["identity"] = {key: field.text() for key, field in self.identity_fields.items()}
        cfg["chassis"].update({"drive_layout": self.layout_combo.currentData(),
                                "max_steering_angle_deg": self.steering.value(),
                                "collision_path": self.collision_path.text().strip(),
                                "suspension_types": self._csv_text(self.suspension_types.text()),
                                "residual_travel_m": self._csv_numbers(self.travel.text()),
                                "kerb_weight_kg": self._csv_numbers(self.weights.text())})
        cfg["engine"].update({"type": self.engine_type.currentData(), **{key: field.value() for key, field in self.engine_fields.items()},
                               "torque_curve": self._parse_torque_curve(self.torque_curve.text())})
        cfg["transmission"].update({"ratios_forward": self._csv_numbers(self.forward_ratios.text()),
                                     "ratios_reverse": self._csv_numbers(self.reverse_ratios.text()),
                                     "differential_ratio": self.diff_ratio.value(), "shift_time": self.shift_time.value()})
        cfg["lights"]["headlight_unit"] = self.headlight_unit.text().strip()
        cfg["cameras"] = {key: field.text().strip() for key, field in self.camera_fields.items()}
        cfg["sounds"] = {"exterior": self.exterior_sound.text().strip(), "interior": self.interior_sound.text().strip()}
        cfg["dlc_dependencies"] = self._csv_text(self.dlc_deps.text())
        cfg["external_references"] = self._csv_text(self.external_refs.text())
        cfg["fallbacks"] = {key: field.text().strip() for key, field in self.fallback_fields.items()}
        messages = ats_diagnostics(draft, self.nodes)
        self.diagnostics.clear()
        for issue in messages:
            prefix = {"error": "ERROR", "warning": "AVISO", "info": "PREPARADO"}[issue["level"]]
            self.diagnostics.addItem(f"{prefix} · {issue['message']}")
        errors = sum(issue["level"] == "error" for issue in messages)
        warnings = sum(issue["level"] == "warning" for issue in messages)
        self.diagnostic_summary.setText(f"{errors} errores · {warnings} avisos · compatibilidad con ATS sin verificar")

    def _commit(self) -> None:
        cfg = self.config
        cfg["status"] = "prepared_unverified"
        cfg["vehicle_type"] = self.vehicle_type.currentData()
        cfg["identity"] = {key: field.text().strip() for key, field in self.identity_fields.items()}
        cfg["chassis"].update({"drive_layout": self.layout_combo.currentData(),
                               "max_steering_angle_deg": self.steering.value(),
                               "collision_path": self.collision_path.text().strip(),
                               "suspension_types": self._csv_text(self.suspension_types.text()),
                               "residual_travel_m": self._csv_numbers(self.travel.text()),
                               "kerb_weight_kg": self._csv_numbers(self.weights.text())})
        cfg["engine"].update({"type": self.engine_type.currentData(), **{key: field.value() for key, field in self.engine_fields.items()},
                              "torque_curve": self._parse_torque_curve(self.torque_curve.text())})
        cfg["transmission"].update({"ratios_forward": self._csv_numbers(self.forward_ratios.text()),
                                    "ratios_reverse": self._csv_numbers(self.reverse_ratios.text()),
                                    "differential_ratio": self.diff_ratio.value(), "shift_time": self.shift_time.value()})
        cfg["lights"]["headlight_unit"] = self.headlight_unit.text().strip()
        cfg["cameras"] = {key: field.text().strip() for key, field in self.camera_fields.items()}
        cfg["sounds"] = {"exterior": self.exterior_sound.text().strip(), "interior": self.interior_sound.text().strip()}
        cfg["dlc_dependencies"] = self._csv_text(self.dlc_deps.text())
        cfg["external_references"] = self._csv_text(self.external_refs.text())
        cfg["fallbacks"] = {key: field.text().strip() for key, field in self.fallback_fields.items()}
        self.project_data["ats_config"] = cfg
        self.accept()
