"""Vehicle component metadata and diagnostics stored in native projects."""

from __future__ import annotations

import re
import unicodedata
from uuid import uuid4

import numpy as np


COMPONENTS = (
    "Body", "Cabin", "Hood", "Doors", "Bed", "Bumper", "Wheels",
    "Chassis", "Windows", "Lights", "Attachments",
)
ROLES = (
    "visual_body_panel", "wheel", "chassis_part", "glass", "light",
    "collision_helper", "attachment_point", "reference_helper",
)
VEHICLE_TYPES = ("Pickup", "Truck", "Bus", "Trailer", "Other")
MOUNT_TYPES = ("Wheel", "Light", "Accessory", "Mirror", "Plate", "Generic")
REQUIRED_COMPONENTS = ("Body", "Cabin", "Chassis", "Wheels")


def _normalized(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def classify_node_name(name: str) -> tuple[str, str]:
    """Suggest a reusable component and role from common modeling names."""
    value = _normalized(name)
    if set(value.split()) & {"rin", "rim", "hubcap"}:
        return "Wheels", "visual_body_panel"
    rules = (
        (("neum", "wheel", "rueda", "tire", "llanta"), ("Wheels", "wheel")),
        (("chasis", "chassis", "frame", "bastidor"), ("Chassis", "chassis_part")),
        (("parabrisas", "ventana", "luneta", "window", "glass"), ("Windows", "glass")),
        (("faro", "luz", "luces", "light", "intermitente", "lampara"), ("Lights", "light")),
        (("parachoque", "bumper", "defensa"), ("Bumper", "visual_body_panel")),
        (("hood", "capo", "cofre"), ("Hood", "visual_body_panel")),
        (("puerta", "door", "manija", "contorno puerta"), ("Doors", "visual_body_panel")),
        (("caja", "bed", "porton", "tailgate", "costado"), ("Bed", "visual_body_panel")),
        (("techo", "roof", "cabina", "cabin", "pilar", "parabrisas"), ("Cabin", "visual_body_panel")),
        (("espejo", "mirror", "mount", "locator", "attachment"), ("Attachments", "visual_body_panel")),
        (("collision", "colision"), ("Body", "collision_helper")),
        (("reference", "referencia", "helper", "ayuda"), ("Attachments", "reference_helper")),
    )
    for needles, result in rules:
        if any(needle in value for needle in needles):
            return result
    return "Body", "visual_body_panel"


def ensure_vehicle_system(project: dict, nodes: list[dict]) -> dict:
    """Create the optional Stage 3 document without changing the project schema."""
    meta = project.setdefault("vehicle_system", {})
    meta.setdefault("name", project.get("project", {}).get("name", "Vehicle"))
    meta.setdefault("type", "Pickup")
    meta.setdefault("chassis_node_id", None)
    meta.setdefault("mount_points", [])
    meta.setdefault("steering", {"max_angle": 35.0})
    meta.setdefault("suspension", {"travel": 0.25})
    meta["steering"].setdefault("max_angle", 35.0)
    meta["suspension"].setdefault("travel", 0.25)
    if not isinstance(meta["mount_points"], list):
        meta["mount_points"] = []
    for node in nodes:
        info = node.get("vehicle")
        if isinstance(info, dict):
            if info.get("role") == "wheel":
                set_node_classification(node, info.get("component"), "wheel")
            continue
        node["vehicle"] = {"component": None, "role": None}
    return meta


def set_node_classification(node: dict, component: str, role: str) -> None:
    info = node.setdefault("vehicle", {})
    info["component"] = component if component in COMPONENTS else None
    info["role"] = role if role in ROLES else None
    if info["role"] == "wheel":
        transform = node.get("transform", {})
        position = list(transform.get("position", [0.0, 0.0, 0.0]))
        scale = [abs(float(value)) for value in transform.get("scale", [1.0, 1.0, 1.0])]
        wheel = info.setdefault("wheel", {})
        wheel.setdefault("position", position)
        wheel.setdefault("rest_position", list(position))
        wheel.setdefault("rotation", list(transform.get("rotation", [0.0, 0.0, 0.0])))
        wheel.setdefault("side", "left" if position[1] >= 0 else "right")
        wheel.setdefault("axle", "front" if position[0] >= 0 else "rear")
        wheel.setdefault("radius", max(scale[0], scale[1]) / 2.0)
        wheel.setdefault("width", min(scale) / 2.0)
        wheel.setdefault("steerable", wheel["axle"] == "front")
        wheel.setdefault("driven", wheel["axle"] == "rear")
    else:
        info.pop("wheel", None)


def auto_classify(nodes: list[dict]) -> None:
    for node in nodes:
        component, role = classify_node_name(node.get("name", ""))
        set_node_classification(node, component, role)


def vehicle_diagnostics(project_data: dict, nodes: list[dict]) -> list[str]:
    meta = project_data.get("vehicle_system", {})
    messages = []
    if not str(meta.get("name", "")).strip():
        messages.append("Falta el nombre del vehículo.")
    classified = {node.get("id"): node for node in nodes}
    missing_role = [node.get("name", "Objeto") for node in nodes
                    if not isinstance(node.get("vehicle"), dict) or node["vehicle"].get("role") not in ROLES]
    if missing_role:
        messages.append(f"{len(missing_role)} piezas sin rol asignado.")
    for component in REQUIRED_COMPONENTS:
        if not any(node.get("vehicle", {}).get("component") == component for node in nodes):
            messages.append(f"Falta el componente obligatorio: {component}.")
    wheels = [node for node in nodes if node.get("vehicle", {}).get("role") == "wheel"]
    if len(wheels) < 4:
        messages.append(f"Faltan ruedas: hay {len(wheels)} registradas de 4.")
    wheel_slots = set()
    steerable_wheels = 0
    driven_wheels = 0
    for node in wheels:
        wheel = node.get("vehicle", {}).get("wheel", {})
        if wheel.get("side") not in {"left", "right"} or wheel.get("axle") not in {"front", "rear"}:
            messages.append(f"Rueda mal clasificada: {node.get('name', 'Objeto')}.")
        else:
            wheel_slots.add((wheel["axle"], wheel["side"]))
            position = wheel.get("position", node.get("transform", {}).get("position", [0, 0, 0]))
            if len(position) >= 2 and ((wheel["side"] == "left") != (float(position[1]) >= 0)):
                messages.append(f"Lado y posición no coinciden: {node.get('name', 'Rueda')}.")
            if len(position) >= 1 and ((wheel["axle"] == "front") != (float(position[0]) >= 0)):
                messages.append(f"Eje y posición no coinciden: {node.get('name', 'Rueda')}.")
        steerable_wheels += bool(wheel.get("steerable"))
        driven_wheels += bool(wheel.get("driven"))
        if float(wheel.get("radius", 0.0)) <= 0 or float(wheel.get("width", 0.0)) <= 0:
            messages.append(f"Radio o ancho inválido: {node.get('name', 'Rueda')}.")
    if len(wheels) >= 4 and len(wheel_slots) < 4:
        messages.append("Ruedas mal clasificadas: revisa eje y lado; se esperan cuatro posiciones distintas.")
    if wheels and not steerable_wheels:
        messages.append("No hay ruedas direccionales registradas.")
    if wheels and not driven_wheels:
        messages.append("No hay ruedas motrices registradas.")
    chassis_id = meta.get("chassis_node_id")
    if not chassis_id or chassis_id not in classified:
        messages.append("Asigna una pieza como referencia de chasis.")
    elif classified[chassis_id].get("vehicle", {}).get("role") != "chassis_part":
        messages.append("La referencia de chasis no tiene el rol chassis_part.")
    seen = set()
    duplicate = False
    for mount in meta.get("mount_points", []):
        name = _normalized(str(mount.get("name", "")))
        if name and name in seen:
            duplicate = True
        seen.add(name)
    if duplicate:
        messages.append("Hay mount points con nombres duplicados.")
    return messages


def create_mount_point(nodes: list[dict], name: str | None = None) -> dict:
    index = 1
    names = {point.get("name") for point in nodes}
    name = name or "Mount Point"
    unique_name = name
    while unique_name in names:
        index += 1
        unique_name = f"{name} {index}"
    return {"id": str(uuid4()), "name": unique_name, "type": "Generic",
            "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
            "attached_node_id": None}


def axle_dimensions(nodes: list[dict]) -> dict[str, float | None]:
    """Return wheelbase and front/rear track from the configured wheel centers."""
    groups: dict[str, list[list[float]]] = {"front": [], "rear": []}
    for node in nodes:
        wheel = node.get("vehicle", {}).get("wheel")
        if node.get("vehicle", {}).get("role") != "wheel" or not isinstance(wheel, dict):
            continue
        axle = wheel.get("axle")
        position = wheel.get("position", node.get("transform", {}).get("position", [0, 0, 0]))
        if axle in groups and len(position) >= 3:
            groups[axle].append([float(value) for value in position[:3]])

    def track(points: list[list[float]]) -> float | None:
        values = [point[1] for point in points]
        return max(values) - min(values) if len(values) >= 2 else None

    front, rear = groups["front"], groups["rear"]
    front_x = sum(point[0] for point in front) / len(front) if front else None
    rear_x = sum(point[0] for point in rear) / len(rear) if rear else None
    return {
        "wheelbase": abs(front_x - rear_x) if front_x is not None and rear_x is not None else None,
        "front_track": track(front),
        "rear_track": track(rear),
    }


def vehicle_preview_node_matrix(node: dict, steering_angle: float = 0.0,
                                wheel_rotation: float = 0.0, suspension_fraction: float = 0.0,
                                max_steering_angle: float = 35.0,
                                suspension_travel: float = 0.25) -> np.ndarray:
    """Build a transient wheel matrix; never writes the source node transform."""
    from .scene import node_matrix

    wheel = node.get("vehicle", {}).get("wheel")
    if node.get("vehicle", {}).get("role") != "wheel" or not isinstance(wheel, dict):
        return node_matrix(node)
    transform = node.get("transform", {})
    rotation = list(wheel.get("rotation", transform.get("rotation", [0.0, 0.0, 0.0])))
    position = list(wheel.get("rest_position", wheel.get("position", transform.get("position", [0, 0, 0]))))
    angle = float(steering_angle) * float(max_steering_angle) if wheel.get("steerable") else 0.0
    travel = float(suspension_fraction) * float(suspension_travel)
    position[2] += travel
    rx, ry, rz = np.radians(rotation)
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    rotate_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rotate_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rotate_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    steering = np.radians(angle)
    cs, ss = np.cos(steering), np.sin(steering)
    steer_z = np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]], dtype=float)
    rolling = np.radians(float(wheel_rotation))
    cr, sr = np.cos(rolling), np.sin(rolling)
    roll_y = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]], dtype=float)
    scale = np.diag([*transform.get("scale", [1.0, 1.0, 1.0]), 1.0])
    result = np.eye(4)
    result[:3, :3] = steer_z @ rotate_z @ rotate_y @ rotate_x @ roll_y @ scale[:3, :3]
    result[:3, 3] = position
    return result


def mount_point_world_position(point: dict, nodes: list[dict]) -> np.ndarray:
    """Resolve a locator in vehicle space or in its attached node's local space."""
    position = np.asarray(point.get("position", [0.0, 0.0, 0.0]), dtype=float)
    attached_id = point.get("attached_node_id")
    if attached_id:
        from .scene import node_matrix
        target = next((node for node in nodes if node.get("id") == attached_id), None)
        if target:
            return (node_matrix(target) @ np.append(position, 1.0))[:3]
    return position
