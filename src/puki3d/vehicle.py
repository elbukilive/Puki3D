"""Vehicle component metadata and diagnostics stored in native projects."""

from __future__ import annotations

import re
import unicodedata
from uuid import uuid4


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
    if not isinstance(meta["mount_points"], list):
        meta["mount_points"] = []
    for node in nodes:
        info = node.get("vehicle")
        if isinstance(info, dict):
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
        wheel.setdefault("side", "left" if position[1] >= 0 else "right")
        wheel.setdefault("axle", "front" if position[0] >= 0 else "rear")
        wheel.setdefault("radius", max(scale[0], scale[1]) / 2.0)
        wheel.setdefault("width", min(scale) / 2.0)
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
    for node in wheels:
        wheel = node.get("vehicle", {}).get("wheel", {})
        if wheel.get("side") not in {"left", "right"} or wheel.get("axle") not in {"front", "rear"}:
            messages.append(f"Rueda mal clasificada: {node.get('name', 'Objeto')}.")
        else:
            wheel_slots.add((wheel["axle"], wheel["side"]))
    if len(wheels) >= 4 and len(wheel_slots) < 4:
        messages.append("Ruedas mal clasificadas: revisa eje y lado; se esperan cuatro posiciones distintas.")
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
            "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}
