"""Scene primitives, transforms, and CPU ray picking."""

from __future__ import annotations

from math import cos, pi, radians, sin
from uuid import uuid4

import numpy as np


PRIMITIVES = {
    "cube": "Cubo",
    "sphere": "Esfera",
    "cylinder": "Cilindro",
    "plane": "Plano",
}


def primitive_geometry(primitive: str) -> dict:
    """Return unit-sized, triangulated geometry suitable for project storage."""
    if primitive == "cube":
        vertices = [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return _mesh(vertices, faces)
    if primitive == "plane":
        return _mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [(0, 1, 2, 3)])
    if primitive == "cylinder":
        segments = 32
        vertices = [[cos(2 * pi * i / segments) * 0.5, sin(2 * pi * i / segments) * 0.5, z] for z in (-0.5, 0.5) for i in range(segments)]
        bottom_center, top_center = len(vertices), len(vertices) + 1
        vertices.extend([[0, 0, -0.5], [0, 0, 0.5]])
        faces = []
        for i in range(segments):
            nxt = (i + 1) % segments
            faces.append((i, nxt, segments + nxt, segments + i))
            faces.append((bottom_center, nxt, i))
            faces.append((top_center, segments + i, segments + nxt))
        return _mesh(vertices, faces)
    if primitive == "sphere":
        segments, rings = 24, 12
        vertices = [[0.0, 0.0, 0.5]]
        for ring in range(1, rings):
            latitude = pi * ring / rings
            for segment in range(segments):
                longitude = 2 * pi * segment / segments
                vertices.append([
                    0.5 * sin(latitude) * cos(longitude),
                    0.5 * sin(latitude) * sin(longitude),
                    0.5 * cos(latitude),
                ])
        bottom = len(vertices)
        vertices.append([0.0, 0.0, -0.5])
        faces = []
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append((0, 1 + segment, 1 + nxt))
            for ring in range(rings - 2):
                a = 1 + ring * segments + segment
                b = 1 + ring * segments + nxt
                c = a + segments
                d = b + segments
                faces.extend(((a, c, b), (b, c, d)))
            last_ring = 1 + (rings - 2) * segments
            faces.append((bottom, last_ring + nxt, last_ring + segment))
        return _mesh(vertices, faces)
    raise ValueError(f"Unsupported primitive: {primitive}")


def _mesh(vertices: list, faces: list[tuple]) -> dict:
    triangles = []
    for face in faces:
        for index in range(1, len(face) - 1):
            triangles.append([face[0], face[index], face[index + 1]])
    return {"vertices": vertices, "triangles": triangles}


def create_node(primitive: str, index: int = 1) -> dict:
    if primitive not in PRIMITIVES:
        raise ValueError(f"Unsupported primitive: {primitive}")
    return {
        "id": str(uuid4()),
        "name": f"{PRIMITIVES[primitive]} {index}",
        "type": "mesh",
        "primitive": primitive,
        "geometry": primitive_geometry(primitive),
        "transform": {
            "position": [0.0, 0.0, 0.0 if primitive == "plane" else 0.5],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
    }


def node_matrix(node: dict) -> np.ndarray:
    transform = node.get("transform", {})
    position = np.asarray(transform.get("position", [0, 0, 0]), dtype=float)
    rotation = np.radians(np.asarray(transform.get("rotation", [0, 0, 0]), dtype=float))
    scale = np.asarray(transform.get("scale", [1, 1, 1]), dtype=float)
    cx, cy, cz = np.cos(rotation)
    sx, sy, sz = np.sin(rotation)
    rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=float)
    ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=float)
    rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    scaling = np.diag([scale[0], scale[1], scale[2], 1.0])
    translation = np.eye(4)
    translation[:3, 3] = position
    return translation @ rz @ ry @ rx @ scaling


def raycast(nodes: list[dict], origin: np.ndarray, direction: np.ndarray) -> str | None:
    """Return the closest hit node ID for a world-space ray."""
    direction = direction / np.linalg.norm(direction)
    closest_distance = float("inf")
    closest_id = None
    for node in nodes:
        geometry = node.get("geometry") or primitive_geometry(node.get("primitive", "cube"))
        vertices = geometry.get("vertices", [])
        matrix = node_matrix(node)
        inverse = np.linalg.inv(matrix)
        local_origin = (inverse @ np.append(origin, 1.0))[:3]
        local_direction = (inverse @ np.append(direction, 0.0))[:3]
        for indices in geometry.get("triangles", []):
            a, b, c = (np.asarray(vertices[index], dtype=float) for index in indices)
            edge1, edge2 = b - a, c - a
            h = np.cross(local_direction, edge2)
            determinant = float(np.dot(edge1, h))
            if abs(determinant) < 1e-9:
                continue
            inverse_det = 1.0 / determinant
            s = local_origin - a
            u = inverse_det * float(np.dot(s, h))
            if u < 0.0 or u > 1.0:
                continue
            q = np.cross(s, edge1)
            v = inverse_det * float(np.dot(local_direction, q))
            if v < 0.0 or u + v > 1.0:
                continue
            distance_local = inverse_det * float(np.dot(edge2, q))
            if distance_local <= 0:
                continue
            local_hit = local_origin + distance_local * local_direction
            world_hit = (matrix @ np.append(local_hit, 1.0))[:3]
            distance_world = float(np.linalg.norm(world_hit - origin))
            if distance_world < closest_distance:
                closest_distance = distance_world
                closest_id = node["id"]
    return closest_id
