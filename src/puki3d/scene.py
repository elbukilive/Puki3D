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

DEFAULT_NODE_COLORS = {
    "cube": (0.55, 0.61, 0.69), "sphere": (0.63, 0.68, 0.75),
    "cylinder": (0.58, 0.65, 0.72), "plane": (0.45, 0.50, 0.57),
}


def node_display_color(node: dict) -> tuple[float, float, float]:
    """Resolve a mesh's optional viewport color, preserving legacy defaults."""
    color = node.get("display_color")
    if isinstance(color, str) and len(color) == 7 and color.startswith("#"):
        try:
            return tuple(int(color[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
        except ValueError:
            pass
    return DEFAULT_NODE_COLORS.get(node.get("primitive"), DEFAULT_NODE_COLORS["cube"])


class MeshTopologyError(ValueError):
    """Raised when polygon data cannot form a valid editable surface."""


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
        faces = [list(reversed(range(segments))), list(range(segments, 2 * segments))]
        for i in range(segments):
            nxt = (i + 1) % segments
            faces.append((i, nxt, segments + nxt, segments + i))
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
    faces = [list(face) for face in faces]
    return _geometry(vertices, faces)


def _geometry(vertices: list, faces: list[list[int]]) -> dict:
    """Build editable polygon topology and its render/picking triangulation."""
    vertices = [[float(value) for value in vertex] for vertex in vertices]
    if any(len(vertex) != 3 or not np.all(np.isfinite(vertex)) for vertex in vertices):
        raise MeshTopologyError("Los vértices deben tener tres coordenadas finitas.")
    faces = [[int(index) for index in face] for face in faces]
    triangles = []
    triangle_faces = []
    face_normals = []
    face_uvs = []
    vertex_normals = np.zeros((len(vertices), 3), dtype=float)
    for face in faces:
        normal, area = _face_normal(vertices, face)
        face_normals.append(normal.tolist())
        dominant_axis = int(np.argmax(np.abs(normal)))
        projected = np.delete(np.asarray([vertices[index] for index in face], dtype=float), dominant_axis, axis=1)
        bounds = np.asarray(vertices, dtype=float)
        projected_bounds = np.delete(bounds, dominant_axis, axis=1)
        uv_min = projected_bounds.min(axis=0)
        uv_size = np.maximum(projected_bounds.max(axis=0) - uv_min, 1e-9)
        face_uvs.append(((projected - uv_min) / uv_size).tolist())
        face_triangles = _triangulate_face(vertices, face)
        triangles.extend(face_triangles)
        triangle_faces.extend([len(face_normals) - 1] * len(face_triangles))
        for index in face:
            vertex_normals[index] += normal * area
    for index, normal in enumerate(vertex_normals):
        length = np.linalg.norm(normal)
        if length > 1e-12:
            vertex_normals[index] /= length
    edges = sorted({tuple(sorted((start, end))) for face in faces for start, end in zip(face, face[1:] + face[:1])})
    edge_lookup = {edge: index for index, edge in enumerate(edges)}
    edge_faces = [[] for _ in edges]
    face_edges = []
    for face_id, face in enumerate(faces):
        face_edge_ids = []
        for start, end in zip(face, face[1:] + face[:1]):
            edge_id = edge_lookup[tuple(sorted((start, end)))]
            face_edge_ids.append(edge_id)
            edge_faces[edge_id].append(face_id)
        face_edges.append(face_edge_ids)
    return {
        "vertices": vertices,
        "edges": [list(edge) for edge in edges],
        "edge_faces": edge_faces,
        "face_edges": face_edges,
        "faces": faces,
        "triangles": triangles,
        "triangle_faces": triangle_faces,
        "face_normals": face_normals,
        "vertex_normals": vertex_normals.tolist(),
        "face_uvs": face_uvs,
        "face_materials": [None] * len(faces),
    }


def _face_normal(vertices: list, face: list[int]) -> tuple[np.ndarray, float]:
    if len(face) < 3 or len(set(face)) != len(face):
        raise MeshTopologyError("Cada cara necesita al menos tres vértices distintos.")
    if any(index < 0 or index >= len(vertices) for index in face):
        raise MeshTopologyError("Una cara referencia un vértice inexistente.")
    points = [np.asarray(vertices[index], dtype=float) for index in face]
    normal = np.zeros(3, dtype=float)
    for current, following in zip(points, points[1:] + points[:1]):
        normal += np.cross(current, following)
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 1e-12:
        raise MeshTopologyError("No se admiten caras degeneradas.")
    return normal / magnitude, magnitude * 0.5


def _triangulate_face(vertices: list, face: list[int]) -> list[list[int]]:
    """Triangulate a simple polygon using ear clipping after dominant-axis projection."""
    normal, _ = _face_normal(vertices, face)
    points = [np.asarray(vertices[index], dtype=float) for index in face]
    dominant_axis = int(np.argmax(np.abs(normal)))
    projected = np.delete(np.asarray(points), dominant_axis, axis=1)
    area = sum(
        projected[i, 0] * projected[(i + 1) % len(face), 1]
        - projected[(i + 1) % len(face), 0] * projected[i, 1]
        for i in range(len(face))
    )
    if abs(area) < 1e-10:
        raise MeshTopologyError("No se puede triangular una cara de área nula.")
    orientation = 1.0 if area > 0 else -1.0
    remaining = list(range(len(face)))
    triangles = []

    def cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def inside_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return all(orientation * value >= -1e-10 for value in (
            cross2(a, b, point), cross2(b, c, point), cross2(c, a, point)
        ))

    def intersects(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        ab_c, ab_d = cross2(a, b, c), cross2(a, b, d)
        cd_a, cd_b = cross2(c, d, a), cross2(c, d, b)
        return ab_c * ab_d < -1e-12 and cd_a * cd_b < -1e-12

    count = len(face)
    for i in range(count):
        a, b = projected[i], projected[(i + 1) % count]
        for j in range(i + 1, count):
            if j in (i, (i + 1) % count) or (j + 1) % count in (i, (i + 1) % count):
                continue
            if intersects(a, b, projected[j], projected[(j + 1) % count]):
                raise MeshTopologyError("No se admiten caras con bordes cruzados.")

    # Collinear intermediate vertices are valid boundaries, but cannot form ears.
    changed = True
    while changed and len(remaining) > 3:
        changed = False
        for offset, current in enumerate(remaining):
            previous, following = remaining[offset - 1], remaining[(offset + 1) % len(remaining)]
            if abs(cross2(projected[previous], projected[current], projected[following])) <= 1e-10:
                remaining.pop(offset)
                changed = True
                break
    if len(remaining) == 3:
        a, b, c = projected[remaining]
        if abs(cross2(a, b, c)) <= 1e-10:
            raise MeshTopologyError("No se puede triangular una cara degenerada.")
        return [[face[index] for index in remaining]]

    while len(remaining) > 3:
        ear_found = False
        for offset, current in enumerate(remaining):
            previous = remaining[offset - 1]
            following = remaining[(offset + 1) % len(remaining)]
            a, b, c = projected[[previous, current, following]]
            if orientation * cross2(a, b, c) <= 1e-10:
                continue
            if any(
                inside_triangle(projected[index], a, b, c)
                for index in remaining
                if index not in {previous, current, following}
            ):
                continue
            triangles.append([face[previous], face[current], face[following]])
            remaining.pop(offset)
            ear_found = True
            break
        if not ear_found:
            raise MeshTopologyError("No se pudo triangular una cara simple.")
    triangles.append([face[index] for index in remaining])
    return triangles


def ensure_editable_geometry(geometry: dict) -> dict:
    """Upgrade legacy triangle-only geometry in place without changing its shape."""
    if not geometry.get("faces"):
        geometry["faces"] = [list(triangle) for triangle in geometry.get("triangles", [])]
    old_faces = geometry["faces"]
    old_face_uvs = geometry.get("face_uvs")
    old_face_materials = geometry.get("face_materials")
    rebuilt = _geometry(geometry.get("vertices", []), old_faces)
    if old_faces == geometry.get("faces"):
        if isinstance(old_face_uvs, list) and len(old_face_uvs) == len(old_faces):
            if all(isinstance(uvs, list) and len(uvs) == len(face) for uvs, face in zip(old_face_uvs, old_faces)):
                rebuilt["face_uvs"] = old_face_uvs
        if isinstance(old_face_materials, list) and len(old_face_materials) == len(old_faces):
            rebuilt["face_materials"] = old_face_materials
    geometry.update(rebuilt)
    return geometry


def _selected_boundary_loops(geometry: dict, edge_ids: set[int]) -> list[list[int]]:
    """Order selected manifold boundary edges into simple closed vertex loops."""
    edges, edge_faces = geometry["edges"], geometry["edge_faces"]
    if not edge_ids:
        raise MeshTopologyError("Selecciona los bordes de un contorno abierto.")
    adjacency: dict[int, set[int]] = {}
    for edge_id in edge_ids:
        if edge_id < 0 or edge_id >= len(edges):
            raise MeshTopologyError("La selección contiene una arista inexistente.")
        if len(edge_faces[edge_id]) != 1:
            raise MeshTopologyError("Fill y Bridge requieren aristas de borde con una sola cara.")
        start, end = edges[edge_id]
        adjacency.setdefault(start, set()).add(end)
        adjacency.setdefault(end, set()).add(start)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise MeshTopologyError("Cada contorno debe ser un lazo cerrado sin ramificaciones.")
    remaining = set(adjacency)
    loops = []
    while remaining:
        start = min(remaining)
        loop = [start]
        previous = None
        current = start
        while True:
            candidates = adjacency[current] - ({previous} if previous is not None else set())
            following = min(candidates)
            if following == start:
                break
            if following in loop:
                raise MeshTopologyError("El contorno contiene un recorrido repetido.")
            loop.append(following)
            previous, current = current, following
        if len(loop) < 3:
            raise MeshTopologyError("Un contorno necesita al menos tres vértices.")
        remaining.difference_update(loop)
        loops.append(loop)
    if sum(len(loop) for loop in loops) != len(adjacency):
        raise MeshTopologyError("La selección no forma contornos independientes.")
    return loops


def _orient_boundary_loop(geometry: dict, loop: list[int], *, reverse_face_direction: bool) -> list[int]:
    directed_boundary = {}
    for face, face_edges in zip(geometry["faces"], geometry["face_edges"]):
        for index, edge_id in enumerate(face_edges):
            if len(geometry["edge_faces"][edge_id]) == 1:
                directed_boundary[edge_id] = (face[index], face[(index + 1) % len(face)])
    direction = []
    for start, end in zip(loop, loop[1:] + loop[:1]):
        edge_id = next((index for index, edge in enumerate(geometry["edges"]) if set(edge) == {start, end}), None)
        if edge_id is None or edge_id not in directed_boundary:
            raise MeshTopologyError("El contorno seleccionado no coincide con bordes abiertos de la malla.")
        direction.append(directed_boundary[edge_id] == (start, end))
    if all(direction):
        return list(reversed(loop)) if reverse_face_direction else loop
    if not any(direction):
        return loop if reverse_face_direction else list(reversed(loop))
    raise MeshTopologyError("La orientación del contorno es inconsistente.")


def fill_boundary_geometry(geometry: dict, edge_ids: set[int]) -> dict:
    """Close one planar boundary loop by adding a face with consistent winding."""
    geometry = ensure_editable_geometry(dict(geometry))
    loops = _selected_boundary_loops(geometry, set(edge_ids))
    if len(loops) != 1:
        raise MeshTopologyError("Fill requiere exactamente un contorno cerrado.")
    loop = _orient_boundary_loop(geometry, loops[0], reverse_face_direction=True)
    points = np.asarray([geometry["vertices"][index] for index in loop], dtype=float)
    normal, _ = _face_normal(geometry["vertices"], loop)
    scale = max(float(np.linalg.norm(points[(index + 1) % len(points)] - point)) for index, point in enumerate(points))
    if any(abs(float(np.dot(point - points[0], normal))) > max(scale, 1.0) * 1e-6 for point in points):
        raise MeshTopologyError("Fill requiere un contorno plano.")
    return _geometry(geometry["vertices"], [*geometry["faces"], loop])


def bridge_boundary_geometry(geometry: dict, edge_ids: set[int]) -> dict:
    """Connect two equal-sized, planar boundary loops with a quad strip."""
    geometry = ensure_editable_geometry(dict(geometry))
    loops = _selected_boundary_loops(geometry, set(edge_ids))
    if len(loops) != 2:
        raise MeshTopologyError("Bridge requiere exactamente dos contornos cerrados.")
    if len(loops[0]) != len(loops[1]):
        raise MeshTopologyError("Bridge requiere contornos con el mismo número de vértices.")
    first = _orient_boundary_loop(geometry, loops[0], reverse_face_direction=True)
    second = _orient_boundary_loop(geometry, loops[1], reverse_face_direction=False)
    vertices = np.asarray(geometry["vertices"], dtype=float)
    # Preserve loop direction and choose the least stretched cyclic correspondence.
    shifts = range(len(second))
    shift = min(shifts, key=lambda offset: sum(
        float(np.linalg.norm(vertices[first[index]] - vertices[second[(index + offset) % len(second)]]))
        for index in range(len(first))
    ))
    second = second[shift:] + second[:shift]
    bridge_faces = [
        [first[index], first[(index + 1) % len(first)],
         second[(index + 1) % len(second)], second[index]]
        for index in range(len(first))
    ]
    return _geometry(geometry["vertices"], [*geometry["faces"], *bridge_faces])


def merge_vertices_geometry(geometry: dict, vertex_ids: set[int]) -> dict:
    """Weld selected vertices at their centroid, rejecting any collapsed face."""
    geometry = ensure_editable_geometry(dict(geometry))
    vertex_ids = set(vertex_ids)
    vertices, faces = geometry["vertices"], geometry["faces"]
    if len(vertex_ids) < 2:
        raise MeshTopologyError("Merge requiere seleccionar al menos dos vértices.")
    if any(index < 0 or index >= len(vertices) for index in vertex_ids):
        raise MeshTopologyError("La selección contiene un vértice inexistente.")
    ordered = sorted(vertex_ids)
    anchor = ordered[0]
    center = np.mean(np.asarray([vertices[index] for index in ordered], dtype=float), axis=0)
    kept = [index for index in range(len(vertices)) if index not in vertex_ids or index == anchor]
    remap = {old: new for new, old in enumerate(kept)}
    mapped_anchor = remap[anchor]
    merged_vertices = [list(center) if index == anchor else list(vertices[index]) for index in kept]
    merged_faces = []
    seen_faces = set()
    for face in faces:
        mapped = [mapped_anchor if index in vertex_ids else remap[index] for index in face]
        if len(set(mapped)) < 3:
            raise MeshTopologyError("Merge cancelado: soldar esos vértices colapsaría una cara.")
        canonical = min(tuple(mapped[index:] + mapped[:index]) for index in range(len(mapped)))
        reverse = list(reversed(mapped))
        canonical = min(canonical, min(tuple(reverse[index:] + reverse[:index]) for index in range(len(reverse))))
        if canonical in seen_faces:
            raise MeshTopologyError("Merge cancelado: crearía caras duplicadas.")
        seen_faces.add(canonical)
        merged_faces.append(mapped)
    result = _geometry(merged_vertices, merged_faces)
    if any(len(adjacent) > 2 for adjacent in result["edge_faces"]):
        raise MeshTopologyError("Merge cancelado: produciría aristas no manifold.")
    return result


def mesh_diagnostics(geometry: dict) -> list[dict]:
    """Return non-blocking topology warnings for open, loose, or invalid mesh data."""
    try:
        geometry = ensure_editable_geometry(dict(geometry))
    except (MeshTopologyError, IndexError, TypeError, ValueError) as exc:
        return [{"severity": "error", "code": "invalid_geometry", "message": f"Malla no válida: {exc}"}]
    diagnostics = []
    boundary_ids = [index for index, adjacent in enumerate(geometry["edge_faces"]) if len(adjacent) == 1]
    nonmanifold_ids = [index for index, adjacent in enumerate(geometry["edge_faces"]) if len(adjacent) > 2]
    used = {index for face in geometry["faces"] for index in face}
    loose_ids = [index for index in range(len(geometry["vertices"])) if index not in used]
    if boundary_ids:
        diagnostics.append({"severity": "warning", "code": "open_boundary", "count": len(boundary_ids),
                            "message": f"Hay {len(boundary_ids)} aristas abiertas; la superficie tiene huecos."})
    if nonmanifold_ids:
        diagnostics.append({"severity": "error", "code": "non_manifold", "count": len(nonmanifold_ids),
                            "message": f"Hay {len(nonmanifold_ids)} aristas compartidas por más de dos caras."})
    face_keys = set()
    duplicate_faces = 0
    for face in geometry["faces"]:
        rotations = [tuple(face[index:] + face[:index]) for index in range(len(face))]
        reverse = list(reversed(face))
        rotations.extend(tuple(reverse[index:] + reverse[:index]) for index in range(len(reverse)))
        key = min(rotations)
        if key in face_keys:
            duplicate_faces += 1
        face_keys.add(key)
    if duplicate_faces:
        diagnostics.append({"severity": "error", "code": "duplicate_faces", "count": duplicate_faces,
                            "message": f"Se detectaron {duplicate_faces} caras duplicadas."})
    inconsistent_edges = []
    directed = {}
    edge_lookup = {tuple(edge): edge_id for edge_id, edge in enumerate(geometry["edges"])}
    for face_id, face in enumerate(geometry["faces"]):
        for start, end in zip(face, face[1:] + face[:1]):
            edge_id = edge_lookup[tuple(sorted((start, end)))]
            if edge_id in directed and directed[edge_id][1] == (start, end):
                inconsistent_edges.append(edge_id)
            directed[edge_id] = (face_id, (start, end))
    if inconsistent_edges:
        diagnostics.append({"severity": "warning", "code": "inconsistent_winding", "count": len(set(inconsistent_edges)),
                            "message": f"Hay {len(set(inconsistent_edges))} aristas con caras orientadas igual."})
    if loose_ids:
        diagnostics.append({"severity": "warning", "code": "loose_vertices", "count": len(loose_ids),
                            "message": f"Hay {len(loose_ids)} vértices sin caras conectadas."})
    if not diagnostics:
        diagnostics.append({"severity": "ok", "code": "closed_manifold", "message": "Malla cerrada: no se detectaron huecos ni aristas no manifold."})
    return diagnostics


def loop_cut_geometry(geometry: dict, edge_id: int, fraction: float = 0.5) -> dict:
    """Insert one continuous cut through a closed loop of quad faces."""
    geometry = ensure_editable_geometry(dict(geometry))
    edges, edge_faces, faces = geometry["edges"], geometry["edge_faces"], geometry["faces"]
    if edge_id < 0 or edge_id >= len(edges) or not 0.0 < fraction < 1.0:
        raise MeshTopologyError("Loop Cut requiere una arista válida y una posición interior.")
    crossed_edges, crossed_faces = {edge_id}, set()
    pending = [edge_id]
    while pending:
        current = pending.pop()
        adjacent = edge_faces[current]
        if len(adjacent) != 2:
            raise MeshTopologyError("Loop Cut requiere una topología cerrada y manifold.")
        for face_id in adjacent:
            face = faces[face_id]
            if len(face) != 4:
                raise MeshTopologyError("Loop Cut solo continúa por caras cuadriláteras.")
            if face_id in crossed_faces:
                continue
            crossed_faces.add(face_id)
            edge_position = geometry["face_edges"][face_id].index(current)
            opposite = geometry["face_edges"][face_id][(edge_position + 2) % 4]
            if opposite not in crossed_edges:
                crossed_edges.add(opposite)
                pending.append(opposite)
    # Every crossed edge must be paired by the traversed quads; otherwise this is an open loop.
    for current in crossed_edges:
        if len(edge_faces[current]) != 2 or not all(face_id in crossed_faces for face_id in edge_faces[current]):
            raise MeshTopologyError("Loop Cut encontró un límite y se canceló sin modificar la malla.")
    vertices = [list(vertex) for vertex in geometry["vertices"]]
    split_ids = {}
    for current in crossed_edges:
        start, end = edges[current]
        point = np.asarray(vertices[start]) * (1.0 - fraction) + np.asarray(vertices[end]) * fraction
        split_ids[current] = len(vertices)
        vertices.append(point.tolist())
    result_faces = []
    for face_id, face in enumerate(faces):
        face_edges = geometry["face_edges"][face_id]
        positions = [index for index, current in enumerate(face_edges) if current in crossed_edges]
        if len(positions) != 2:
            if face_id in crossed_faces:
                raise MeshTopologyError("La topología no permite cerrar este Loop Cut.")
            result_faces.append(list(face))
            continue
        first, second = positions
        if (first - second) % 4 != 2:
            raise MeshTopologyError("Loop Cut no encontró aristas opuestas en una cara.")
        first_id, second_id = split_ids[face_edges[first]], split_ids[face_edges[second]]
        result_faces.append([first_id, *face[(first + 1):(second + 1)], second_id])
        result_faces.append([second_id, *face[(second + 1):] + face[:first + 1], first_id])
    return _geometry(vertices, result_faces)


def bevel_geometry(geometry: dict, edge_ids: list[int] | set[int], width: float) -> dict:
    """Chamfer selected manifold edges with one segment and reject open results."""
    geometry = ensure_editable_geometry(dict(geometry))
    if not np.isfinite(width) or width <= 0:
        raise MeshTopologyError("El ancho de Bevel debe ser positivo.")
    initially_used = {index for face in geometry["faces"] for index in face}
    initially_loose = set(range(len(geometry["vertices"]))) - initially_used
    mesh_vertices = np.asarray(geometry["vertices"], dtype=float)
    mesh_center = np.mean(mesh_vertices, axis=0)
    convex = True
    for face, normal in zip(geometry["faces"], geometry["face_normals"]):
        anchor = mesh_vertices[face[0]]
        signed = [float(np.dot(point - anchor, normal)) for point in mesh_vertices]
        if not (max(signed) <= 1e-6 or min(signed) >= -1e-6):
            convex = False
            break
    oriented_faces = []
    if convex:
        for face, normal in zip(geometry["faces"], geometry["face_normals"]):
            anchor = np.asarray(geometry["vertices"][face[0]])
            oriented_faces.append(list(face) if float(np.dot(np.asarray(normal), mesh_center - anchor)) <= 1e-8 else list(reversed(face)))
        geometry = _geometry(geometry["vertices"], oriented_faces)
    edges, edge_faces = geometry["edges"], geometry["edge_faces"]
    if not edge_ids:
        raise MeshTopologyError("Selecciona al menos una arista para Bevel.")
    vertices = geometry["vertices"]
    original_boundary_count = sum(len(adjacent) == 1 for adjacent in edge_faces)
    planes = []
    for edge_id in sorted(set(edge_ids)):
        if edge_id < 0 or edge_id >= len(edges):
            raise MeshTopologyError("La arista seleccionada ya no existe.")
        adjacent = edge_faces[edge_id]
        if len(adjacent) != 2:
            raise MeshTopologyError("Bevel requiere aristas manifold con dos caras.")
        normals = [np.asarray(geometry["face_normals"][face_id], dtype=float) for face_id in adjacent]
        bevel_normal = normals[0] + normals[1]
        length = float(np.linalg.norm(bevel_normal))
        if length <= 1e-10:
            raise MeshTopologyError("No se puede biselar una arista con caras coplanares opuestas.")
        bevel_normal /= length
        midpoint = (np.asarray(vertices[edges[edge_id][0]]) + np.asarray(vertices[edges[edge_id][1]])) * 0.5
        plane = float(np.dot(bevel_normal, midpoint) - width * max(float(np.dot(bevel_normal, normals[0])), 1e-6))
        planes.append((bevel_normal, plane))
    result = geometry
    for bevel_normal, plane in planes:
        vertices = result["vertices"]
        new_vertices = [list(vertex) for vertex in vertices]
        clipped_faces, cap_segments, intersections = [], [], {}
        for face in result["faces"]:
            clipped, face_crossings = [], []
            for start, end in zip(face, face[1:] + face[:1]):
                a, b = np.asarray(vertices[start]), np.asarray(vertices[end])
                da, db = float(np.dot(bevel_normal, a) - plane), float(np.dot(bevel_normal, b) - plane)
                inside_a, inside_b = da <= 1e-9, db <= 1e-9
                if inside_a:
                    clipped.append(start)
                if inside_a != inside_b:
                    key = tuple(sorted((start, end)))
                    if key not in intersections:
                        t = da / (da - db)
                        intersections[key] = len(new_vertices)
                        new_vertices.append((a + (b - a) * t).tolist())
                    intersection_id = intersections[key]
                    clipped.append(intersection_id)
                    face_crossings.append(intersection_id)
            deduplicated = []
            for index in clipped:
                if not deduplicated or index != deduplicated[-1]:
                    deduplicated.append(index)
            if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
                deduplicated.pop()
            if len(deduplicated) >= 3:
                clipped_faces.append(deduplicated)
            if len(set(face_crossings)) == 2:
                cap_segments.append(tuple(set(face_crossings)))
        adjacency: dict[int, set[int]] = {}
        for start, end in cap_segments:
            adjacency.setdefault(start, set()).add(end)
            adjacency.setdefault(end, set()).add(start)
        if len(adjacency) < 3 or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            raise MeshTopologyError("El ancho de Bevel es demasiado grande o colapsa la malla.")
        cap_cycles = []
        remaining = set(adjacency)
        while remaining:
            start = min(remaining)
            cycle = [start]
            previous, current = None, start
            while True:
                candidates = adjacency[current] - ({previous} if previous is not None else set())
                following = min(candidates)
                if following == start:
                    break
                if following in cycle:
                    raise MeshTopologyError("El corte de Bevel no forma ciclos simples.")
                cycle.append(following)
                previous, current = current, following
            if len(cycle) < 3:
                raise MeshTopologyError("El ancho de Bevel es demasiado grande o colapsa la malla.")
            remaining.difference_update(cycle)
            cap_cycles.append(cycle)
        helper = np.eye(3)[int(np.argmin(np.abs(bevel_normal)))]
        axis_u = np.cross(bevel_normal, helper)
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(bevel_normal, axis_u)
        for cap in cap_cycles:
            center = np.mean([new_vertices[index] for index in cap], axis=0)
            cap.sort(key=lambda index: np.arctan2(
                np.dot(np.asarray(new_vertices[index]) - center, axis_v),
                np.dot(np.asarray(new_vertices[index]) - center, axis_u),
            ))
            if np.dot(np.cross(np.asarray(new_vertices[cap[1]]) - new_vertices[cap[0]], np.asarray(new_vertices[cap[2]]) - new_vertices[cap[0]]), bevel_normal) < 0:
                cap.reverse()
            clipped_faces.append(cap)
        result = _geometry(new_vertices, clipped_faces)
        incidences = [len(adjacent) for adjacent in result["edge_faces"]]
        boundary_count = sum(count == 1 for count in incidences)
        if any(count > 2 for count in incidences) or boundary_count > original_boundary_count:
            raise MeshTopologyError("Bevel crearía bordes abiertos o más de dos caras por arista.")
    used = {index for face in result["faces"] for index in face} | initially_loose
    if len(used) != len(result["vertices"]):
        kept = sorted(used)
        remap = {old: new for new, old in enumerate(kept)}
        result = _geometry([result["vertices"][index] for index in kept],
                           [[remap[index] for index in face] for face in result["faces"]])
    return result


def inset_face_geometry(geometry: dict, face_id: int, amount: float) -> dict:
    """Return a mesh with one convex polygon face inset by a world/local distance."""
    geometry = ensure_editable_geometry(dict(geometry))
    faces = geometry["faces"]
    if face_id < 0 or face_id >= len(faces):
        raise MeshTopologyError("La cara seleccionada ya no existe.")
    if not np.isfinite(amount) or amount <= 0:
        raise MeshTopologyError("El inset debe tener una distancia positiva.")
    face = faces[face_id]
    vertices = geometry["vertices"]
    normal, _ = _face_normal(vertices, face)
    origin = np.asarray(vertices[face[0]], dtype=float)
    edge = np.asarray(vertices[face[1]], dtype=float) - origin
    edge -= normal * float(np.dot(edge, normal))
    edge_length = float(np.linalg.norm(edge))
    if edge_length <= 1e-12:
        raise MeshTopologyError("La cara seleccionada tiene una arista degenerada.")
    u = edge / edge_length
    v = np.cross(normal, u)
    points3 = [np.asarray(vertices[index], dtype=float) for index in face]
    scale = max(float(np.linalg.norm(points3[(i + 1) % len(face)] - point)) for i, point in enumerate(points3))
    if any(abs(float(np.dot(point - origin, normal))) > max(scale, 1.0) * 1e-7 for point in points3):
        raise MeshTopologyError("Inset requiere una cara plana.")
    points2 = [np.array([np.dot(point - origin, u), np.dot(point - origin, v)]) for point in points3]
    signed_area = sum(
        points2[i][0] * points2[(i + 1) % len(face)][1] - points2[(i + 1) % len(face)][0] * points2[i][1]
        for i in range(len(face))
    ) * 0.5
    orientation = 1.0 if signed_area > 0 else -1.0
    if abs(signed_area) <= 1e-12:
        raise MeshTopologyError("La cara seleccionada tiene área nula.")

    def cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    inward, edge_directions = [], []
    for start, end in zip(points2, points2[1:] + points2[:1]):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            raise MeshTopologyError("La cara seleccionada tiene bordes degenerados.")
        direction = direction / length
        edge_directions.append(direction)
        inward.append(orientation * np.array([-direction[1], direction[0]]))
    turns = [
        orientation * cross2(
            points2[(i + 1) % len(face)] - points2[i],
            points2[(i + 2) % len(face)] - points2[(i + 1) % len(face)],
        )
        for i in range(len(face))
    ]
    if any(turn <= 1e-10 for turn in turns):
        raise MeshTopologyError("Inset requiere una cara convexa sin vértices colineales.")

    inner_points = []
    for i, point in enumerate(points2):
        previous_edge = (i - 1) % len(face)
        normal_a, normal_b = inward[previous_edge], inward[i]
        line_a, line_b = point + normal_a * amount, point + normal_b * amount
        direction_a, direction_b = edge_directions[previous_edge], edge_directions[i]
        determinant = cross2(direction_a, direction_b)
        if abs(determinant) <= 1e-10:
            raise MeshTopologyError("La distancia de inset colapsa la cara.")
        t = cross2(line_b - line_a, direction_b) / determinant
        inner = line_a + direction_a * t
        if any(orientation * cross2(end - start, inner - start) < -1e-9
               for start, end in zip(points2, points2[1:] + points2[:1])):
            raise MeshTopologyError("El inset es demasiado grande para la cara seleccionada.")
        inner_points.append(inner)
    inner_area = sum(
        inner_points[i][0] * inner_points[(i + 1) % len(face)][1] - inner_points[(i + 1) % len(face)][0] * inner_points[i][1]
        for i in range(len(face))
    ) * 0.5
    if inner_area * orientation <= 1e-10:
        raise MeshTopologyError("La distancia de inset colapsa la cara.")

    new_vertices = [list(point) for point in vertices]
    inner_ids = []
    for point in inner_points:
        inner_ids.append(len(new_vertices))
        new_vertices.append((origin + u * point[0] + v * point[1]).tolist())
    new_faces = [list(face_value) for face_value in faces]
    new_faces[face_id] = inner_ids
    for i, (start, end) in enumerate(zip(face, face[1:] + face[:1])):
        new_faces.append([start, end, inner_ids[(i + 1) % len(face)], inner_ids[i]])
    return _geometry(new_vertices, new_faces)


def boolean_difference_geometry(target: dict, cutter: dict, tolerance: float | None = None) -> dict:
    """Subtract a closed cutter mesh from a closed target mesh using BSP clipping.

    The operation is intentionally conservative: both inputs and the result must
    be closed two-manifold surfaces. Invalid, touching, or collapsed results are
    rejected before the caller replaces its editable mesh.
    """
    target = ensure_editable_geometry(dict(target))
    cutter = ensure_editable_geometry(dict(cutter))
    for label, geometry in (("base", target), ("cortadora", cutter)):
        if not geometry["faces"] or any(len(adjacent) != 2 for adjacent in geometry["edge_faces"]):
            raise MeshTopologyError(f"La malla {label} debe estar cerrada y ser manifold para Diferencia.")
        if any(item["severity"] == "warning" for item in mesh_diagnostics(geometry)):
            raise MeshTopologyError(f"La malla {label} tiene problemas topológicos; Diferencia se canceló.")

    def signed_volume(geometry: dict) -> float:
        points = np.asarray(geometry["vertices"], dtype=float)
        volume = 0.0
        for face in geometry["faces"]:
            anchor = points[face[0]]
            for index in range(1, len(face) - 1):
                volume += float(np.dot(anchor, np.cross(points[face[index]], points[face[index + 1]]))) / 6.0
        return volume

    # Edge winding checks catch local inconsistencies, but a consistently
    # inward-facing closed shell passes them. Normalize both solids before BSP
    # clipping, whose inside/outside classification depends on outward normals.
    if abs(signed_volume(target)) <= 1e-12 or abs(signed_volume(cutter)) <= 1e-12:
        raise MeshTopologyError("Diferencia requiere sólidos cerrados con volumen válido.")
    if signed_volume(target) < 0:
        target = _geometry(target["vertices"], [list(reversed(face)) for face in target["faces"]])
    if signed_volume(cutter) < 0:
        cutter = _geometry(cutter["vertices"], [list(reversed(face)) for face in cutter["faces"]])
    all_points = np.asarray(target["vertices"] + cutter["vertices"], dtype=float)
    scale = max(float(np.max(np.ptp(all_points, axis=0))), 1.0)
    eps = float(tolerance if tolerance is not None else scale * 1e-7)
    if not np.isfinite(eps) or eps <= 0:
        raise MeshTopologyError("La tolerancia de Diferencia debe ser positiva.")

    class Plane:
        def __init__(self, normal: np.ndarray, offset: float):
            self.normal, self.offset = normal, offset

        @classmethod
        def from_points(cls, points):
            normal, area = _face_normal(points, list(range(len(points))))
            if area <= eps:
                raise MeshTopologyError("Diferencia encontró una cara degenerada.")
            normal = np.asarray(normal, dtype=float)
            return cls(normal, float(np.dot(normal, points[0])))

        def flipped(self):
            return Plane(-self.normal, -self.offset)

        def split(self, polygon, coplanar_front, coplanar_back, front, back):
            COPLANAR, FRONT, BACK, SPANNING = 0, 1, 2, 3
            types = []
            polygon_type = 0
            for vertex in polygon.vertices:
                distance = float(np.dot(self.normal, vertex) - self.offset)
                kind = FRONT if distance > eps else BACK if distance < -eps else COPLANAR
                polygon_type |= kind
                types.append(kind)
            if polygon_type == COPLANAR:
                (coplanar_front if float(np.dot(self.normal, polygon.plane.normal)) > 0 else coplanar_back).append(polygon)
            elif polygon_type == FRONT:
                front.append(polygon)
            elif polygon_type == BACK:
                back.append(polygon)
            else:
                front_vertices, back_vertices = [], []
                for index, vertex in enumerate(polygon.vertices):
                    following = (index + 1) % len(polygon.vertices)
                    next_vertex = polygon.vertices[following]
                    kind, next_kind = types[index], types[following]
                    if kind != BACK:
                        front_vertices.append(vertex)
                    if kind != FRONT:
                        back_vertices.append(vertex)
                    if (kind | next_kind) == SPANNING:
                        direction = next_vertex - vertex
                        denominator = float(np.dot(self.normal, direction))
                        if abs(denominator) <= eps:
                            continue
                        factor = (self.offset - float(np.dot(self.normal, vertex))) / denominator
                        intersection = vertex + direction * factor
                        front_vertices.append(intersection)
                        back_vertices.append(intersection.copy())
                for vertices, destination in ((front_vertices, front), (back_vertices, back)):
                    cleaned = []
                    for vertex in vertices:
                        if not cleaned or np.linalg.norm(vertex - cleaned[-1]) > eps:
                            cleaned.append(vertex)
                    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= eps:
                        cleaned.pop()
                    if len(cleaned) >= 3:
                        destination.append(Polygon(cleaned))

    class Polygon:
        def __init__(self, vertices):
            self.vertices = [np.asarray(vertex, dtype=float) for vertex in vertices]
            self.plane = Plane.from_points(self.vertices)

        def flipped(self):
            return Polygon(list(reversed(self.vertices)))

    class Node:
        def __init__(self, polygons=None):
            self.plane = None
            self.front = self.back = None
            self.polygons = []
            if polygons:
                self.build(polygons)

        def clone(self):
            result = Node()
            result.plane = self.plane
            result.front = self.front.clone() if self.front else None
            result.back = self.back.clone() if self.back else None
            result.polygons = list(self.polygons)
            return result

        def invert(self):
            self.polygons = [polygon.flipped() for polygon in self.polygons]
            if self.plane:
                self.plane = self.plane.flipped()
            if self.front:
                self.front.invert()
            if self.back:
                self.back.invert()
            self.front, self.back = self.back, self.front

        def clip_polygons(self, polygons):
            if self.plane is None:
                return list(polygons)
            front, back = [], []
            for polygon in polygons:
                self.plane.split(polygon, front, back, front, back)
            if self.front:
                front = self.front.clip_polygons(front)
            if self.back:
                back = self.back.clip_polygons(back)
            else:
                back = []
            return front + back

        def clip_to(self, other):
            self.polygons = other.clip_polygons(self.polygons)
            if self.front:
                self.front.clip_to(other)
            if self.back:
                self.back.clip_to(other)

        def all_polygons(self):
            result = list(self.polygons)
            if self.front:
                result.extend(self.front.all_polygons())
            if self.back:
                result.extend(self.back.all_polygons())
            return result

        def build(self, polygons):
            if not polygons:
                return
            if self.plane is None:
                self.plane = polygons[0].plane
            front, back = [], []
            for polygon in polygons:
                self.plane.split(polygon, self.polygons, self.polygons, front, back)
            if front:
                if self.front is None:
                    self.front = Node()
                self.front.build(front)
            if back:
                if self.back is None:
                    self.back = Node()
                self.back.build(back)

    def polygons_from(geometry):
        return [Polygon([geometry["vertices"][index] for index in face]) for face in geometry["faces"]]

    a, b = Node(polygons_from(target)), Node(polygons_from(cutter))
    a.invert()
    a.clip_to(b)
    b.clip_to(a)
    b.invert()
    b.clip_to(a)
    b.invert()
    a.build(b.all_polygons())
    a.invert()
    polygons = a.all_polygons()
    if not polygons:
        raise MeshTopologyError("La Diferencia eliminaría por completo la malla base.")

    vertices, faces, lookup = [], [], {}
    quantum = max(eps, 1e-10)
    for polygon in polygons:
        face = []
        for point in polygon.vertices:
            key = tuple(int(round(float(value) / quantum)) for value in point)
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append(point.tolist())
            vertex_id = lookup[key]
            if not face or face[-1] != vertex_id:
                face.append(vertex_id)
        if len(face) > 1 and face[0] == face[-1]:
            face.pop()
        if len(set(face)) >= 3:
            faces.append(face)
    # BSP fragments can leave a long edge opposite several shorter coplanar
    # edges. Split those long edges at every welded vertex before building the
    # shared-edge table so the result has a consistent manifold topology.
    points = np.asarray(vertices, dtype=float)
    conformed_faces = []
    for face in faces:
        conformed = []
        for start_id, end_id in zip(face, face[1:] + face[:1]):
            start, end = points[start_id], points[end_id]
            direction = end - start
            length_squared = float(np.dot(direction, direction))
            insertions = []
            if length_squared > eps * eps:
                for candidate_id, candidate in enumerate(points):
                    if candidate_id in (start_id, end_id):
                        continue
                    fraction = float(np.dot(candidate - start, direction) / length_squared)
                    if eps / np.sqrt(length_squared) < fraction < 1.0 - eps / np.sqrt(length_squared):
                        closest = start + direction * fraction
                        if np.linalg.norm(candidate - closest) <= eps * 2:
                            insertions.append((fraction, candidate_id))
            conformed.append(start_id)
            conformed.extend(candidate for _, candidate in sorted(insertions))
        conformed_faces.append(conformed)
    faces = conformed_faces
    try:
        result = _geometry(vertices, faces)
    except (IndexError, MeshTopologyError, ValueError) as exc:
        raise MeshTopologyError("Diferencia generó una malla degenerada; se conservó la original.") from exc
    if not result["faces"] or any(len(adjacent) != 2 for adjacent in result["edge_faces"]):
        boundary = sum(len(adjacent) == 1 for adjacent in result["edge_faces"])
        nonmanifold = sum(len(adjacent) > 2 for adjacent in result["edge_faces"])
        raise MeshTopologyError(f"Diferencia generó una superficie no manifold ({boundary} bordes abiertos, {nonmanifold} aristas compartidas de más de dos caras); se conservó la original.")
    if any(item["severity"] == "warning" for item in mesh_diagnostics(result)):
        raise MeshTopologyError("Diferencia generó una topología inválida; se conservó la original.")
    result_volume = signed_volume(result)
    if abs(result_volume) <= eps ** 3:
        raise MeshTopologyError("Diferencia produjo un volumen nulo; se conservó la original.")
    if result_volume < 0:
        result = _geometry(result["vertices"], [list(reversed(face)) for face in result["faces"]])
    return result


def cylindrical_cutter_geometry(center: list[float], radius: float, axis: int, depth: float, segments: int = 16) -> dict:
    """Build a low-poly cylindrical cutting volume along one local axis."""
    if axis not in (0, 1, 2) or not np.isfinite(radius) or radius <= 0 or not np.isfinite(depth) or depth <= 0:
        raise MeshTopologyError("La cortadora requiere eje válido, radio y profundidad positivos.")
    if segments < 6 or segments > 64:
        raise MeshTopologyError("La cortadora cilíndrica requiere entre 6 y 64 segmentos.")
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise MeshTopologyError("El centro de la cortadora debe tener tres coordenadas finitas.")
    other_axes = [index for index in range(3) if index != axis]
    vertices = []
    for offset in (-depth / 2, depth / 2):
        for segment in range(segments):
            angle = 2 * pi * segment / segments
            point = center.copy()
            point[axis] += offset
            point[other_axes[0]] += radius * cos(angle)
            point[other_axes[1]] += radius * sin(angle)
            vertices.append(point.tolist())
    faces = [list(reversed(range(segments))), list(range(segments, 2 * segments))]
    faces.extend([[i, (i + 1) % segments, (i + 1) % segments + segments, i + segments] for i in range(segments)])
    return _geometry(vertices, faces)


def mirrored_geometry(geometry: dict, axis: int, tolerance: float = 1e-6) -> tuple[dict, dict]:
    """Evaluate a non-destructive welded mirror modifier and return source index maps."""
    if axis not in (0, 1, 2):
        raise MeshTopologyError("El eje de Mirror debe ser X, Y o Z.")
    geometry = ensure_editable_geometry(dict(geometry))
    vertices, faces = geometry["vertices"], geometry["faces"]
    coordinates = np.asarray(vertices, dtype=float)
    if not len(coordinates):
        raise MeshTopologyError("No se puede reflejar una malla vacía.")
    positive = bool(np.any(coordinates[:, axis] > tolerance))
    negative = bool(np.any(coordinates[:, axis] < -tolerance))
    if positive and negative:
        raise MeshTopologyError("Mirror requiere una mitad de malla a un solo lado del eje.")
    if not positive and not negative:
        raise MeshTopologyError("La malla no tiene geometría fuera del plano de espejo.")

    reflected_map = {}
    evaluated_vertices = [list(vertex) for vertex in vertices]
    vertex_sources = list(range(len(vertices)))
    for source_id, vertex in enumerate(coordinates):
        if abs(vertex[axis]) <= tolerance:
            reflected_map[source_id] = source_id
        else:
            reflected = vertex.copy()
            reflected[axis] *= -1.0
            reflected_map[source_id] = len(evaluated_vertices)
            evaluated_vertices.append(reflected.tolist())
            vertex_sources.append(source_id)

    evaluated_faces, face_sources = [], []
    for face_id, face in enumerate(faces):
        if all(abs(coordinates[index, axis]) <= tolerance for index in face):
            continue
        evaluated_faces.append(list(face))
        face_sources.append(face_id)
    for face_id, face in enumerate(faces):
        if all(abs(coordinates[index, axis]) <= tolerance for index in face):
            continue
        evaluated_faces.append([reflected_map[index] for index in reversed(face)])
        face_sources.append(face_id)

    evaluated = _geometry(evaluated_vertices, evaluated_faces)
    source_edge_ids = {tuple(edge): index for index, edge in enumerate(geometry["edges"])}
    edge_sources = []
    for start, end in evaluated["edges"]:
        source_edge = tuple(sorted((vertex_sources[start], vertex_sources[end])))
        edge_sources.append(source_edge_ids.get(source_edge, -1))
    maps = {"vertices": vertex_sources, "edges": edge_sources, "faces": face_sources}
    if any(index < 0 for index in edge_sources):
        raise MeshTopologyError("Mirror produjo una arista sin correspondencia en la mitad original.")
    return evaluated, maps


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
