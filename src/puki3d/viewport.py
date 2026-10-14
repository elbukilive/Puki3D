"""Interactive OpenGL scene view with orbit navigation, picking, and transforms."""

from __future__ import annotations

from math import cos, pi, radians, sin

import numpy as np
from copy import deepcopy
from OpenGL.GL import (
    GL_AMBIENT, GL_COLOR_BUFFER_BIT, GL_COLOR_MATERIAL, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_DIFFUSE, GL_LIGHT0, GL_LIGHTING,
    GL_LIGHT_MODEL_AMBIENT, GL_LINES, GL_MODELVIEW, GL_NORMALIZE,
    GL_LINE_LOOP, GL_POINTS, GL_PROJECTION, GL_SMOOTH,
    GL_TRIANGLES, glBegin, glClear, glClearColor,
    glColor3f, glColorMaterial, glDepthFunc, glDisable, glEnable, glEnd,
    glFlush, glLightModelfv, glLightfv, glLineWidth, glLoadMatrixf,
    glMatrixMode, glMultMatrixf, glNormal3f, glPointSize,
    glPopMatrix, glPushMatrix, glShadeModel, glVertex3f, glViewport,
    GL_FRONT_AND_BACK, GL_LEQUAL, GL_LINE, GL_FILL, GL_AMBIENT_AND_DIFFUSE,
)
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .camera import (
    OrbitCamera,
    closest_segment_fraction,
    project_to_screen,
    ray_axis_parameter,
    signed_angle_about_axis,
)
from .scene import (
    MeshTopologyError,
    _triangulate_face,
    ensure_editable_geometry,
    bevel_geometry,
    fill_boundary_geometry,
    bridge_boundary_geometry,
    merge_vertices_geometry,
    mesh_diagnostics,
    inset_face_geometry,
    loop_cut_geometry,
    mirrored_geometry,
    node_matrix,
    raycast,
)


def _silhouette_edge_ids(geometry: dict, eye_local: np.ndarray) -> list[int]:
    """Return real mesh edges on the camera-facing silhouette, excluding triangulation diagonals."""
    vertices = np.asarray(geometry.get("vertices", []), dtype=float)
    faces = geometry.get("faces", [])
    normals = geometry.get("face_normals", [])
    if not len(vertices) or len(faces) != len(normals):
        return []
    facing = []
    for face, normal in zip(faces, normals):
        center = np.mean(vertices[face], axis=0)
        facing.append(float(np.dot(np.asarray(normal), eye_local - center)) > 1e-8)
    result = []
    for edge_id, adjacent in enumerate(geometry.get("edge_faces", [])):
        if len(adjacent) != 2 or facing[adjacent[0]] != facing[adjacent[1]]:
            result.append(edge_id)
    return result


def _snap_axis_amount(start: float, amount: float, targets: list[float], step: float = 0.25,
                      tolerance: float = 0.12) -> float:
    """Snap an axis-constrained translation to the grid or a nearby vertex."""
    snapped = round((start + amount) / step) * step
    target = min(targets, key=lambda value: abs(value - snapped), default=None)
    if target is not None and abs(target - snapped) <= tolerance:
        snapped = target
    return snapped - start


class SceneViewport(QOpenGLWidget):
    selectionChanged = Signal(object)
    transformChanged = Signal(object)
    sceneChanged = Signal()
    modeChanged = Signal(str, str)
    extrusionStateChanged = Signal(bool)
    insetStateChanged = Signal(bool)
    statusMessage = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.nodes: list[dict] = []
        self.selected_id: str | None = None
        self.mode = "object"
        self.component_mode = "face"
        self.selected_components: set[int] = set()
        self.snapping = False
        self._undo_stack: list[tuple[str, str, object]] = []
        self._redo_stack: list[tuple[str, str, object]] = []
        self.tool = "select"
        self.camera = OrbitCamera()
        self._last_mouse = QPoint()
        self._drag_start = QPoint()
        self._drag_transform: dict | None = None
        self._drag_geometry: dict | None = None
        self._drag_geometry_before: dict | None = None
        self._drag_pivot_world: np.ndarray | None = None
        self._extrusion: dict | None = None
        self._extrusion_dragging = False
        self._inset: dict | None = None
        self._inset_dragging = False
        self._loop_cut: dict | None = None
        self._bevel: dict | None = None
        self._operation_dragging = False
        self._dragging_transform = False
        self._drag_axis: int | None = None
        self._drag_axis_start: float | None = None
        self._drag_rotation_start: np.ndarray | None = None
        self._drag_gizmo_length = 1.0
        self._drag_axis_screen: np.ndarray | None = None
        self._hover_axis: int | None = None
        self._orbiting = False
        self._panning = False

    @property
    def selected_node(self) -> dict | None:
        return next((node for node in self.nodes if node.get("id") == self.selected_id), None)

    def set_nodes(self, nodes: list[dict]) -> None:
        self.nodes = nodes
        if self.selected_id and not any(node.get("id") == self.selected_id for node in nodes):
            self.selected_id = None
            self.selectionChanged.emit(None)
        self.update()

    def select_node(self, node_id: str | None) -> None:
        if node_id != self.selected_id:
            self.selected_components.clear()
        self.selected_id = node_id
        self.update()
        self.selectionChanged.emit(self.selected_node)

    def set_tool(self, tool: str) -> None:
        if tool in {"select", "move", "rotate", "scale"}:
            self.tool = tool
            self._hover_axis = None
            self.update()

    def set_mode(self, mode: str) -> None:
        if mode not in {"object", "edit"} or self.mode == mode:
            return
        self.mode = mode
        if mode == "edit" and self.selected_node:
            ensure_editable_geometry(self.selected_node.setdefault("geometry", {}))
        self.selected_components.clear()
        self._hover_axis = None
        self.modeChanged.emit(self.mode, self.component_mode)
        self.update()

    def set_component_mode(self, mode: str) -> None:
        if mode not in {"vertex", "edge", "face"}:
            return
        self.component_mode = mode
        self.selected_components.clear()
        self.modeChanged.emit(self.mode, self.component_mode)
        self.update()

    def select_all_components(self) -> None:
        node = self.selected_node
        if self.mode != "edit" or node is None:
            return
        geometry, source_maps = self._evaluated_geometry(node)
        if self.component_mode == "vertex":
            self.selected_components = set(source_maps["vertices"])
        elif self.component_mode == "edge":
            self.selected_components = set(source_maps["edges"])
        else:
            self.selected_components = set(source_maps["faces"])
        self.update()

    def _component_edges(self, geometry: dict) -> list[tuple[int, int]]:
        if "edges" in geometry:
            return [tuple(edge) for edge in geometry["edges"]]
        edges = set()
        for face in geometry.get("faces", []):
            for start, end in zip(face, face[1:] + face[:1]):
                edges.add(tuple(sorted((start, end))))
        return sorted(edges)

    def _selected_vertex_indices(self, geometry: dict) -> set[int]:
        if self.component_mode == "vertex":
            return set(self.selected_components)
        if self.component_mode == "edge":
            edges = self._component_edges(geometry)
            return {index for edge_id in self.selected_components if edge_id < len(edges) for index in edges[edge_id]}
        faces = geometry.get("faces", [])
        return {index for face_id in self.selected_components if face_id < len(faces) for index in faces[face_id]}

    def _component_pivot(self, node: dict) -> np.ndarray:
        geometry = ensure_editable_geometry(node.setdefault("geometry", {}))
        indices = self._selected_vertex_indices(geometry)
        if not indices:
            return np.asarray(node["transform"]["position"], dtype=float)
        local = np.mean([geometry["vertices"][index] for index in sorted(indices)], axis=0)
        return (node_matrix(node) @ np.append(local, 1.0))[:3]

    def _snapshot_geometry(self) -> dict | None:
        node = self.selected_node
        if node is None:
            return None
        import copy
        return copy.deepcopy(node.get("geometry", {}))

    def _push_edit_history(self, node_id: str, kind: str, value: object) -> None:
        self._undo_stack.append((node_id, kind, deepcopy(value)))
        self._undo_stack = self._undo_stack[-100:]
        self._redo_stack.clear()

    def _record_geometry_change(self, before: dict | None) -> None:
        if before is None or self.selected_node is None:
            return
        after = self._snapshot_geometry()
        if before != after:
            self._push_edit_history(self.selected_id, "geometry", before)
            self.sceneChanged.emit()
            self.statusMessage.emit(self._topology_status(self.selected_node))

    def _selected_boundary_edge_ids(self, geometry: dict) -> set[int]:
        if self.component_mode == "edge":
            return set(self.selected_components)
        if self.component_mode == "vertex":
            return {edge_id for edge_id, (a, b) in enumerate(geometry["edges"])
                    if a in self.selected_components and b in self.selected_components
                    and len(geometry["edge_faces"][edge_id]) == 1}
        return set()

    def _apply_topology_operation(self, operation, *, selection: set[int] | None = None) -> None:
        node = self.selected_node
        if self.mode != "edit" or node is None:
            return
        before = self._snapshot_geometry()
        try:
            node["geometry"] = operation(before)
        except MeshTopologyError as exc:
            self.statusMessage.emit(str(exc))
            return
        self.selected_components = set(selection or ())
        self._record_geometry_change(before)
        self.update()
        self.statusMessage.emit(self._topology_status(node))

    def _topology_status(self, node: dict) -> str:
        diagnostics = mesh_diagnostics(node.get("geometry", {}))
        return " · ".join(item["message"] for item in diagnostics if item["severity"] != "ok") or diagnostics[0]["message"]

    def fill_selected_boundary(self) -> None:
        node = self.selected_node
        if node is None:
            return
        geometry = ensure_editable_geometry(node.get("geometry", {}))
        edges = self._selected_boundary_edge_ids(geometry)
        self._apply_topology_operation(lambda before: fill_boundary_geometry(before, edges))

    def bridge_selected_boundaries(self) -> None:
        node = self.selected_node
        if node is None:
            return
        geometry = ensure_editable_geometry(node.get("geometry", {}))
        edges = self._selected_boundary_edge_ids(geometry)
        self._apply_topology_operation(lambda before: bridge_boundary_geometry(before, edges))

    def merge_selected_vertices(self) -> None:
        if self.component_mode != "vertex":
            self.statusMessage.emit("Merge requiere el modo de vértices.")
            return
        selected = set(self.selected_components)
        self._apply_topology_operation(lambda before: merge_vertices_geometry(before, selected))

    def _record_mirror_change(self, node: dict, before: dict | None) -> None:
        if before != node.get("mirror"):
            self._push_edit_history(node["id"], "mirror", before)
            self.sceneChanged.emit()

    def undo_edit(self) -> None:
        if not self._undo_stack:
            return
        node_id, kind, value = self._undo_stack.pop()
        node = next((candidate for candidate in self.nodes if candidate.get("id") == node_id), None)
        if node is None:
            return
        current = deepcopy(node.get(kind))
        self._redo_stack.append((node_id, kind, current))
        if kind == "mirror" and value is None:
            node.pop("mirror", None)
        else:
            node[kind] = value
        self.update()
        self.sceneChanged.emit()
        if kind == "geometry":
            self.statusMessage.emit(self._topology_status(node))

    def redo_edit(self) -> None:
        if not self._redo_stack:
            return
        node_id, kind, value = self._redo_stack.pop()
        node = next((candidate for candidate in self.nodes if candidate.get("id") == node_id), None)
        if node is None:
            return
        current = deepcopy(node.get(kind))
        self._undo_stack.append((node_id, kind, current))
        if kind == "mirror" and value is None:
            node.pop("mirror", None)
        else:
            node[kind] = value
        self.update()
        self.sceneChanged.emit()
        if kind == "geometry":
            self.statusMessage.emit(self._topology_status(node))

    def _evaluated_geometry(self, node: dict) -> tuple[dict, dict]:
        source = ensure_editable_geometry(node.setdefault("geometry", {}))
        mirror = node.get("mirror")
        if mirror is None:
            return source, {
                "vertices": list(range(len(source["vertices"]))),
                "edges": list(range(len(source["edges"]))),
                "faces": list(range(len(source["faces"]))),
            }
        return mirrored_geometry(source, int(mirror["axis"]))

    def set_mirror_axis(self, axis: int | None) -> None:
        node = self.selected_node
        if node is None or self.mode != "edit":
            return
        before = deepcopy(node.get("mirror"))
        if axis is None or (node.get("mirror") and int(node["mirror"].get("axis", -1)) == axis):
            node.pop("mirror", None)
        else:
            mirrored_geometry(node.get("geometry", {}), int(axis))
            node["mirror"] = {"axis": int(axis), "merge": True, "clip": True}
        self._record_mirror_change(node, before)
        self.update()

    def loop_cut_selected_edge(self) -> None:
        node = self.selected_node
        if self.mode != "edit" or self.component_mode != "edge" or node is None or len(self.selected_components) != 1:
            return
        before = self._snapshot_geometry()
        edge_id = next(iter(self.selected_components))
        try:
            preview = loop_cut_geometry(before, edge_id, 0.5)
        except MeshTopologyError:
            return
        self._loop_cut = {"node_id": node["id"], "before": before, "edge_id": edge_id,
                          "preview": preview, "start_mouse": QPoint(self._last_mouse), "fraction": 0.5}
        node["geometry"] = preview
        self.update()

    def _set_loop_fraction(self, fraction: float) -> None:
        state, node = self._loop_cut, self.selected_node
        if state is None or node is None or node["id"] != state["node_id"]:
            return
        fraction = float(np.clip(fraction, 0.05, 0.95))
        try:
            state["preview"] = loop_cut_geometry(state["before"], state["edge_id"], fraction)
        except MeshTopologyError:
            return
        state["fraction"] = fraction
        node["geometry"] = deepcopy(state["preview"])
        self.update()

    def bevel_selected_edges(self) -> None:
        node = self.selected_node
        if self.mode != "edit" or self.component_mode != "edge" or node is None or not self.selected_components:
            return
        before = self._snapshot_geometry()
        try:
            preview = bevel_geometry(before, self.selected_components, 0.01)
        except MeshTopologyError:
            return
        self._bevel = {"node_id": node["id"], "before": before, "edge_ids": set(self.selected_components),
                       "preview": preview, "start_mouse": QPoint(self._last_mouse), "width": 0.01}
        node["geometry"] = preview
        self.update()

    def _set_bevel_width(self, width: float) -> None:
        state, node = self._bevel, self.selected_node
        if state is None or node is None or node["id"] != state["node_id"]:
            return
        width = max(0.0, float(width))
        if width <= 1e-6:
            node["geometry"] = deepcopy(state["before"])
            state["width"] = 0.0
            self.update()
            return
        try:
            state["preview"] = bevel_geometry(state["before"], state["edge_ids"], width)
        except MeshTopologyError:
            return
        state["width"] = width
        node["geometry"] = deepcopy(state["preview"])
        self.update()

    def _finish_operation(self, kind: str) -> None:
        state = getattr(self, f"_{kind}")
        if state is None:
            return
        node = next((candidate for candidate in self.nodes if candidate.get("id") == state["node_id"]), None)
        setattr(self, f"_{kind}", None)
        self._operation_dragging = False
        if node is not None and state.get("width", state.get("fraction", 0.0)) > 0:
            node["geometry"] = state["preview"]
            self.selected_components.clear()
            self._push_edit_history(state["node_id"], "geometry", state["before"])
            self.sceneChanged.emit()
            self.statusMessage.emit(self._topology_status(node))
        elif node is not None:
            node["geometry"] = state["before"]
        self.update()

    def _cancel_operation(self, kind: str) -> None:
        state = getattr(self, f"_{kind}")
        if state is None:
            return
        node = next((candidate for candidate in self.nodes if candidate.get("id") == state["node_id"]), None)
        setattr(self, f"_{kind}", None)
        self._operation_dragging = False
        if node is not None:
            node["geometry"] = state["before"]
        self.update()

    def inset_selected_face(self) -> None:
        node = self.selected_node
        if self._inset is not None or self._extrusion is not None:
            return
        if self.mode != "edit" or self.component_mode != "face" or node is None or len(self.selected_components) != 1:
            return
        face_id = next(iter(self.selected_components))
        baseline = self._snapshot_geometry()
        try:
            inset_face_geometry(baseline, face_id, 0.001)
        except MeshTopologyError:
            return
        self._inset = {
            "node_id": node["id"],
            "before": baseline,
            "face_id": face_id,
            "start_mouse": QPoint(self._last_mouse),
            "amount": 0.0,
        }
        self._inset_dragging = False
        self.insetStateChanged.emit(True)
        self.update()

    @property
    def insetting(self) -> bool:
        return self._inset is not None

    def _set_inset_amount(self, amount: float) -> None:
        state = self._inset
        node = self.selected_node
        if state is None or node is None or node.get("id") != state["node_id"]:
            return
        amount = max(0.0, float(amount))
        if amount <= 1e-6:
            node["geometry"] = deepcopy(state["before"])
            state["amount"] = 0.0
            self.update()
            return
        try:
            node["geometry"] = inset_face_geometry(state["before"], state["face_id"], amount)
        except MeshTopologyError:
            return
        state["amount"] = amount
        self.update()

    def _update_inset(self, point: QPoint) -> None:
        state = self._inset
        if state is None:
            return
        amount = (point.x() - state["start_mouse"].x()) * self.camera.distance / max(self.height(), 1)
        self._set_inset_amount(amount)

    def _finish_inset(self) -> None:
        state = self._inset
        if state is None:
            return
        self._inset = None
        self._inset_dragging = False
        self.insetStateChanged.emit(False)
        if state["amount"] > 1e-6:
            self._push_edit_history(state["node_id"], "geometry", state["before"])
            self.sceneChanged.emit()
            self.statusMessage.emit(self._topology_status(self.selected_node))
        self.update()

    def _cancel_inset(self) -> None:
        state = self._inset
        if state is None:
            return
        node = next((candidate for candidate in self.nodes if candidate.get("id") == state["node_id"]), None)
        self._inset = None
        self._inset_dragging = False
        if node is not None:
            node["geometry"] = state["before"]
        self.insetStateChanged.emit(False)
        self.update()

    def delete_selected_components(self) -> None:
        node = self.selected_node
        if self.mode != "edit" or node is None or not self.selected_components:
            return
        geometry = ensure_editable_geometry(node.setdefault("geometry", {}))
        before = self._snapshot_geometry()
        vertices, faces = geometry["vertices"], geometry["faces"]
        if self.component_mode == "face":
            geometry["faces"] = [face for index, face in enumerate(faces) if index not in self.selected_components]
        elif self.component_mode == "vertex":
            removed = set(self.selected_components)
            kept = [index for index in range(len(vertices)) if index not in removed]
            remap = {old: new for new, old in enumerate(kept)}
            geometry["vertices"] = [vertices[index] for index in kept]
            geometry["faces"] = [[remap[index] for index in face] for face in faces if all(index in remap for index in face)]
        else:
            edges = self._component_edges(geometry)
            selected_edges = {edges[index] for index in self.selected_components if index < len(edges)}
            geometry["faces"] = [face for face in faces if not any(tuple(sorted(edge)) in selected_edges for edge in zip(face, face[1:] + face[:1]))]
        from .scene import _geometry
        node["geometry"] = _geometry(geometry["vertices"], geometry["faces"])
        self.selected_components.clear()
        self.update()
        self._record_geometry_change(before)

    def extrude_selected_face(self) -> None:
        """Start a mouse-controlled face extrusion preview; click/Enter confirms."""
        node = self.selected_node
        if self._extrusion is not None:
            return
        if self.mode != "edit" or self.component_mode != "face" or node is None or len(self.selected_components) != 1:
            return
        geometry = ensure_editable_geometry(node.setdefault("geometry", {}))
        faces, vertices = geometry["faces"], geometry["vertices"]
        face_id = next(iter(self.selected_components))
        if face_id >= len(faces):
            return
        face = faces[face_id]
        points = [np.asarray(vertices[index], dtype=float) for index in face]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            return
        normal /= length
        baseline = deepcopy(geometry)
        world_matrix = node_matrix(node)
        world_normal = world_matrix[:3, :3] @ normal
        world_normal /= np.linalg.norm(world_normal)
        pivot_local = np.mean(points, axis=0)
        pivot_world = (world_matrix @ np.append(pivot_local, 1.0))[:3]
        screen_start = self._project(pivot_world)
        screen_end = self._project(pivot_world + world_normal)
        screen_axis = None
        pixels_per_unit = 0.0
        if screen_start is not None and screen_end is not None:
            screen_vector = screen_end - screen_start
            pixels_per_unit = float(np.linalg.norm(screen_vector))
            if pixels_per_unit > 1e-3:
                screen_axis = screen_vector / pixels_per_unit
        ray_origin, ray_direction = self._ray_for_point(self._last_mouse)
        start_parameter = ray_axis_parameter(ray_origin, ray_direction, pivot_world, world_normal)
        original_face = list(face)
        new_face = list(range(len(vertices), len(vertices) + len(face)))
        new_faces = [list(value) for value in faces]
        new_faces[face_id] = new_face
        side_faces = [[face[i], face[(i + 1) % len(face)], new_face[(i + 1) % len(face)], new_face[i]] for i in range(len(face))]
        new_faces.extend(side_faces)
        new_vertices = [list(vertex) for vertex in vertices] + [list(vertices[index]) for index in face]
        self._extrusion = {
            "node_id": node["id"],
            "before": baseline,
            "vertices": new_vertices,
            "faces": new_faces,
            "face_id": face_id,
            "original_face": original_face,
            "new_face": new_face,
            "world_normal": world_normal,
            "pivot_world": pivot_world,
            "start_mouse": QPoint(self._last_mouse),
            "start_parameter": start_parameter,
            "screen_axis": screen_axis,
            "pixels_per_unit": pixels_per_unit,
            "distance": 0.0,
        }
        self._set_extrusion_distance(0.0)
        self.extrusionStateChanged.emit(True)
        self.update()

    @property
    def extruding(self) -> bool:
        return self._extrusion is not None

    def _set_extrusion_distance(self, distance: float) -> None:
        state = self._extrusion
        node = self.selected_node
        if state is None or node is None or node.get("id") != state["node_id"]:
            return
        distance = float(np.clip(distance, -100.0, 100.0))
        if abs(distance) <= 1e-6:
            node["geometry"] = deepcopy(state["before"])
            state["distance"] = 0.0
            self.update()
            return
        baseline_vertices = state["vertices"]
        geometry_vertices = [list(vertex) for vertex in baseline_vertices[:len(state["vertices"]) - len(state["new_face"])]]
        matrix = node_matrix(node)
        inverse_linear = np.linalg.inv(matrix[:3, :3])
        local_delta = inverse_linear @ (state["world_normal"] * distance)
        for new_index, source_index in zip(state["new_face"], state["original_face"]):
            geometry_vertices.append((np.asarray(baseline_vertices[new_index], dtype=float) + local_delta).tolist())
        from .scene import _geometry
        node["geometry"] = _geometry(geometry_vertices, state["faces"])
        state["distance"] = distance
        self.update()

    def _update_extrusion(self, point: QPoint) -> None:
        state = self._extrusion
        if state is None:
            return
        origin, direction = self._ray_for_point(point)
        pivot = state["pivot_world"]
        parameter = ray_axis_parameter(origin, direction, pivot, state["world_normal"])
        start_parameter = state["start_parameter"]
        if state["screen_axis"] is not None and state["pixels_per_unit"] > 1e-3:
            delta = np.array([point.x() - state["start_mouse"].x(), point.y() - state["start_mouse"].y()], dtype=float)
            distance = float(np.dot(delta, state["screen_axis"])) / max(state["pixels_per_unit"], 1e-3)
        elif parameter is not None and start_parameter is not None:
            distance = parameter - start_parameter
        else:
            delta_y = state["start_mouse"].y() - point.y()
            distance = delta_y * self.camera.distance / max(self.height(), 1)
        self._set_extrusion_distance(distance)

    def _finish_extrusion(self) -> None:
        state = self._extrusion
        if state is None:
            return
        before = state["before"]
        self._extrusion = None
        self._extrusion_dragging = False
        self.extrusionStateChanged.emit(False)
        if abs(state["distance"]) < 1e-6:
            node = next((candidate for candidate in self.nodes if candidate.get("id") == state["node_id"]), None)
            if node is not None:
                node["geometry"] = before
            self.update()
            return
        self._push_edit_history(state["node_id"], "geometry", before)
        self.sceneChanged.emit()
        self.statusMessage.emit(self._topology_status(self.selected_node))

    def _cancel_extrusion(self) -> None:
        state = self._extrusion
        if state is None:
            return
        node = next((candidate for candidate in self.nodes if candidate.get("id") == state["node_id"]), None)
        self._extrusion = None
        self._extrusion_dragging = False
        if node is not None:
            node["geometry"] = state["before"]
        self.extrusionStateChanged.emit(False)
        self.update()

    def _pick_component(self, point: QPoint) -> int | None:
        node = self.selected_node
        if node is None:
            return None
        source = ensure_editable_geometry(node.setdefault("geometry", {}))
        try:
            geometry, maps = self._evaluated_geometry(node)
        except MeshTopologyError:
            geometry = source
            maps = {
                "vertices": list(range(len(source["vertices"]))),
                "edges": list(range(len(source["edges"]))),
                "faces": list(range(len(source["faces"]))),
            }
        vertices = geometry["vertices"]
        mouse = np.array([point.x(), point.y()], dtype=float)
        if self.component_mode == "vertex":
            projected = [(maps["vertices"][index], self._project((node_matrix(node) @ np.append(vertex, 1.0))[:3])) for index, vertex in enumerate(vertices)]
            values = [(source_id, float(np.linalg.norm(screen - mouse))) for source_id, screen in projected if screen is not None]
            if values and min(values, key=lambda item: item[1])[1] <= 14:
                return min(values, key=lambda item: item[1])[0]
        elif self.component_mode == "edge":
            best = (None, 10.0)
            for edge_id, (start, end) in enumerate(self._component_edges(geometry)):
                a = self._project((node_matrix(node) @ np.append(vertices[start], 1.0))[:3])
                b = self._project((node_matrix(node) @ np.append(vertices[end], 1.0))[:3])
                if a is not None and b is not None:
                    distance, _ = closest_segment_fraction(mouse, a, b)
                    if distance < best[1]:
                        best = (maps["edges"][edge_id], distance)
            return best[0]
        else:
            ray_origin, ray_direction = self._ray_for_point(point)
            inverse = np.linalg.inv(node_matrix(node))
            local_origin = (inverse @ np.append(ray_origin, 1.0))[:3]
            local_direction = (inverse @ np.append(ray_direction, 0.0))[:3]
            closest, face_hit = float("inf"), None
            for face_id, face in enumerate(geometry.get("faces", [])):
                for triangle in _triangulate_face(vertices, face):
                    a, b, c = (np.asarray(vertices[index], dtype=float) for index in triangle)
                    edge1, edge2 = b - a, c - a
                    h = np.cross(local_direction, edge2)
                    determinant = float(np.dot(edge1, h))
                    if abs(determinant) < 1e-9:
                        continue
                    inverse_det = 1.0 / determinant
                    s = local_origin - a
                    u = inverse_det * float(np.dot(s, h))
                    q = np.cross(s, edge1)
                    v = inverse_det * float(np.dot(local_direction, q))
                    distance = inverse_det * float(np.dot(edge2, q))
                    if u >= 0 and v >= 0 and u + v <= 1 and distance > 0 and distance < closest:
                        closest, face_hit = distance, maps["faces"][face_id]
            return face_hit
        return None

    def _apply_component_transform(self, point: QPoint) -> None:
        node = self.selected_node
        if node is None or self._drag_geometry is None or self._drag_axis is None:
            return
        geometry = ensure_editable_geometry(node.setdefault("geometry", {}))
        indices = self._selected_vertex_indices(geometry)
        if not indices:
            return
        axis_index = self._drag_axis
        axis = self._axis_vector(axis_index)
        origin, direction = self._ray_for_point(point)
        pivot_world = self._drag_pivot_world
        if self.tool == "rotate":
            current = self._ray_plane_vector(point, pivot_world, axis)
            if current is None or self._drag_rotation_start is None:
                return
            amount = signed_angle_about_axis(self._drag_rotation_start, current, axis)
        else:
            parameter = ray_axis_parameter(origin, direction, pivot_world, axis)
            amount = parameter - self._drag_axis_start if parameter is not None and self._drag_axis_start is not None else (point.x() - self._drag_start.x()) * 0.01
        if self.snapping and self.tool == "move":
            if len(indices) == 1:
                selected_index = next(iter(indices))
                selected_world = (matrix @ np.append(baseline[selected_index], 1.0))[:3]
                targets = []
                for index, vertex in enumerate(baseline):
                    if index in indices:
                        continue
                    world = (matrix @ np.append(vertex, 1.0))[:3]
                    targets.append(float(np.dot(world, axis)))
                amount = _snap_axis_amount(float(np.dot(selected_world, axis)), amount, targets)
            else:
                amount = round(amount / 0.25) * 0.25
        baseline = self._drag_geometry["vertices"]
        matrix = node_matrix(node)
        inverse = np.linalg.inv(matrix)
        transform = np.eye(3)
        if self.tool == "rotate":
            c, s = cos(radians(amount)), sin(radians(amount))
            if axis_index == 0: transform = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis_index == 1: transform = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else: transform = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        elif self.tool == "scale":
            transform[axis_index, axis_index] = float(np.exp(np.clip(amount * 0.5, -4, 4)))
        for index in indices:
            original = np.asarray(baseline[index], dtype=float)
            world = (matrix @ np.append(original, 1.0))[:3]
            if self.tool == "move":
                world += axis * amount
            else:
                world = pivot_world + transform @ (world - pivot_world)
            geometry["vertices"][index] = (inverse @ np.append(world, 1.0))[:3].tolist()
        mirror = node.get("mirror")
        if mirror and mirror.get("clip", True):
            mirror_axis = int(mirror["axis"])
            for index, original in enumerate(baseline):
                source_side = float(original[mirror_axis])
                moved_side = float(geometry["vertices"][index][mirror_axis])
                if abs(source_side) <= 1e-6 or source_side * moved_side < 0:
                    geometry["vertices"][index][mirror_axis] = 0.0
        self.update()

    def focus_selected(self) -> None:
        node = self.selected_node
        if node is None:
            return
        self.camera.target = np.asarray(node["transform"]["position"], dtype=float).copy()
        self.update()

    def initializeGL(self) -> None:  # noqa: N802 - Qt API
        glClearColor(0.055, 0.060, 0.073, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.28, 0.30, 0.34, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.20, 0.20, 0.22, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.82, 0.82, 0.78, 1.0])

    def resizeGL(self, width: int, height: int) -> None:  # noqa: N802 - Qt API
        glViewport(0, 0, max(width, 1), max(height, 1))

    def paintGL(self) -> None:  # noqa: N802 - Qt API
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        width, height = max(self.width(), 1), max(self.height(), 1)
        projection = self.camera.projection_matrix(width / height)
        view, eye, right, camera_up = self.camera.view_matrix()
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(np.asarray(projection.T, dtype=np.float32))
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(np.asarray(view.T, dtype=np.float32))
        glLightfv(GL_LIGHT0, 0x1203, [eye[0], eye[1], eye[2], 1.0])  # GL_POSITION
        self._draw_grid()
        self._draw_nodes()
        selected = self.selected_node
        if selected:
            self._draw_selection(selected, right, camera_up)
        glFlush()

    def _draw_grid(self) -> None:
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for coordinate in range(-24, 25):
            if coordinate == 0:
                continue
            glColor3f(0.15, 0.17, 0.20)
            glVertex3f(float(coordinate), -24.0, 0.0)
            glVertex3f(float(coordinate), 24.0, 0.0)
            glVertex3f(-24.0, float(coordinate), 0.0)
            glVertex3f(24.0, float(coordinate), 0.0)
        glColor3f(0.92, 0.28, 0.31)
        glVertex3f(-24.0, 0.0, 0.0)
        glVertex3f(24.0, 0.0, 0.0)
        glColor3f(0.35, 0.82, 0.46)
        glVertex3f(0.0, -24.0, 0.0)
        glVertex3f(0.0, 24.0, 0.0)
        glColor3f(0.35, 0.57, 1.0)
        glVertex3f(0.0, 0.0, -0.01)
        glVertex3f(0.0, 0.0, 4.0)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_nodes(self) -> None:
        colors = {
            "cube": (0.55, 0.61, 0.69), "sphere": (0.63, 0.68, 0.75),
            "cylinder": (0.58, 0.65, 0.72), "plane": (0.45, 0.50, 0.57),
        }
        for node in self.nodes:
            try:
                geometry, _ = self._evaluated_geometry(node)
            except MeshTopologyError:
                geometry = node.get("geometry", {})
            vertices = geometry.get("vertices", [])
            triangles = geometry.get("triangles", [])
            if not vertices or not triangles:
                continue
            glPushMatrix()
            glMultMatrixf(np.asarray(node_matrix(node).T, dtype=np.float32))
            glColor3f(*colors.get(node.get("primitive"), colors["cube"]))
            glBegin(GL_TRIANGLES)
            for triangle in triangles:
                a, b, c = (np.asarray(vertices[index], dtype=float) for index in triangle)
                normal = np.cross(b - a, c - a)
                length = np.linalg.norm(normal)
                if length:
                    normal /= length
                glNormal3f(float(normal[0]), float(normal[1]), float(normal[2]))
                for index in triangle:
                    point = vertices[index]
                    glVertex3f(float(point[0]), float(point[1]), float(point[2]))
            glEnd()
            glPopMatrix()

    def _draw_selection(self, node: dict, camera_right: np.ndarray, camera_up: np.ndarray) -> None:
        try:
            geometry, _ = self._evaluated_geometry(node)
        except MeshTopologyError:
            geometry = node.get("geometry", {})
        vertices = geometry.get("vertices", [])
        triangles = geometry.get("triangles", [])
        if not vertices or not triangles:
            return
        if self.mode == "object":
            glDisable(GL_LIGHTING)
            glPushMatrix()
            matrix = node_matrix(node)
            glMultMatrixf(np.asarray(matrix.T, dtype=np.float32))
            glColor3f(0.82, 1.0, 0.31)
            glLineWidth(1.5)
            glBegin(GL_LINES)
            _, eye, _, _ = self.camera.view_matrix()
            eye_local = (np.linalg.inv(matrix) @ np.append(eye, 1.0))[:3]
            edges = geometry.get("edges", [])
            for edge_id in _silhouette_edge_ids(geometry, eye_local):
                for index in edges[edge_id]:
                    point = vertices[index]
                    glVertex3f(float(point[0]), float(point[1]), float(point[2]))
            glEnd()
            glPopMatrix()
        if self.mode == "edit" and node.get("id") == self.selected_id:
            self._draw_edit_components(node)
        if self.tool in {"move", "rotate", "scale"}:
            origin = self._component_pivot(node) if self.mode == "edit" else np.asarray(node["transform"]["position"], dtype=float)
            if self.tool == "rotate":
                self._draw_rotation_gizmo(origin)
            else:
                self._draw_axis_gizmo(origin, camera_right, camera_up)
        glEnable(GL_LIGHTING)

    def _draw_edit_components(self, node: dict) -> None:
        source = ensure_editable_geometry(node.setdefault("geometry", {}))
        try:
            geometry, maps = self._evaluated_geometry(node)
        except MeshTopologyError:
            geometry = source
            maps = {
                "vertices": list(range(len(source["vertices"]))),
                "edges": list(range(len(source["edges"]))),
                "faces": list(range(len(source["faces"]))),
            }
        vertices, faces = geometry["vertices"], geometry["faces"]
        matrix = node_matrix(node)
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glMultMatrixf(np.asarray(matrix.T, dtype=np.float32))
        glLineWidth(1.5)
        glBegin(GL_LINES)
        edge_ids = self._component_edges(geometry)
        for edge_id, (start, end) in enumerate(edge_ids):
            source_edge = maps["edges"][edge_id]
            selected = source_edge in self.selected_components if self.component_mode == "edge" else False
            glColor3f(*( (1.0, 0.95, 0.30) if selected else (0.10, 0.12, 0.15) ))
            for index in (start, end):
                glVertex3f(*map(float, vertices[index]))
        glEnd()
        if self.component_mode == "face":
            glColor3f(1.0, 0.88, 0.24)
            glLineWidth(3.0)
            glBegin(GL_LINES)
            for face_id, face in enumerate(faces):
                if maps["faces"][face_id] in self.selected_components:
                    for start, end in zip(face, face[1:] + face[:1]):
                        glVertex3f(*map(float, vertices[start]))
                        glVertex3f(*map(float, vertices[end]))
            glEnd()
        glPointSize(9.0)
        glBegin(GL_POINTS)
        for vertex_id, vertex in enumerate(vertices):
            source_vertex = maps["vertices"][vertex_id]
            if self.component_mode == "vertex":
                selected = source_vertex in self.selected_components
            else:
                selected = source_vertex in self._selected_vertex_indices(source)
            glColor3f(*( (1.0, 0.93, 0.24) if selected else (0.20, 0.76, 1.0) ))
            glVertex3f(*map(float, vertex))
        glEnd()
        glPopMatrix()
        glEnable(GL_LIGHTING)

    @property
    def _gizmo_length(self) -> float:
        return max(0.7, self.camera.distance * 0.12)

    @staticmethod
    def _axis_vector(axis_index: int) -> np.ndarray:
        axis = np.zeros(3, dtype=float)
        axis[axis_index] = 1.0
        return axis

    def _draw_axis_gizmo(self, origin: np.ndarray, camera_right: np.ndarray, camera_up: np.ndarray) -> None:
        length = self._gizmo_length
        for index, color in enumerate(((0.95, 0.30, 0.32), (0.36, 0.88, 0.48), (0.40, 0.62, 1.0))):
            axis = self._axis_vector(index)
            end = origin + axis * length
            active = index == (self._drag_axis if self._dragging_transform else self._hover_axis)
            glLineWidth(4.2 if active else 2.4)
            glColor3f(*((1.0, 0.96, 0.48) if active else color))
            glBegin(GL_LINES)
            glVertex3f(*map(float, origin))
            glVertex3f(*map(float, end))
            if self.tool == "move":
                side = np.cross(axis, camera_up)
                if np.linalg.norm(side) < 1e-6:
                    side = np.cross(axis, camera_right)
                side /= np.linalg.norm(side)
                base = end - axis * length * 0.16
                glVertex3f(*map(float, end))
                glVertex3f(*map(float, base + side * length * 0.07))
                glVertex3f(*map(float, end))
                glVertex3f(*map(float, base - side * length * 0.07))
            glEnd()
        if self.tool == "scale":
            glPointSize(10.0)
            glBegin(GL_POINTS)
            for index, color in enumerate(((0.95, 0.30, 0.32), (0.36, 0.88, 0.48), (0.40, 0.62, 1.0))):
                active = index == (self._drag_axis if self._dragging_transform else self._hover_axis)
                glColor3f(*((1.0, 0.96, 0.48) if active else color))
                end = origin + self._axis_vector(index) * length
                glVertex3f(*map(float, end))
            glEnd()

    def _rotation_ring_points(self, origin: np.ndarray, axis_index: int) -> list[np.ndarray]:
        radius = max(0.55, self.camera.distance * 0.095)
        plane_axes = ((1, 2), (0, 2), (0, 1))[axis_index]
        points = []
        for step in range(65):
            angle = 2 * pi * step / 64
            point = np.asarray(origin, dtype=float).copy()
            point[plane_axes[0]] += radius * cos(angle)
            point[plane_axes[1]] += radius * sin(angle)
            points.append(point)
        return points

    def _draw_rotation_gizmo(self, origin: np.ndarray) -> None:
        colors = ((0.95, 0.30, 0.32), (0.36, 0.88, 0.48), (0.40, 0.62, 1.0))
        for index, color in enumerate(colors):
            active = index == (self._drag_axis if self._dragging_transform else self._hover_axis)
            glLineWidth(4.0 if active else 2.2)
            glColor3f(*((1.0, 0.96, 0.48) if active else color))
            glBegin(GL_LINE_LOOP)
            for point in self._rotation_ring_points(origin, index)[:-1]:
                glVertex3f(*map(float, point))
            glEnd()

    def _project(self, point: np.ndarray) -> np.ndarray | None:
        view, _, _, _ = self.camera.view_matrix()
        projection = self.camera.projection_matrix(max(self.width(), 1) / max(self.height(), 1))
        return project_to_screen(point, view, projection, max(self.width(), 1), max(self.height(), 1))

    def _pick_gizmo_axis(self, point: QPoint) -> int | None:
        node = self.selected_node
        if node is None or self.tool not in {"move", "rotate", "scale"}:
            return None
        mouse = np.array([point.x(), point.y()], dtype=float)
        origin = self._component_pivot(node) if self.mode == "edit" else np.asarray(node["transform"]["position"], dtype=float)
        best_axis, best_distance = None, 11.0
        if self.tool == "rotate":
            for axis_index in range(3):
                projected = [self._project(value) for value in self._rotation_ring_points(origin, axis_index)]
                for start, end in zip(projected, projected[1:]):
                    if start is None or end is None:
                        continue
                    distance, _ = closest_segment_fraction(mouse, start, end)
                    if distance < best_distance:
                        best_axis, best_distance = axis_index, distance
            return best_axis

        start = self._project(origin)
        if start is None:
            return None
        for axis_index in range(3):
            end = self._project(origin + self._axis_vector(axis_index) * self._gizmo_length)
            if end is None:
                continue
            distance, fraction = closest_segment_fraction(mouse, start, end)
            if fraction >= 0.12 and distance < best_distance:
                best_axis, best_distance = axis_index, distance
        return best_axis

    def _ray_plane_vector(self, point: QPoint, pivot: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
        origin, direction = self._ray_for_point(point)
        denominator = float(np.dot(direction, normal))
        if abs(denominator) < 1e-7:
            return None
        distance = float(np.dot(pivot - origin, normal) / denominator)
        if distance <= 0:
            return None
        return origin + direction * distance - pivot

    def _ray_for_point(self, point: QPoint) -> tuple[np.ndarray, np.ndarray]:
        x = 2.0 * point.x() / max(self.width(), 1) - 1.0
        y = 1.0 - 2.0 * point.y() / max(self.height(), 1)
        projection = self.camera.projection_matrix(max(self.width(), 1) / max(self.height(), 1))
        view, eye, _, _ = self.camera.view_matrix()
        inverse = np.linalg.inv(projection @ view)
        near = inverse @ np.array([x, y, -1.0, 1.0])
        far = inverse @ np.array([x, y, 1.0, 1.0])
        near, far = near[:3] / near[3], far[:3] / far[3]
        direction = far - near
        direction /= np.linalg.norm(direction)
        return eye, direction

    def _pick(self, point: QPoint) -> str | None:
        origin, direction = self._ray_for_point(point)
        evaluated_nodes = []
        for node in self.nodes:
            try:
                geometry, _ = self._evaluated_geometry(node)
            except MeshTopologyError:
                geometry = node.get("geometry", {})
            evaluated_nodes.append({**node, "geometry": geometry})
        return raycast(evaluated_nodes, origin, direction)

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray]:
        view, _, right, camera_up = self.camera.view_matrix()
        return right, camera_up

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.setFocus()
        self._last_mouse = event.position().toPoint()
        self._drag_start = self._last_mouse
        if self._loop_cut is not None or self._bevel is not None:
            if event.button() == Qt.MouseButton.LeftButton:
                self._operation_dragging = True
            elif event.button() == Qt.MouseButton.RightButton:
                self._cancel_operation("loop_cut" if self._loop_cut is not None else "bevel")
            event.accept()
            return
        if self._extrusion is not None:
            if event.button() == Qt.MouseButton.LeftButton:
                self._extrusion_dragging = True
            elif event.button() == Qt.MouseButton.RightButton:
                self._cancel_extrusion()
            event.accept()
            return
        if self._inset is not None:
            if event.button() == Qt.MouseButton.LeftButton:
                self._inset_dragging = True
            elif event.button() == Qt.MouseButton.RightButton:
                self._cancel_inset()
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._orbiting = not bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._panning = not self._orbiting
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.tool != "select" and self.selected_node:
                axis_index = self._pick_gizmo_axis(self._last_mouse)
                if axis_index is not None:
                    node = self.selected_node
                    self._drag_axis = axis_index
                    self._hover_axis = axis_index
                    if self.mode == "edit":
                        self._drag_geometry_before = self._snapshot_geometry()
                        import copy
                        self._drag_geometry = copy.deepcopy(node["geometry"])
                        self._drag_pivot_world = self._component_pivot(node)
                        pivot = self._drag_pivot_world
                        axis = self._axis_vector(axis_index)
                        ray_origin, ray_direction = self._ray_for_point(self._last_mouse)
                        self._drag_axis_start = ray_axis_parameter(ray_origin, ray_direction, pivot, axis)
                        self._drag_rotation_start = self._ray_plane_vector(self._last_mouse, pivot, axis) if self.tool == "rotate" else None
                        self._dragging_transform = self.tool != "rotate" or self._drag_rotation_start is not None
                        if self._dragging_transform:
                            event.accept()
                            return
                        self._drag_axis = None
                        self._drag_geometry = self._drag_geometry_before = None
                        event.accept()
                        return
                    self._drag_transform = {
                        key: list(node["transform"][key])
                        for key in ("position", "rotation", "scale")
                    }
                    axis = self._axis_vector(axis_index)
                    pivot = np.asarray(node["transform"]["position"], dtype=float)
                    ray_origin, ray_direction = self._ray_for_point(self._last_mouse)
                    if self.tool == "rotate":
                        self._drag_rotation_start = self._ray_plane_vector(self._last_mouse, pivot, axis)
                        if self._drag_rotation_start is None:
                            self._drag_axis = None
                            self._drag_transform = None
                            event.accept()
                            return
                    else:
                        self._drag_axis_start = ray_axis_parameter(ray_origin, ray_direction, pivot, axis)
                        origin_screen = self._project(pivot)
                        end_screen = self._project(pivot + axis * self._gizmo_length)
                        if origin_screen is not None and end_screen is not None:
                            screen_vector = end_screen - origin_screen
                            screen_length = float(np.linalg.norm(screen_vector))
                            self._drag_axis_screen = screen_vector / screen_length if screen_length > 1e-6 else None
                        else:
                            self._drag_axis_screen = None
                        self._drag_gizmo_length = self._gizmo_length
                    self._dragging_transform = True
                    event.accept()
                    return
            hit_id = self._pick(self._last_mouse)
            if self.mode == "edit" and hit_id == self.selected_id and self.selected_node:
                component = self._pick_component(self._last_mouse)
                if component is not None:
                    if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                        self.selected_components.clear()
                    if component in self.selected_components:
                        self.selected_components.remove(component)
                    else:
                        self.selected_components.add(component)
                    self.update()
                    event.accept()
                    return
            if hit_id != self.selected_id:
                self.select_node(hit_id)
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        point = event.position().toPoint()
        delta = point - self._last_mouse
        if self._loop_cut is not None and self._operation_dragging:
            self._set_loop_fraction(0.5 + (point.x() - self._loop_cut["start_mouse"].x()) / 160.0)
        elif self._bevel is not None and self._operation_dragging:
            delta = abs(point.x() - self._bevel["start_mouse"].x())
            self._set_bevel_width(delta * self.camera.distance / max(self.height(), 1))
        elif self._extrusion is not None:
            self._update_extrusion(point)
        elif self._inset is not None:
            self._update_inset(point)
        elif self._orbiting:
            self.camera.orbit(delta.x(), delta.y())
            self._hover_axis = self._pick_gizmo_axis(point)
            self.update()
        elif self._panning:
            self.camera.pan(delta.x(), delta.y(), self.height())
            self._hover_axis = self._pick_gizmo_axis(point)
            self.update()
        elif self._dragging_transform and self.mode == "edit" and self._drag_geometry is not None:
            self._apply_component_transform(point)
        elif self._dragging_transform and self.selected_node and self._drag_transform is not None and self._drag_axis is not None:
            node = self.selected_node
            transform = self._drag_transform
            axis_index = self._drag_axis
            axis = self._axis_vector(axis_index)
            pivot = np.asarray(transform["position"], dtype=float)
            if self.tool == "rotate" and self._drag_rotation_start is not None:
                current = self._ray_plane_vector(point, pivot, axis)
                if current is None:
                    self._last_mouse = point
                    event.accept()
                    return
                angle = signed_angle_about_axis(self._drag_rotation_start, current, axis)
                rotation = list(transform["rotation"])
                rotation[axis_index] = (rotation[axis_index] + angle) % 360.0
                node["transform"]["rotation"] = rotation
            else:
                ray_origin, ray_direction = self._ray_for_point(point)
                current_parameter = ray_axis_parameter(ray_origin, ray_direction, pivot, axis)
                if current_parameter is not None and self._drag_axis_start is not None:
                    amount = current_parameter - self._drag_axis_start
                elif self._drag_axis_screen is not None:
                    mouse_delta = np.array([point.x() - self._drag_start.x(), point.y() - self._drag_start.y()])
                    pixel_delta = float(np.dot(mouse_delta, self._drag_axis_screen))
                    pixel_length = float(np.linalg.norm(self._project(pivot + axis * self._drag_gizmo_length) - self._project(pivot)))
                    amount = pixel_delta * self._drag_gizmo_length / max(pixel_length, 1.0)
                else:
                    amount = 0.0
                if self.tool == "move":
                    position = np.asarray(transform["position"], dtype=float)
                    position[axis_index] += amount
                    node["transform"]["position"] = position.tolist()
                elif self.tool == "scale":
                    scale = np.asarray(transform["scale"], dtype=float)
                    factor = float(np.exp(np.clip(amount * 0.5, -4.0, 4.0)))
                    scale[axis_index] = float(np.clip(scale[axis_index] * factor, 0.01, 100.0))
                    node["transform"]["scale"] = scale.tolist()
            self.update()
            self.transformChanged.emit(node)
        elif not self._orbiting and not self._panning:
            hover_axis = self._pick_gizmo_axis(point)
            if hover_axis != self._hover_axis:
                self._hover_axis = hover_axis
                self.update()
        self._last_mouse = point
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._loop_cut is not None and self._operation_dragging:
            self._finish_operation("loop_cut")
            event.accept()
            return
        if self._bevel is not None and self._operation_dragging:
            self._finish_operation("bevel")
            event.accept()
            return
        if self._extrusion is not None and self._extrusion_dragging:
            self._finish_extrusion()
            event.accept()
            return
        if self._inset is not None and self._inset_dragging:
            self._finish_inset()
            event.accept()
            return
        if self._dragging_transform:
            geometry_edit = self.mode == "edit" and self._drag_geometry_before is not None
            if geometry_edit:
                self._record_geometry_change(self._drag_geometry_before)
            self._dragging_transform = False
            self._drag_transform = None
            self._drag_geometry = self._drag_geometry_before = None
            self._drag_pivot_world = None
            self._drag_axis = None
            self._drag_axis_start = None
            self._drag_rotation_start = None
            self._drag_axis_screen = None
            if not geometry_edit:
                self.sceneChanged.emit()
        self._orbiting = self._panning = False
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if (self._loop_cut is not None or self._bevel is not None) and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish_operation("loop_cut" if self._loop_cut is not None else "bevel")
            event.accept()
            return
        if (self._loop_cut is not None or self._bevel is not None) and event.key() == Qt.Key.Key_Escape:
            self._cancel_operation("loop_cut" if self._loop_cut is not None else "bevel")
            event.accept()
            return
        if self._inset is not None and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish_inset()
            event.accept()
            return
        if self._inset is not None and event.key() == Qt.Key.Key_Escape:
            self._cancel_inset()
            event.accept()
            return
        if self._extrusion is not None and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish_extrusion()
            event.accept()
            return
        if self._extrusion is not None and event.key() == Qt.Key.Key_Escape:
            self._cancel_extrusion()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F and self.mode == "object":
            self.focus_selected()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Tab:
            self.set_mode("edit" if self.mode == "object" else "object")
            event.accept()
            return
        if self.mode == "edit" and event.key() == Qt.Key.Key_A:
            self.select_all_components()
            event.accept()
            return
        if self.mode == "edit" and event.key() in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3):
            self.set_component_mode({Qt.Key.Key_1: "vertex", Qt.Key.Key_2: "edge", Qt.Key.Key_3: "face"}[event.key()])
            event.accept()
            return
        if self.mode == "edit" and event.key() == Qt.Key.Key_E:
            self.extrude_selected_face()
            event.accept()
            return
        if self.mode == "edit" and event.key() == Qt.Key.Key_I:
            self.inset_selected_face()
            event.accept()
            return
        if self.mode == "edit" and event.key() == Qt.Key.Key_L:
            self.loop_cut_selected_edge()
            event.accept()
            return
        if self.mode == "edit" and event.key() == Qt.Key.Key_B:
            self.bevel_selected_edges()
            event.accept()
            return
        if self.mode == "edit" and event.key() == Qt.Key.Key_F:
            self.fill_selected_boundary()
            event.accept()
            return
        if self.mode == "edit" and event.key() == Qt.Key.Key_M:
            self.merge_selected_vertices()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._dragging_transform and self._drag_transform is not None:
            node = self.selected_node
            if node is not None:
                node["transform"] = {key: list(value) for key, value in self._drag_transform.items()}
                self.transformChanged.emit(node)
            self._dragging_transform = False
            self._drag_transform = None
            self._drag_axis = None
            self._drag_axis_start = None
            self._drag_rotation_start = None
            self._drag_axis_screen = None
            self._hover_axis = None
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self.mode == "edit" and self._drag_geometry_before is not None:
            node = self.selected_node
            if node is not None:
                node["geometry"] = self._drag_geometry_before
            self._dragging_transform = False
            self._drag_geometry = self._drag_geometry_before = None
            self._drag_axis = None
            self._drag_pivot_world = None
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        steps = event.angleDelta().y() / 120.0
        self.camera.zoom(steps)
        self.update()
        event.accept()
