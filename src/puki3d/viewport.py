"""Interactive OpenGL scene view with orbit navigation, picking, and transforms."""

from __future__ import annotations

from math import cos, pi, radians, sin

import numpy as np
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
from .scene import node_matrix, raycast


class SceneViewport(QOpenGLWidget):
    selectionChanged = Signal(object)
    transformChanged = Signal(object)
    sceneChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.nodes: list[dict] = []
        self.selected_id: str | None = None
        self.tool = "select"
        self.camera = OrbitCamera()
        self._last_mouse = QPoint()
        self._drag_start = QPoint()
        self._drag_transform: dict | None = None
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
        self.selected_id = node_id
        self.update()
        self.selectionChanged.emit(self.selected_node)

    def set_tool(self, tool: str) -> None:
        if tool in {"select", "move", "rotate", "scale"}:
            self.tool = tool
            self._hover_axis = None
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
        geometry = node.get("geometry", {})
        vertices = geometry.get("vertices", [])
        triangles = geometry.get("triangles", [])
        if not vertices or not triangles:
            return
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glMultMatrixf(np.asarray(node_matrix(node).T, dtype=np.float32))
        glColor3f(0.82, 1.0, 0.31)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        for triangle in triangles:
            for start, end in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                for index in (start, end):
                    point = vertices[index]
                    glVertex3f(float(point[0]), float(point[1]), float(point[2]))
        glEnd()
        glPopMatrix()
        if self.tool in {"move", "rotate", "scale"}:
            origin = np.asarray(node["transform"]["position"], dtype=float)
            if self.tool == "rotate":
                self._draw_rotation_gizmo(origin)
            else:
                self._draw_axis_gizmo(origin, camera_right, camera_up)
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
        origin = np.asarray(node["transform"]["position"], dtype=float)
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
        return raycast(self.nodes, origin, direction)

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray]:
        view, _, right, camera_up = self.camera.view_matrix()
        return right, camera_up

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.setFocus()
        self._last_mouse = event.position().toPoint()
        self._drag_start = self._last_mouse
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
            if hit_id != self.selected_id:
                self.select_node(hit_id)
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        point = event.position().toPoint()
        delta = point - self._last_mouse
        if self._orbiting:
            self.camera.orbit(delta.x(), delta.y())
            self._hover_axis = self._pick_gizmo_axis(point)
            self.update()
        elif self._panning:
            self.camera.pan(delta.x(), delta.y(), self.height())
            self._hover_axis = self._pick_gizmo_axis(point)
            self.update()
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
        if self._dragging_transform:
            self._dragging_transform = False
            self._drag_transform = None
            self._drag_axis = None
            self._drag_axis_start = None
            self._drag_rotation_start = None
            self._drag_axis_screen = None
            self.sceneChanged.emit()
        self._orbiting = self._panning = False
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_F:
            self.focus_selected()
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
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        steps = event.angleDelta().y() / 120.0
        self.camera.zoom(steps)
        self.update()
        event.accept()
