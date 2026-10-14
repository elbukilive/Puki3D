"""Orbit camera math shared by the viewport and its tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians, sin, tan

import numpy as np


@dataclass
class OrbitCamera:
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.35]))
    distance: float = 8.0
    azimuth: float = radians(-48.0)
    elevation: float = radians(28.0)
    field_of_view: float = radians(48.0)

    def orbit(self, delta_x: float, delta_y: float, sensitivity: float = 0.008) -> None:
        self.azimuth -= delta_x * sensitivity
        self.elevation = float(np.clip(self.elevation + delta_y * sensitivity, radians(-85), radians(85)))

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        horizontal = cos(self.elevation)
        eye = self.target + self.distance * np.array([
            horizontal * cos(self.azimuth), horizontal * sin(self.azimuth), sin(self.elevation)
        ])
        forward = self.target - eye
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, forward)
        return eye, forward, right, camera_up

    def view_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eye, forward, right, camera_up = self.basis()
        view = np.eye(4)
        view[0, :3], view[0, 3] = right, -np.dot(right, eye)
        view[1, :3], view[1, 3] = camera_up, -np.dot(camera_up, eye)
        view[2, :3], view[2, 3] = -forward, np.dot(forward, eye)
        return view, eye, right, camera_up

    def projection_matrix(self, aspect: float, near: float = 0.05, far: float = 250.0) -> np.ndarray:
        f = 1.0 / tan(self.field_of_view / 2.0)
        return np.array([
            [f / max(aspect, 1e-9), 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ], dtype=float)

    def pan(self, delta_x: float, delta_y: float, viewport_height: int) -> None:
        _, _, right, camera_up = self.basis()
        units_per_pixel = self.distance * 2 * tan(self.field_of_view / 2) / max(viewport_height, 1)
        self.target += right * (-delta_x * units_per_pixel) + camera_up * (delta_y * units_per_pixel)

    def zoom(self, wheel_steps: float) -> None:
        self.distance = float(np.clip(self.distance * (0.86 ** wheel_steps), 0.5, 100.0))


def project_to_screen(point: np.ndarray, view: np.ndarray, projection: np.ndarray,
                      width: int, height: int) -> np.ndarray | None:
    clip = projection @ view @ np.append(np.asarray(point, dtype=float), 1.0)
    if clip[3] <= 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    return np.array([(ndc[0] + 1.0) * width * 0.5, (1.0 - ndc[1]) * height * 0.5])


def closest_segment_fraction(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= 1e-12:
        return float(np.linalg.norm(point - start)), 0.0
    fraction = float(np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0))
    closest = start + fraction * segment
    return float(np.linalg.norm(point - closest)), fraction


def ray_axis_parameter(origin: np.ndarray, direction: np.ndarray, pivot: np.ndarray,
                       axis: np.ndarray) -> float | None:
    """Closest point on an axis to a ray, measured from the pivot along the axis."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    offset = np.asarray(origin, dtype=float) - np.asarray(pivot, dtype=float)
    dot = float(np.dot(direction, axis))
    denominator = 1.0 - dot * dot
    if denominator <= 1e-5:
        return None
    ray_offset = float(np.dot(direction, offset))
    axis_offset = float(np.dot(axis, offset))
    ray_distance = (dot * axis_offset - ray_offset) / denominator
    return axis_offset + dot * ray_distance


def signed_angle_about_axis(start: np.ndarray, end: np.ndarray, axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    start = np.asarray(start, dtype=float) - axis * float(np.dot(start, axis))
    end = np.asarray(end, dtype=float) - axis * float(np.dot(end, axis))
    start_length, end_length = np.linalg.norm(start), np.linalg.norm(end)
    if start_length <= 1e-9 or end_length <= 1e-9:
        return 0.0
    start, end = start / start_length, end / end_length
    return float(np.degrees(np.arctan2(np.dot(axis, np.cross(start, end)), np.dot(start, end))))
