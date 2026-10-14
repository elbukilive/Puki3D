"""Orbit camera and gizmo math tests."""

import unittest

import numpy as np

from puki3d.camera import (
    OrbitCamera,
    closest_segment_fraction,
    project_to_screen,
    ray_axis_parameter,
    signed_angle_about_axis,
)


class CameraTests(unittest.TestCase):
    def test_orbit_updates_angles_and_clamps_elevation(self) -> None:
        camera = OrbitCamera()
        old_azimuth = camera.azimuth

        camera.orbit(20, 1000)

        self.assertNotEqual(camera.azimuth, old_azimuth)
        self.assertAlmostEqual(camera.elevation, np.radians(85))

    def test_pan_moves_target_in_camera_plane(self) -> None:
        camera = OrbitCamera()
        old_target = camera.target.copy()

        camera.pan(12, -5, 600)

        self.assertFalse(np.allclose(camera.target, old_target))
        self.assertEqual(camera.target.shape, (3,))

    def test_zoom_is_exponential_and_clamped(self) -> None:
        camera = OrbitCamera()
        camera.zoom(2)
        self.assertAlmostEqual(camera.distance, 8.0 * 0.86**2)
        camera.zoom(1000)
        self.assertEqual(camera.distance, 0.5)
        camera.zoom(-1000)
        self.assertEqual(camera.distance, 100.0)

    def test_camera_target_projects_to_viewport_center(self) -> None:
        camera = OrbitCamera()
        view, _, _, _ = camera.view_matrix()
        projection = camera.projection_matrix(16 / 9)

        screen = project_to_screen(camera.target, view, projection, 1600, 900)

        np.testing.assert_allclose(screen, [800, 450], atol=1e-8)

    def test_axis_drag_parameter_and_rotation_are_axis_specific(self) -> None:
        x_axis = np.array([1.0, 0.0, 0.0])
        pivot = np.zeros(3)
        start = ray_axis_parameter(np.array([2.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), pivot, x_axis)
        end = ray_axis_parameter(np.array([3.5, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), pivot, x_axis)

        self.assertAlmostEqual(end - start, 1.5)
        self.assertAlmostEqual(
            signed_angle_about_axis(np.array([0, 1, 0]), np.array([0, 0, 1]), x_axis),
            90.0,
        )

    def test_gizmo_line_hit_math_projects_to_nearest_segment_point(self) -> None:
        distance, fraction = closest_segment_fraction(
            np.array([5.0, 2.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])
        )

        self.assertAlmostEqual(distance, 2.0)
        self.assertAlmostEqual(fraction, 0.5)


if __name__ == "__main__":
    unittest.main()
