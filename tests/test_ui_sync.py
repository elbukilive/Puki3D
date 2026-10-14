"""Workspace selection and inspector synchronization tests."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from puki3d.project import Project
from puki3d.ui import MainWindow


class WorkspaceSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project = Project.create(Path(self.temp_dir.name), "UI Sync", "object")
        self.window = MainWindow()
        self.window.show_project(self.project)

    def tearDown(self) -> None:
        self.window.autosave_timer.stop()
        self.window.project = None
        self.window.close()
        self.temp_dir.cleanup()

    def test_outliner_and_inspector_follow_selection_and_edits(self) -> None:
        self.window.add_primitive("cube")
        node = self.project.data["scenes"][0]["nodes"][0]

        self.assertEqual(self.window.outliner.count(), 1)
        self.assertEqual(self.window.viewport.selected_id, node["id"])
        self.assertTrue(self.window.transform_panel.isEnabled())

        self.window.outliner.setCurrentRow(0)
        self.assertEqual(self.window.viewport.selected_id, node["id"])
        self.window.transform_fields["position"][0].setValue(3.25)
        self.assertEqual(node["transform"]["position"][0], 3.25)

        self.window.viewport.select_node(None)
        self.assertEqual(self.window.outliner.currentRow(), -1)
        self.assertFalse(self.window.transform_panel.isEnabled())

    def test_axis_drag_is_independent_starts_without_jump_and_can_be_cancelled(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        viewport.set_tool("move")
        self.app.processEvents()
        node = viewport.selected_node
        for tool in ("move", "scale"):
            viewport.set_tool(tool)
            for axis_index in range(3):
                original = {key: list(value) for key, value in node["transform"].items()}
                pivot = original["position"]
                endpoint = list(pivot)
                endpoint[axis_index] += viewport._gizmo_length
                origin_screen, end_screen = viewport._project(pivot), viewport._project(endpoint)
                vector = end_screen - origin_screen
                handle = QPoint(round(origin_screen[0] + vector[0] * 0.72),
                                round(origin_screen[1] + vector[1] * 0.72))
                self.assertEqual(viewport._pick_gizmo_axis(handle), axis_index)

                QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=handle)
                self.assertEqual(node["transform"], original)
                delta = QPoint(round(vector[0] / max((vector @ vector) ** 0.5, 1) * 28),
                               round(vector[1] / max((vector @ vector) ** 0.5, 1) * 28))
                QTest.mouseMove(viewport, handle + delta)
                self.app.processEvents()
                changed_key = "position" if tool == "move" else "scale"
                for key in ("position", "rotation", "scale"):
                    for index, value in enumerate(node["transform"][key]):
                        if key == changed_key and index == axis_index:
                            self.assertNotEqual(value, original[key][index])
                        else:
                            self.assertEqual(value, original[key][index])
                self.assertEqual(viewport.selected_id, node["id"])
                self.assertEqual(self.window.outliner.currentRow(), 0)
                self.assertAlmostEqual(
                    self.window.transform_fields[changed_key][axis_index].value(),
                    node["transform"][changed_key][axis_index],
                    places=3,
                )

                QTest.keyClick(viewport, Qt.Key.Key_Escape)
                self.app.processEvents()
                self.assertEqual(node["transform"], original)
                QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=handle + delta)

    def test_camera_controls_and_focus_selection(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        self.app.processEvents()
        node = viewport.selected_node
        before_orbit = (viewport.camera.azimuth, viewport.camera.elevation)
        start = QPoint(120, 120)
        QTest.mousePress(viewport, Qt.MouseButton.MiddleButton, pos=start)
        QTest.mouseMove(viewport, start + QPoint(30, 15))
        QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton, pos=start + QPoint(30, 15))
        self.app.processEvents()
        self.assertNotEqual((viewport.camera.azimuth, viewport.camera.elevation), before_orbit)

        viewport.set_tool("move")
        pivot = node["transform"]["position"]
        for axis_index in range(3):
            endpoint = list(pivot)
            endpoint[axis_index] += viewport._gizmo_length
            origin_screen, end_screen = viewport._project(pivot), viewport._project(endpoint)
            handle = QPoint(round(origin_screen[0] * 0.28 + end_screen[0] * 0.72),
                            round(origin_screen[1] * 0.28 + end_screen[1] * 0.72))
            self.assertEqual(viewport._pick_gizmo_axis(handle), axis_index)

        before_pan = viewport.camera.target.copy()
        QTest.mousePress(viewport, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.ShiftModifier, pos=start)
        QTest.mouseMove(viewport, start + QPoint(22, -12))
        QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.ShiftModifier, pos=start + QPoint(22, -12))
        self.app.processEvents()
        self.assertFalse((viewport.camera.target == before_pan).all())

        viewport.camera.target = [40.0, -30.0, 10.0]
        viewport.setFocus()
        self.app.processEvents()
        QTest.keyClick(viewport, Qt.Key.Key_F)
        self.app.processEvents()
        self.assertEqual(list(viewport.camera.target), node["transform"]["position"])

    def test_rotation_gizmo_changes_only_the_selected_axis(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        viewport.set_tool("rotate")
        self.app.processEvents()
        node = viewport.selected_node
        pivot = node["transform"]["position"]

        for axis_index in range(3):
            original = {key: list(value) for key, value in node["transform"].items()}
            ring = viewport._rotation_ring_points(pivot, axis_index)
            start_point = viewport._project(ring[5])
            end_point = viewport._project(ring[15])
            start = QPoint(round(start_point[0]), round(start_point[1]))
            end = QPoint(round(end_point[0]), round(end_point[1]))
            self.assertEqual(viewport._pick_gizmo_axis(start), axis_index)

            QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
            self.assertEqual(node["transform"], original)
            QTest.mouseMove(viewport, end)
            self.app.processEvents()
            for index, value in enumerate(node["transform"]["rotation"]):
                if index == axis_index:
                    self.assertNotEqual(value, original["rotation"][index])
                else:
                    self.assertEqual(value, original["rotation"][index])
            QTest.keyClick(viewport, Qt.Key.Key_Escape)
            self.app.processEvents()
            self.assertEqual(node["transform"], original)
            QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)


if __name__ == "__main__":
    unittest.main()
