"""Workspace selection and inspector synchronization tests."""

import os
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from puki3d.project import Project
from puki3d.scene import _geometry, node_matrix
from puki3d.ui import MainWindow
from puki3d.viewport import _silhouette_edge_ids


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

    def test_object_selection_outline_uses_mesh_silhouette_not_triangulation(self) -> None:
        self.window.add_primitive("cube")
        geometry = self.window.viewport.selected_node["geometry"]
        edges = _silhouette_edge_ids(geometry, np.array([4.0, 4.0, 4.0]))
        self.assertEqual(len(edges), 6)
        self.assertTrue(all(edge_id < len(geometry["edges"]) for edge_id in edges))

    def test_edit_mode_extrudes_and_undo_redo_persists_mesh(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        viewport.set_mode("edit")
        viewport.selected_components = {1}
        original = copy.deepcopy(node["geometry"])
        screen = viewport._project([0.0, 0.0, 1.0])
        viewport._last_mouse = QPoint(round(screen[0]), round(screen[1]))
        viewport.extrude_selected_face()
        self.assertTrue(viewport.extruding)
        viewport._update_extrusion(viewport._last_mouse + QPoint(0, -70))
        self.assertGreater(viewport._extrusion["distance"], 0.0)
        viewport._finish_extrusion()
        self.assertFalse(viewport.extruding)

        self.assertEqual(len(node["geometry"]["vertices"]), 12)
        self.assertEqual(len(node["geometry"]["faces"]), 10)
        self.assertNotEqual(node["geometry"], original)
        viewport.undo_edit()
        self.assertEqual(node["geometry"], original)
        viewport.redo_edit()
        self.assertEqual(len(node["geometry"]["vertices"]), 12)

        self.window.save_project()
        reopened = Project.load(self.project.path)
        saved_node = reopened.data["scenes"][0]["nodes"][0]
        self.assertEqual(saved_node["geometry"], node["geometry"])

    def test_select_all_components_shortcut_tracks_active_mode(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        viewport.set_mode("edit")
        for mode, count in (("vertex", 8), ("edge", 12), ("face", 6)):
            viewport.set_component_mode(mode)
            QTest.keyClick(viewport, Qt.Key.Key_A)
            self.assertEqual(viewport.selected_components, set(range(count)))

    def test_escape_cancels_active_extrusion_without_changing_mesh(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        viewport.set_mode("edit")
        viewport.selected_components = {1}
        original = copy.deepcopy(node["geometry"])
        screen = viewport._project([0.0, 0.0, 1.0])
        viewport._last_mouse = QPoint(round(screen[0]), round(screen[1]))
        viewport.extrude_selected_face()
        viewport._update_extrusion(viewport._last_mouse + QPoint(0, -50))
        viewport._cancel_extrusion()
        self.assertFalse(viewport.extruding)
        self.assertEqual(node["geometry"], original)
        self.assertFalse(viewport._undo_stack)

    def test_inset_preview_commits_undoes_redoes_and_persists(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        viewport.set_mode("edit")
        viewport.set_component_mode("face")
        viewport.selected_components = {1}
        original = copy.deepcopy(node["geometry"])
        screen = viewport._project([0.0, 0.0, 1.0])
        viewport._last_mouse = QPoint(round(screen[0]), round(screen[1]))
        viewport.inset_selected_face()
        self.assertTrue(viewport.insetting)
        viewport._update_inset(viewport._last_mouse + QPoint(8, 0))
        self.assertGreater(viewport._inset["amount"], 0.0)
        viewport._finish_inset()
        self.assertFalse(viewport.insetting)
        self.assertEqual(len(node["geometry"]["faces"]), 10)
        viewport.undo_edit()
        self.assertEqual(node["geometry"], original)
        viewport.redo_edit()
        self.assertEqual(len(node["geometry"]["vertices"]), 12)

        self.window.save_project()
        reopened = Project.load(self.project.path)
        self.assertEqual(reopened.data["scenes"][0]["nodes"][0]["geometry"], node["geometry"])

    def test_loop_cut_preview_commits_undoes_redoes_and_persists(self) -> None:
        self.window.add_primitive("cube")
        viewport, node = self.window.viewport, self.window.viewport.selected_node
        viewport.set_mode("edit")
        viewport.set_component_mode("edge")
        viewport.selected_components = {0}
        original = copy.deepcopy(node["geometry"])
        viewport._last_mouse = QPoint(200, 200)
        viewport.loop_cut_selected_edge()
        self.assertIsNotNone(viewport._loop_cut)
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=QPoint(200, 200))
        QTest.mouseMove(viewport, QPoint(224, 200))
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=QPoint(224, 200))
        self.assertIsNone(viewport._loop_cut)
        self.assertEqual(len(node["geometry"]["faces"]), 10)
        viewport.undo_edit()
        self.assertEqual(node["geometry"], original)
        viewport.redo_edit()
        self.assertEqual(len(node["geometry"]["edges"]), 20)
        self.window.save_project()
        loaded = Project.load(self.project.path).data["scenes"][0]["nodes"][0]
        self.assertEqual(loaded["geometry"], node["geometry"])

    def test_bevel_preview_commits_undoes_and_redoes(self) -> None:
        self.window.add_primitive("cube")
        viewport, node = self.window.viewport, self.window.viewport.selected_node
        viewport.set_mode("edit")
        viewport.set_component_mode("edge")
        viewport.selected_components = {0}
        original = copy.deepcopy(node["geometry"])
        viewport._last_mouse = QPoint(200, 200)
        viewport.bevel_selected_edges()
        self.assertIsNotNone(viewport._bevel)
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=QPoint(200, 200))
        QTest.mouseMove(viewport, QPoint(202, 200))
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=QPoint(202, 200))
        self.assertIsNone(viewport._bevel)
        self.assertEqual(len(node["geometry"]["faces"]), 7)
        viewport.undo_edit()
        self.assertEqual(node["geometry"], original)
        viewport.redo_edit()
        self.assertEqual(len(node["geometry"]["faces"]), 7)

    def test_mirror_modifier_is_welded_toggleable_undoable_and_persistent(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        viewport.set_mode("edit")
        node["geometry"] = _geometry(
            [[0, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0, 0.5, -0.5],
             [0, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [0, 0.5, 0.5]],
            [[0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [0, 1, 5, 4], [3, 2, 6, 7], [4, 5, 6, 7]],
        )
        viewport.set_mirror_axis(0)
        self.assertEqual(node["mirror"], {"axis": 0, "merge": True, "clip": True})
        evaluated, maps = viewport._evaluated_geometry(node)
        self.assertEqual(len(evaluated["vertices"]), 12)
        self.assertEqual(len(evaluated["faces"]), 10)
        self.assertTrue(all(index >= 0 for index in maps["edges"]))
        viewport.undo_edit()
        self.assertNotIn("mirror", node)
        viewport.redo_edit()
        self.assertEqual(node["mirror"]["axis"], 0)
        self.window.save_project()
        reopened = Project.load(self.project.path)
        loaded = reopened.data["scenes"][0]["nodes"][0]
        self.assertEqual(loaded["mirror"], node["mirror"])
        self.assertEqual(loaded["geometry"], node["geometry"])

    def test_edit_face_picking_uses_the_face_under_the_cursor(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        viewport.set_mode("edit")
        viewport.set_component_mode("face")
        screen = viewport._project([0.0, 0.0, 1.0])
        picked = viewport._pick_component(QPoint(round(screen[0]), round(screen[1])))
        self.assertEqual(picked, 1)

    def test_mirrored_face_can_be_picked_and_selects_source_face(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        node["geometry"]["vertices"] = [
            [vertex[0] + 0.5, vertex[1], vertex[2]]
            for vertex in node["geometry"]["vertices"]
        ]
        viewport.set_mode("edit")
        viewport.set_mirror_axis(0)
        viewport.set_component_mode("face")
        face = node["geometry"]["faces"][1]
        center = np.mean(np.asarray(node["geometry"]["vertices"])[face], axis=0)
        world = (node_matrix(node) @ np.append(center, 1.0))[:3]
        screen = viewport._project(world)
        point = QPoint(round(screen[0]), round(screen[1]))

        self.assertEqual(viewport._pick_component(point), 1)
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=point)
        self.assertEqual(viewport.selected_components, {1})

    def test_vertex_edge_and_face_picking_use_visible_mesh_elements(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        viewport.set_mode("edit")

        viewport.set_component_mode("vertex")
        point = node_matrix(node) @ np.append(node["geometry"]["vertices"][4], 1.0)
        screen = viewport._project(point[:3])
        self.assertEqual(viewport._pick_component(QPoint(round(screen[0]), round(screen[1]))), 4)

        viewport.set_component_mode("edge")
        edges = viewport._component_edges(node["geometry"])
        edge_id = edges.index((4, 5))
        midpoint = (np.asarray(node["geometry"]["vertices"][4]) + np.asarray(node["geometry"]["vertices"][5])) / 2
        point = node_matrix(node) @ np.append(midpoint, 1.0)
        screen = viewport._project(point[:3])
        self.assertEqual(viewport._pick_component(QPoint(round(screen[0]), round(screen[1]))), edge_id)

        viewport.set_component_mode("face")
        screen = viewport._project([0.0, 0.0, 1.0])
        self.assertEqual(viewport._pick_component(QPoint(round(screen[0]), round(screen[1]))), 1)

    def test_component_move_and_scale_can_target_each_xyz_axis(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        viewport.set_mode("edit")
        viewport.set_component_mode("vertex")
        viewport.selected_components = set(range(8))

        for tool in ("move", "scale"):
            viewport.set_tool(tool)
            for axis_index in range(3):
                before = copy.deepcopy(node["geometry"])
                viewport._drag_geometry_before = before
                viewport._drag_geometry = copy.deepcopy(before)
                viewport._drag_pivot_world = viewport._component_pivot(node)
                viewport._drag_axis = axis_index
                viewport._drag_axis_start = None
                viewport._drag_start = QPoint(100, 100)
                viewport._apply_component_transform(QPoint(200, 100))
                moved_axis = [
                    index for index in range(3)
                    if node["geometry"]["vertices"][0][index] != before["vertices"][0][index]
                ]
                self.assertEqual(moved_axis, [axis_index])
                viewport._record_geometry_change(before)
                viewport.undo_edit()
                self.assertEqual(node["geometry"], before)

    def test_component_rotation_can_target_each_xyz_axis(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        viewport.set_mode("edit")
        viewport.set_component_mode("vertex")
        viewport.selected_components = set(range(8))
        viewport.set_tool("rotate")
        vectors = (
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        )

        for axis_index, (start_vector, end_vector) in enumerate(vectors):
            before = copy.deepcopy(node["geometry"])
            pivot = viewport._component_pivot(node)
            start_screen = viewport._project(pivot + start_vector)
            end_screen = viewport._project(pivot + end_vector)
            start_point = QPoint(round(start_screen[0]), round(start_screen[1]))
            end_point = QPoint(round(end_screen[0]), round(end_screen[1]))
            viewport._drag_geometry_before = before
            viewport._drag_geometry = copy.deepcopy(before)
            viewport._drag_pivot_world = pivot
            viewport._drag_axis = axis_index
            viewport._drag_rotation_start = viewport._ray_plane_vector(start_point, pivot, viewport._axis_vector(axis_index))

            viewport._apply_component_transform(end_point)
            self.assertAlmostEqual(
                node["geometry"]["vertices"][0][axis_index],
                before["vertices"][0][axis_index],
                places=6,
            )
            self.assertNotEqual(node["geometry"], before)
            viewport._record_geometry_change(before)
            viewport.undo_edit()
            self.assertEqual(node["geometry"], before)

    def test_edit_gizmo_moves_selected_vertex_and_undoes(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        viewport.set_mode("edit")
        viewport.set_component_mode("vertex")
        viewport.selected_components = {4}
        viewport.set_tool("move")
        before = viewport._snapshot_geometry()
        viewport._drag_geometry_before = before
        viewport._drag_geometry = copy.deepcopy(node["geometry"])
        viewport._drag_pivot_world = viewport._component_pivot(node)
        viewport._drag_axis = 0
        viewport._drag_axis_start = None
        viewport._drag_start = QPoint(100, 100)

        viewport._apply_component_transform(QPoint(200, 100))
        viewport._record_geometry_change(before)
        self.assertEqual(node["geometry"]["vertices"][4][0], 0.5)
        viewport.undo_edit()
        self.assertEqual(node["geometry"]["vertices"][4][0], -0.5)

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
