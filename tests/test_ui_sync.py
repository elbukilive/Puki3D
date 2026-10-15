"""Workspace selection and inspector synchronization tests."""

import os
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton

from puki3d.project import Project
from puki3d.scene import _geometry, node_matrix, primitive_geometry, cylindrical_cutter_geometry
from puki3d.ui import MainWindow
from puki3d.viewport import _silhouette_edge_ids, _snap_axis_amount


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

    def test_object_viewport_color_uses_inspector_and_persists(self) -> None:
        self.window.add_primitive("cube")
        node = self.window.viewport.selected_node
        self.assertTrue(self.window.display_color_button.isEnabled())
        with patch("puki3d.ui.QColorDialog.getColor", return_value=QColor("#123456")):
            self.window.choose_display_color()
        self.assertEqual(node["display_color"], "#123456")
        self.project.save()
        reopened = Project.load(self.project.path)
        self.assertEqual(reopened.data["scenes"][0]["nodes"][0]["display_color"], "#123456")

    def test_modeling_tools_are_grouped_and_enabled_for_the_active_component_mode(self) -> None:
        grouped_tools = {button.text() for button in self.window.findChildren(QToolButton)}
        self.assertIn("Modelado", grouped_tools)
        self.assertIn("Topología", grouped_tools)
        self.assertIn("Vista", grouped_tools)

        self.window.add_primitive("cube")
        viewport = self.window.viewport
        viewport.set_mode("edit")
        viewport.set_component_mode("vertex")
        self.assertTrue(self.window.fill_action.isEnabled())
        self.assertTrue(self.window.bridge_action.isEnabled())
        self.assertTrue(self.window.merge_action.isEnabled())

        viewport.set_component_mode("face")
        self.assertTrue(self.window.inset_action.isEnabled())
        self.assertFalse(self.window.fill_action.isEnabled())

        viewport.set_component_mode("edge")
        self.assertTrue(self.window.loop_cut_action.isEnabled())
        self.assertTrue(self.window.bevel_action.isEnabled())

    def test_axis_snapping_targets_grid_and_nearby_vertices(self) -> None:
        self.assertAlmostEqual(_snap_axis_amount(0.12, 0.18, []), 0.13)
        self.assertAlmostEqual(_snap_axis_amount(0.12, 0.22, [0.34]), 0.22)

    def test_object_snapping_targets_grid_and_vertices_from_other_objects(self) -> None:
        self.window.add_primitive("cube")
        first = self.window.viewport.selected_node
        self.window.add_primitive("cube")
        second = self.window.viewport.selected_node
        second["transform"]["position"][0] = 1.1
        self.window.viewport.snapping = True

        self.assertAlmostEqual(self.window.viewport._snap_object_axis_amount(first, 0, 0.62), 0.6)

    def test_new_primitives_and_duplicates_keep_a_predictable_alignment(self) -> None:
        self.window.add_primitive("cube")
        first = self.window.viewport.selected_node
        first["transform"]["position"] = [2.0, -0.5, 1.25]

        self.window.add_primitive("cube")
        second = self.window.viewport.selected_node
        self.assertEqual(second["transform"]["position"], [2.0, -0.5, 1.25])

        self.window.duplicate_selected()
        duplicate = self.window.viewport.selected_node
        self.assertEqual(duplicate["transform"], second["transform"])
        self.assertEqual(duplicate["geometry"], second["geometry"])
        self.assertIsNot(duplicate["geometry"], second["geometry"])

    def test_selected_object_can_be_renamed_for_outliner_organization(self) -> None:
        self.window.add_primitive("cube")
        node = self.window.viewport.selected_node
        with patch("puki3d.ui.QInputDialog.getText", return_value=("Body · Chasis", True)):
            self.window.rename_selected_object()

        self.assertEqual(node["name"], "Body · Chasis")
        self.assertIn("Body · Chasis", self.window.outliner.currentItem().text())
        self.assertEqual(self.window.object_hint.text(), "Body · Chasis")

    def test_object_snap_move_and_camera_views_are_available(self) -> None:
        self.window.add_primitive("cube")
        viewport = self.window.viewport
        node = viewport.selected_node
        viewport.set_mode("edit")
        viewport.set_component_mode("vertex")
        viewport.selected_components = {4}
        viewport.set_tool("move")
        viewport.snapping = True
        before = copy.deepcopy(node["geometry"])
        viewport._drag_geometry_before = before
        viewport._drag_geometry = copy.deepcopy(before)
        viewport._drag_pivot_world = viewport._component_pivot(node)
        viewport._drag_axis = 0
        viewport._drag_axis_start = None
        viewport._drag_start = QPoint(100, 100)

        viewport._apply_component_transform(QPoint(200, 100))
        self.assertAlmostEqual(node["geometry"]["vertices"][4][0], 0.5)
        viewport._record_geometry_change(before)
        viewport.undo_edit()
        self.assertEqual(node["geometry"], before)

        viewport.set_camera_view("side")
        self.assertAlmostEqual(viewport.camera.azimuth, np.radians(90.0))
        viewport.set_camera_view("three_quarter")
        self.assertAlmostEqual(viewport.camera.azimuth, np.radians(-48.0))

    def test_topology_diagnostics_refresh_after_geometry_changes_and_undo(self) -> None:
        self.window.add_primitive("cube")
        viewport, node = self.window.viewport, self.window.viewport.selected_node
        viewport.set_mode("edit")
        before = copy.deepcopy(node["geometry"])
        node["geometry"] = _geometry(before["vertices"], before["faces"][:-1])
        viewport._record_geometry_change(before)
        self.assertIn("aristas abiertas", self.window.edit_status.text())
        viewport.undo_edit()
        self.assertIn("Malla cerrada", self.window.edit_status.text())

    def test_fill_shortcut_closes_selected_boundary_in_edit_mode(self) -> None:
        self.window.add_primitive("cube")
        viewport, node = self.window.viewport, self.window.viewport.selected_node
        geometry = node["geometry"]
        node["geometry"] = _geometry(geometry["vertices"], geometry["faces"][:-1])
        viewport.set_mode("edit")
        viewport.set_component_mode("vertex")
        viewport.selected_components = set(range(len(geometry["vertices"])))

        QTest.keyClick(viewport, Qt.Key.Key_F)

        self.assertEqual(len(node["geometry"]["faces"]), 6)
        self.assertIn("Malla cerrada", self.window.edit_status.text())

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

    def test_fill_operation_undoes_redoes_and_persists_with_mirror_enabled(self) -> None:
        self.window.add_primitive("cube")
        viewport, node = self.window.viewport, self.window.viewport.selected_node
        viewport.set_mode("edit")
        node["geometry"] = _geometry(
            [[0, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0, 0.5, -0.5],
             [0, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [0, 0.5, 0.5]],
            [[0, 4, 7, 3], [1, 2, 6, 5], [0, 3, 2, 1], [0, 1, 5, 4], [3, 7, 6, 2]],
        )
        viewport.set_component_mode("edge")
        viewport.selected_components = {index for index, adjacent in enumerate(node["geometry"]["edge_faces"]) if len(adjacent) == 1}
        viewport.set_mirror_axis(0)
        before_fill = copy.deepcopy(node["geometry"])
        viewport.fill_selected_boundary()
        self.assertEqual(len(node["geometry"]["faces"]), 6)
        self.assertTrue(all(len(adjacent) == 2 for adjacent in node["geometry"]["edge_faces"]))
        viewport.undo_edit()
        self.assertEqual(node["geometry"], before_fill)
        viewport.redo_edit()
        self.assertEqual(len(node["geometry"]["faces"]), 6)
        self.window.save_project()
        loaded = Project.load(self.project.path).data["scenes"][0]["nodes"][0]
        self.assertEqual(loaded["geometry"], node["geometry"])
        self.assertEqual(loaded["mirror"], node["mirror"])

    def test_bridge_operation_undoes_redoes_and_persists(self) -> None:
        self.window.add_primitive("cube")
        viewport, node = self.window.viewport, self.window.viewport.selected_node
        first = node["geometry"]
        second = copy.deepcopy(first)
        node["geometry"] = _geometry(
            first["vertices"] + [[x + 2, y, z] for x, y, z in second["vertices"]],
            first["faces"][:-1] + [[index + 8 for index in face] for face in second["faces"][:-1]],
        )
        viewport.set_mode("edit")
        viewport.set_component_mode("edge")
        viewport.selected_components = {index for index, adjacent in enumerate(node["geometry"]["edge_faces"]) if len(adjacent) == 1}
        before = copy.deepcopy(node["geometry"])
        viewport.bridge_selected_boundaries()
        self.assertTrue(all(len(adjacent) == 2 for adjacent in node["geometry"]["edge_faces"]))
        viewport.undo_edit()
        self.assertEqual(node["geometry"], before)
        viewport.redo_edit()
        self.assertTrue(all(len(adjacent) == 2 for adjacent in node["geometry"]["edge_faces"]))
        self.window.save_project()
        loaded = Project.load(self.project.path).data["scenes"][0]["nodes"][0]
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

    def test_boolean_cut_is_undoable_and_redoable_from_object_mode(self) -> None:
        self.window.add_primitive("cube")
        node = self.window.viewport.selected_node
        before = copy.deepcopy(node["geometry"])
        cutter = cylindrical_cutter_geometry([0, 0, 0], 0.25, axis=1, depth=2.0, segments=12)

        self.window.viewport.apply_boolean_difference(cutter)
        after = copy.deepcopy(node["geometry"])
        self.assertNotEqual(after, before)
        self.window.viewport.undo_edit()
        self.assertEqual(node["geometry"], before)
        self.window.viewport.redo_edit()
        self.assertEqual(node["geometry"], after)
        self.project.save()
        reopened = Project.load(self.project.path)
        self.assertEqual(reopened.data["scenes"][0]["nodes"][0]["geometry"], after)
        self.assertTrue(all(len(adjacent) == 2 for adjacent in reopened.data["scenes"][0]["nodes"][0]["geometry"]["edge_faces"]))

    def test_boolean_difference_menu_uses_other_object_transform(self) -> None:
        self.window.add_primitive("cube")
        target = self.window.viewport.selected_node
        self.window.add_primitive("cylinder")
        cutter = self.window.viewport.selected_node
        cutter["name"] = "Cortadora"
        cutter["transform"].update({"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [0.25, 0.25, 2.0]})
        self.window.viewport.select_node(target["id"])
        original = copy.deepcopy(target["geometry"])

        with patch("puki3d.ui.QInputDialog.getItem", return_value=("Cortadora", True)):
            self.window.boolean_difference_selected()

        self.assertNotEqual(target["geometry"], original)
        self.assertTrue(all(len(adjacent) == 2 for adjacent in target["geometry"]["edge_faces"]))

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
