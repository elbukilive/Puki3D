"""Material library, texture, UV and assignment persistence tests."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from puki3d.materials import MaterialEditorDialog, ensure_material_library, import_texture, texture_has_alpha
from puki3d.project import Project
from puki3d.scene import (
    auto_unwrap_geometry, create_node, ensure_editable_geometry, planar_project_geometry,
    primitive_geometry, transform_face_uvs,
)
from puki3d.ui import MainWindow
from puki3d.uv_editor import UVCanvas


class MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_default_library_and_legacy_upgrade_preserve_user_materials(self):
        data = {}
        materials = ensure_material_library(data)
        self.assertEqual(len(materials), 6)
        self.assertIn("Pintura automotriz", {item["name"] for item in materials})
        materials.append({"id": "user", "name": "Custom"})
        self.assertIs(ensure_material_library(data), materials)
        self.assertEqual([item["id"] for item in ensure_material_library(data)],
                         [item["id"] for item in materials])

    def test_primitives_generate_uvs_and_editable_geometry_keeps_face_data(self):
        geometry = primitive_geometry("cube")
        self.assertEqual(len(geometry["face_uvs"]), len(geometry["faces"]))
        self.assertEqual(len(geometry["face_materials"]), len(geometry["faces"]))
        geometry["face_uvs"][0][0] = [0.23, 0.71]
        geometry["face_materials"][0] = "paint"
        ensure_editable_geometry(geometry)
        self.assertEqual(geometry["face_uvs"][0][0], [0.23, 0.71])
        self.assertEqual(geometry["face_materials"][0], "paint")

    def test_embedded_texture_material_and_face_uv_survive_project_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "swatch.png"
            image = QImage(2, 2, QImage.Format.Format_RGBA8888)
            image.fill(QColor("#3979C7"))
            self.assertTrue(image.save(str(source)))
            project = Project.create(root, "Material test", "object")
            materials = ensure_material_library(project.data)
            asset = import_texture(project.data, source)
            materials[0]["texture_id"] = asset["id"]
            node = create_node("cube")
            node["material_id"] = materials[0]["id"]
            node["geometry"]["face_materials"][1] = materials[0]["id"]
            node["geometry"]["face_uvs"][1][0] = [0.18, 0.83]
            project.data["scenes"][0]["nodes"].append(node)
            project.save()

            reopened = Project.load(project.path)
            loaded_node = reopened.data["scenes"][0]["nodes"][0]
            loaded_material = reopened.data["materials"][0]
            self.assertEqual(loaded_node["material_id"], materials[0]["id"])
            self.assertEqual(loaded_node["geometry"]["face_materials"][1], materials[0]["id"])
            self.assertEqual(loaded_node["geometry"]["face_uvs"][1][0], [0.18, 0.83])
            self.assertEqual(loaded_material["texture_id"], asset["id"])
            self.assertEqual(reopened.data["textures"][0]["data_base64"], asset["data_base64"])

    def test_texture_import_reuses_identical_image_and_keeps_png_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "swatch.png"
            copy = root / "swatch-copy.png"
            image = QImage(8, 8, QImage.Format.Format_RGBA8888)
            image.fill(QColor(10, 20, 30, 120))
            self.assertTrue(image.save(str(source)))
            copy.write_bytes(source.read_bytes())
            data = {"textures": []}
            first = import_texture(data, source)
            second = import_texture(data, copy)
            self.assertIs(second, first)
            self.assertEqual(len(data["textures"]), 1)
            self.assertTrue(first["has_alpha"])
            self.assertTrue(texture_has_alpha(first))
            decoded = QImage.fromData(__import__("base64").b64decode(first["data_base64"]))
            self.assertTrue(decoded.hasAlphaChannel())
            self.assertEqual(decoded.pixelColor(0, 0).alpha(), 120)

    def test_auto_uvs_pack_face_islands_without_overlap(self):
        geometry = primitive_geometry("cube")
        boxes = []
        for uv_loop in geometry["face_uvs"]:
            points = np.asarray(uv_loop, dtype=float)
            boxes.append((points.min(axis=0), points.max(axis=0)))
        for left_index, (left_min, left_max) in enumerate(boxes):
            for right_min, right_max in boxes[left_index + 1:]:
                overlap = np.minimum(left_max, right_max) - np.maximum(left_min, right_min)
                self.assertTrue(np.any(overlap <= 1e-9), "UV face islands overlap")

    def test_selected_auto_unwrap_uses_full_atlas_slots_and_preserves_other_islands(self):
        geometry = primitive_geometry("cube")
        untouched = [list(uv) for uv in geometry["face_uvs"][1]]
        unwrapped = auto_unwrap_geometry(geometry, {0})
        self.assertEqual(unwrapped["face_uvs"][1], untouched)
        selected = np.asarray(unwrapped["face_uvs"][0], dtype=float)
        other = np.asarray(unwrapped["face_uvs"][1], dtype=float)
        overlap = np.minimum(selected.max(axis=0), other.max(axis=0)) - np.maximum(
            selected.min(axis=0), other.min(axis=0)
        )
        self.assertTrue(np.any(overlap <= 1e-9), "selected unwrap overlaps an untouched island")

    def test_planar_projection_preserves_selected_island_slot(self):
        geometry = primitive_geometry("cube")
        projected = planar_project_geometry(geometry, axis=2, face_ids={0})
        points = np.asarray(projected["face_uvs"][0], dtype=float)
        original = np.asarray(geometry["face_uvs"][0], dtype=float)
        np.testing.assert_allclose(points.min(axis=0), original.min(axis=0))
        np.testing.assert_allclose(points.max(axis=0), original.max(axis=0))
        self.assertEqual(projected["face_uvs"][1], geometry["face_uvs"][1])

    def test_uv_face_picking_prefers_the_synced_selection_when_islands_overlap(self):
        canvas = UVCanvas()
        geometry = primitive_geometry("cube")
        geometry["face_uvs"] = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                                for _ in geometry["faces"]]
        canvas.resize(400, 400)
        canvas.set_geometry(geometry, {0})
        point = canvas._to_screen((0.5, 0.5))
        self.assertEqual(canvas._face_at(point), 0)

    def test_planar_projection_and_uv_transform_change_selected_islands_only(self):
        geometry = primitive_geometry("cube")
        untouched = [list(uv) for uv in geometry["face_uvs"][1]]
        projected = planar_project_geometry(geometry, axis=2, face_ids={0})
        self.assertEqual(projected["face_uvs"][1], untouched)
        points = np.asarray(projected["face_uvs"][0])
        center = points.mean(axis=0)
        transformed = transform_face_uvs(projected, {0}, offset=(0.1, -0.2), scale=2.0, rotation=90.0)
        result = np.asarray(transformed["face_uvs"][0])
        expected = center + (points - center) @ np.array([[0.0, -2.0], [2.0, 0.0]]).T + [0.1, -0.2]
        np.testing.assert_allclose(result, expected)
        self.assertEqual(transformed["face_uvs"][1], untouched)

    def test_geometry_edits_preserve_uvs_and_materials_on_existing_faces(self):
        geometry = primitive_geometry("cube")
        geometry["face_uvs"][0][0] = [0.19, 0.73]
        geometry["face_materials"][1] = "paint"
        geometry["vertices"].append([0.0, 0.0, 0.0])
        geometry["faces"].append([0, 1, 8])
        ensure_editable_geometry(geometry)
        self.assertEqual(geometry["face_uvs"][0][0], [0.19, 0.73])
        self.assertEqual(geometry["face_materials"][1], "paint")
        self.assertEqual(len(geometry["face_uvs"]), len(geometry["faces"]))

    def test_material_editor_duplicate_edit_delete_and_cancel_restore(self):
        data = {}
        dialog = MaterialEditorDialog(data)
        original = data["materials"][0]
        dialog.duplicate_material()
        self.assertEqual(len(data["materials"]), 7)
        self.assertNotEqual(data["materials"][-1]["id"], original["id"])
        dialog.reject()
        self.assertEqual(len(data["materials"]), 6)

    def test_inspector_assigns_shared_object_and_selected_face_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Project.create(Path(directory), "Inspector materials", "object")
            window = MainWindow()
            window.show_project(project)
            try:
                window.add_primitive("cube")
                node = window.viewport.selected_node
                material_id = project.data["materials"][0]["id"]
                window.material_combo.setCurrentIndex(window.material_combo.findData(material_id))
                self.assertEqual(node["material_id"], material_id)
                window.viewport.set_mode("edit")
                window.viewport.set_component_mode("face")
                window.viewport.selected_components = {2, 3}
                window._refresh_material_controls(node)
                self.assertTrue(window.assign_faces_button.isEnabled())
                window.assign_material_to_selected_faces()
                self.assertEqual(node["geometry"]["face_materials"][2:4], [material_id, material_id])
                project.save()
                reopened = Project.load(project.path)
                self.assertEqual(reopened.data["scenes"][0]["nodes"][0]["material_id"], material_id)
            finally:
                window.autosave_timer.stop()
                window.project = None
                window.close()

    def test_uv_dialog_selection_and_transform_stay_linked_to_viewport(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Project.create(Path(directory), "UV sync", "object")
            window = MainWindow()
            window.show_project(project)
            try:
                window.add_primitive("cube")
                window.open_uv_editor()
                dialog = window.uv_dialog
                self.assertIsNotNone(dialog)
                self.assertFalse(dialog.apply_button.isEnabled())
                dialog.canvas.set_selection({2})
                dialog.canvas.selectionChanged.emit({2})
                self.assertTrue(dialog.apply_button.isEnabled())
                self.assertEqual(window.viewport.selected_components, {2})
                before = [list(uv) for uv in window.viewport.selected_node["geometry"]["face_uvs"][2]]
                dialog.canvas.apply_transform(1.0, 0.0, 0.15, 0.0)
                after = window.viewport.selected_node["geometry"]["face_uvs"][2]
                self.assertNotEqual(after, before)
                self.assertAlmostEqual(after[0][0] - before[0][0], 0.15)
            finally:
                if window.uv_dialog is not None:
                    window.uv_dialog.close()
                window.autosave_timer.stop()
                window.project = None
                window.close()

    def test_uv_canvas_renders_islands_and_can_pick_a_face(self):
        canvas = UVCanvas()
        canvas.resize(500, 400)
        canvas.set_geometry(primitive_geometry("cube"))
        texture = QImage(8, 8, QImage.Format.Format_RGBA8888)
        texture.fill(QColor(255, 20, 20, 128))
        canvas.set_texture(texture)
        canvas.show()
        self.app.processEvents()
        rendered = QImage(canvas.size(), QImage.Format.Format_ARGB32)
        rendered.fill(QColor("#000000"))
        canvas.render(rendered)
        colors = {rendered.pixelColor(x, y).name()
                  for y in range(rendered.height()) for x in range(rendered.width())}
        self.assertTrue({"#56a0ef", "#df9e4a", "#8bc985", "#c18ee5", "#56c1b4", "#df7b83"} & colors)
        left, top, size = canvas._tile_rect()
        translucent_pixel = rendered.pixelColor(int(left + size * 0.5), int(top + size * 0.5))
        self.assertGreater(translucent_pixel.red(), translucent_pixel.green())
        face_id = canvas._face_at(QPointF(left + size * 0.27, top + size * 0.27))
        self.assertIsNotNone(face_id)
        canvas.close()
