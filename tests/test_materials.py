"""Material library, texture, UV and assignment persistence tests."""

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from puki3d.materials import MaterialEditorDialog, ensure_material_library, import_texture
from puki3d.project import Project
from puki3d.scene import create_node, ensure_editable_geometry, primitive_geometry
from puki3d.ui import MainWindow


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
