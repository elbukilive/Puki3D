"""Persistence tests for Puki3D project documents."""

import json
from pathlib import Path
import tempfile
import unittest

from puki3d.project import Project, ProjectError


class ProjectPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_create_and_open_project(self) -> None:
        project = Project.create(self.root, "My First Pickup", "vehicle")

        self.assertTrue(project.path.is_file())
        self.assertEqual(project.path.name, "my-first-pickup.puki3d")
        self.assertEqual(project.data["format"], "puki3d.project")
        self.assertEqual(project.data["schema_version"], 1)

        reopened = Project.load(project.path)
        self.assertEqual(reopened.project_id, project.project_id)
        self.assertEqual(reopened.name, "My First Pickup")
        self.assertEqual(reopened.kind, "vehicle")
        self.assertEqual(reopened.data["scenes"][0]["name"], "Escena principal")
        self.assertEqual(reopened.data["scenes"][0]["nodes"], [])

    def test_save_and_open_preserve_editable_scene_data(self) -> None:
        project = Project.create(self.root, "Pickup", "vehicle")
        scene = project.data["scenes"][0]
        scene["nodes"].append(
            {
                "id": "node-1",
                "name": "Cabina",
                "type": "mesh",
                "primitive": "cube",
                "geometry": {"vertices": [[-1, -1, -1], [1, 1, 1]]},
                "transform": {
                    "position": [1.25, 2.0, -3.5],
                    "rotation": [0.0, 45.0, 0.0],
                    "scale": [2.0, 1.0, 0.5],
                },
            }
        )
        scene_data = json.loads(json.dumps(scene))

        project.save()
        reopened = Project.load(project.path)

        self.assertEqual(reopened.data["scenes"][0], scene_data)
        self.assertEqual(reopened.data["scenes"][0]["nodes"][0]["geometry"]["vertices"], [[-1, -1, -1], [1, 1, 1]])
        self.assertEqual(reopened.data["scenes"][0]["nodes"][0]["transform"]["rotation"], [0.0, 45.0, 0.0])

    def test_rename_preserves_data_and_moves_project_file(self) -> None:
        project = Project.create(self.root, "Pickup", "vehicle")
        project.data["scenes"][0]["nodes"].append(
            {"id": "node-1", "name": "Caja", "primitive": "cube", "transform": {"position": [1, 2, 3]}}
        )
        project.save()
        old_path = project.path
        expected_scenes = json.loads(json.dumps(project.data["scenes"]))

        project.rename("Pickup de prueba")

        self.assertFalse(old_path.exists())
        self.assertTrue(project.path.is_file())
        self.assertEqual(project.path.name, "pickup-de-prueba.puki3d")
        reopened = Project.load(project.path)
        self.assertEqual(reopened.name, "Pickup de prueba")
        self.assertEqual(reopened.data["scenes"], expected_scenes)

    def test_failed_rename_keeps_original_project(self) -> None:
        project = Project.create(self.root, "Pickup", "vehicle")
        conflicting = Project.create(self.root, "Ocupado", "object")
        original_path = project.path
        original_data = json.loads(json.dumps(project.data))

        with self.assertRaises(ProjectError):
            project.rename("Ocupado")

        self.assertEqual(project.path, original_path)
        self.assertTrue(original_path.is_file())
        self.assertEqual(project.data, original_data)
        self.assertTrue(conflicting.path.is_file())


if __name__ == "__main__":
    unittest.main()
