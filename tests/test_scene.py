"""Scene construction, picking, transform, and persistence tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from puki3d.project import Project
from puki3d.scene import PRIMITIVES, _geometry, create_node, ensure_editable_geometry, node_matrix, primitive_geometry, raycast


class SceneTests(unittest.TestCase):
    def test_supported_primitives_have_valid_triangulated_geometry(self) -> None:
        for primitive in PRIMITIVES:
            with self.subTest(primitive=primitive):
                geometry = primitive_geometry(primitive)
                vertices = geometry["vertices"]
                triangles = geometry["triangles"]
                self.assertGreater(len(vertices), 3)
                self.assertGreater(len(triangles), 1)
                self.assertTrue(all(len(triangle) == 3 for triangle in triangles))
                self.assertTrue(all(0 <= index < len(vertices) for triangle in triangles for index in triangle))

    def test_node_matrix_applies_scale_rotation_and_translation(self) -> None:
        node = create_node("cube")
        node["transform"] = {
            "position": [2, 3, 4],
            "rotation": [0, 0, 90],
            "scale": [2, 1, 1],
        }

        transformed = node_matrix(node) @ np.array([1.0, 0.0, 0.0, 1.0])

        np.testing.assert_allclose(transformed[:3], [2.0, 5.0, 4.0], atol=1e-8)

    def test_raycast_selects_nearest_transformed_object(self) -> None:
        near_node = create_node("cube", 1)
        near_node["transform"]["position"] = [0.0, 0.0, 0.5]
        far_node = create_node("cube", 2)
        far_node["transform"]["position"] = [0.0, 0.0, -2.0]

        selected = raycast(
            [far_node, near_node],
            np.array([0.0, 0.0, 5.0]),
            np.array([0.0, 0.0, -1.0]),
        )

        self.assertEqual(selected, near_node["id"])

    def test_geometry_and_transform_survive_project_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Project.create(Path(directory), "Model test", "object")
            node = create_node("sphere")
            node["transform"] = {
                "position": [1.25, -2.0, 0.5],
                "rotation": [10.0, 25.0, 45.0],
                "scale": [1.5, 0.75, 2.0],
            }
            project.data["scenes"][0]["nodes"].append(node)
            project.save()

            reopened = Project.load(project.path)
            loaded = reopened.data["scenes"][0]["nodes"][0]

            self.assertEqual(loaded["id"], node["id"])
            self.assertEqual(loaded["primitive"], "sphere")
            self.assertEqual(loaded["geometry"], node["geometry"])
            self.assertEqual(loaded["transform"], node["transform"])

    def test_legacy_triangle_geometry_is_upgraded_to_editable_faces(self) -> None:
        geometry = {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "triangles": [[0, 1, 2]]}
        ensure_editable_geometry(geometry)
        self.assertEqual(geometry["faces"], [[0, 1, 2]])
        self.assertEqual(geometry["triangles"], [[0, 1, 2]])

    def test_concave_face_triangulation_preserves_polygon_area(self) -> None:
        vertices = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [1, 1, 0], [0, 2, 0]]
        geometry = _geometry(vertices, [[0, 1, 2, 3, 4]])
        triangle_area = sum(
            np.linalg.norm(np.cross(
                np.asarray(vertices[triangle[1]]) - vertices[triangle[0]],
                np.asarray(vertices[triangle[2]]) - vertices[triangle[0]],
            )) / 2
            for triangle in geometry["triangles"]
        )
        self.assertEqual(len(geometry["triangles"]), 3)
        self.assertAlmostEqual(triangle_area, 3.0)


if __name__ == "__main__":
    unittest.main()
