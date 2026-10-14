"""Scene construction, picking, transform, and persistence tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from puki3d.project import Project
from puki3d.scene import (
    MeshTopologyError,
    PRIMITIVES,
    _geometry,
    create_node,
    ensure_editable_geometry,
    inset_face_geometry,
    mirrored_geometry,
    node_matrix,
    primitive_geometry,
    raycast,
)


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

    def test_geometry_builds_shared_edges_and_consistent_normals(self) -> None:
        geometry = primitive_geometry("cube")
        self.assertEqual(len(geometry["edges"]), 12)
        self.assertEqual(len(geometry["face_normals"]), 6)
        self.assertEqual(len(geometry["vertex_normals"]), 8)
        for normal in geometry["face_normals"]:
            self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0)

    def test_geometry_rejects_degenerate_and_self_intersecting_faces(self) -> None:
        with self.assertRaises(MeshTopologyError):
            _geometry([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]])
        vertices = [[0, 0, 0], [2, 2, 0], [0, 2, 0], [2, 0, 0]]
        with self.assertRaises(MeshTopologyError):
            _geometry(vertices, [[0, 1, 2, 3]])

    def test_inset_builds_inner_face_and_shared_border_without_degeneracy(self) -> None:
        original = primitive_geometry("cube")
        inset = inset_face_geometry(original, 1, 0.1)
        self.assertEqual(len(inset["vertices"]), 12)
        self.assertEqual(len(inset["faces"]), 10)
        self.assertEqual(len(inset["edges"]), 20)
        self.assertTrue(all(len(normal) == 3 for normal in inset["face_normals"]))

    def test_inset_rejects_a_distance_that_collapses_the_face(self) -> None:
        with self.assertRaises(MeshTopologyError):
            inset_face_geometry(primitive_geometry("cube"), 1, 0.5)

    def test_mirror_welds_the_center_seam_and_reverses_winding(self) -> None:
        half = _geometry(
            [[0, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0, 0.5, -0.5],
             [0, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [0, 0.5, 0.5]],
            [[0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [0, 1, 5, 4], [3, 2, 6, 7], [4, 5, 6, 7]],
        )
        mirrored, maps = mirrored_geometry(half, axis=0)
        self.assertEqual(len(mirrored["vertices"]), 12)
        self.assertEqual(len(mirrored["faces"]), 10)
        self.assertEqual(len(maps["vertices"]), 12)
        self.assertEqual(mirrored["faces"][-1], [7, 11, 10, 4])
        self.assertEqual(len(mirrored["edges"]), 20)

    def test_mirror_rejects_a_full_mesh_crossing_the_axis(self) -> None:
        with self.assertRaises(MeshTopologyError):
            mirrored_geometry(primitive_geometry("cube"), axis=0)


if __name__ == "__main__":
    unittest.main()
