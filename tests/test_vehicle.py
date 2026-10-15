"""Vehicle structure metadata and inspector behavior."""

import copy
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from puki3d.project import Project
from puki3d.vehicle import (
    axle_dimensions, auto_classify, mount_point_world_position,
    vehicle_diagnostics, vehicle_preview_node_matrix,
)
from puki3d.vehicle_ui import VehicleSystemDialog
from puki3d.vehicle_preview import VehiclePreviewDialog


def sample_nodes():
    names_and_positions = (
        ("Body shell", [0, 0, 0.3]),
        ("Cabin roof", [0, 0, 1.4]),
        ("Chassis frame", [0, 0, -0.2]),
        ("Front left tire", [1.4, 0.9, 0]),
        ("Front right tire", [1.4, -0.9, 0]),
        ("Rear left tire", [-1.3, 0.9, 0]),
        ("Rear right tire", [-1.3, -0.9, 0]),
        ("Windshield glass", [0.5, 0, 1]),
        ("Headlight", [2.2, 0.6, 0.5]),
    )
    return [
        {"id": f"node-{index}", "name": name,
         "transform": {"position": position, "rotation": [0, 0, 0], "scale": [0.8, 0.8, 0.2]}}
        for index, (name, position) in enumerate(names_and_positions)
    ]


class VehicleModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_name_suggestions_assign_main_parts_wheels_and_roles(self):
        nodes = sample_nodes()
        auto_classify(nodes)
        self.assertEqual(nodes[2]["vehicle"]["component"], "Chassis")
        self.assertEqual(nodes[2]["vehicle"]["role"], "chassis_part")
        self.assertEqual(nodes[3]["vehicle"]["wheel"]["side"], "left")
        self.assertEqual(nodes[3]["vehicle"]["wheel"]["axle"], "front")
        self.assertEqual(nodes[4]["vehicle"]["wheel"]["side"], "right")
        self.assertEqual(nodes[7]["vehicle"]["role"], "glass")
        self.assertEqual(nodes[8]["vehicle"]["role"], "light")

    def test_diagnostics_detect_unclassified_parts_wheels_chassis_and_duplicate_mounts(self):
        nodes = sample_nodes()
        auto_classify(nodes)
        data = {"project": {"name": "Pickup"}, "vehicle_system": {
            "name": "Pickup", "chassis_node_id": nodes[2]["id"],
            "mount_points": [{"name": "Front plate"}, {"name": "Front plate"}],
        }}
        self.assertIn("Hay mount points con nombres duplicados.", vehicle_diagnostics(data, nodes))
        nodes[3]["vehicle"]["wheel"]["side"] = "center"
        nodes[0]["vehicle"]["role"] = None
        self.assertTrue(any("piezas sin rol" in message for message in vehicle_diagnostics(data, nodes)))
        self.assertTrue(any("Ruedas mal clasificadas" in message for message in vehicle_diagnostics(data, nodes)))

    def test_inspector_applies_suggestions_and_mount_points_to_compatible_project_data(self):
        nodes = sample_nodes()
        original = {"project": {"name": "Pickup"}, "scenes": [{"nodes": []}],
                    "vehicle_system": {"mount_points": [
                        {"name": "Front plate", "type": "Plate", "position": [2.3, 0.0, 0.4]},
                    ]}}
        dialog = VehicleSystemDialog(original, nodes, nodes[0]["id"])
        self.assertEqual(dialog.mounts_table.cellWidget(0, 3).value(), 2.3)
        self.assertEqual(dialog.mounts_table.cellWidget(0, 5).value(), 0.4)
        dialog._suggest_classifications()
        self.assertEqual(sum(n.get("vehicle", {}).get("role") == "wheel" for n in dialog.nodes), 4)
        dialog.add_mount_point()
        self.assertEqual(dialog.meta["mount_points"][1]["attached_node_id"], nodes[0]["id"])
        self.assertEqual(dialog.meta["mount_points"][1]["position"], [0.0, 0.0, 0.0])
        dialog.mounts_table.item(0, 0).setText("Front plate")
        dialog.mounts_table.cellWidget(0, 1).setCurrentText("Plate")
        dialog.accept()
        self.assertEqual(dialog.result(), dialog.DialogCode.Accepted)
        self.assertEqual(len(original["vehicle_system"]["mount_points"]), 1)
        self.assertEqual(dialog.project_data["vehicle_system"]["type"], "Pickup")

    def test_project_round_trip_preserves_vehicle_hierarchy_and_wheel_specs(self):
        with tempfile.TemporaryDirectory() as root:
            project = Project.create(Path(root), "Pickup", "vehicle")
            nodes = sample_nodes()
            auto_classify(nodes)
            project.data["scenes"][0]["nodes"] = nodes
            project.data["vehicle_system"] = {
                "name": "Pickup 4x4", "type": "Pickup", "chassis_node_id": nodes[2]["id"],
                "mount_points": [{"id": "mount-1", "name": "Plate front", "type": "Plate",
                                  "position": [2.3, 0, 0.4], "rotation": [0, 0, 0]}],
            }
            expected = copy.deepcopy(project.data["vehicle_system"])
            project.save()
            reopened = Project.load(project.path)
            self.assertEqual(reopened.data["vehicle_system"], expected)
            self.assertEqual(reopened.data["scenes"][0]["nodes"][3]["vehicle"]["wheel"]["radius"], 0.4)


    def test_axle_dimensions_are_derived_from_configured_wheel_centers(self):
        nodes = sample_nodes()
        auto_classify(nodes)
        dimensions = axle_dimensions(nodes)
        self.assertAlmostEqual(dimensions["wheelbase"], 2.7)
        self.assertAlmostEqual(dimensions["front_track"], 1.8)
        self.assertAlmostEqual(dimensions["rear_track"], 1.8)

    def test_preview_matrix_moves_and_rotates_wheels_without_changing_saved_transform(self):
        node = sample_nodes()[3]
        auto_classify([node])
        original = copy.deepcopy(node["transform"])
        straight = vehicle_preview_node_matrix(node)
        angled = vehicle_preview_node_matrix(node, steering_angle=0.5, max_steering_angle=40)
        lifted = vehicle_preview_node_matrix(node, suspension_fraction=1, suspension_travel=0.2)
        self.assertNotEqual(angled[:3, :3].tolist(), straight[:3, :3].tolist())
        np.testing.assert_allclose(lifted[:3, 3], np.asarray(straight[:3, 3]) + [0, 0, 0.2])
        self.assertEqual(node["transform"], original)

    def test_mount_point_follows_attached_component_transform(self):
        node = sample_nodes()[0]
        point = {"position": [0.25, 0.0, 0.0], "attached_node_id": node["id"]}
        np.testing.assert_allclose(mount_point_world_position(point, [node]), [0.2, 0.0, 0.3])

    def test_wheel_orientation_drive_steering_and_axle_controls_are_editable(self):
        nodes = sample_nodes()
        auto_classify(nodes)
        dialog = VehicleSystemDialog({"project": {"name": "Pickup"}}, nodes)
        self.assertIn("Distancia entre ejes: 2.700 m", dialog.axle_metrics.text())
        self.assertEqual(dialog.max_steering.value(), 35.0)
        self.assertEqual(dialog.suspension_travel.value(), 0.25)
        dialog.wheels_table.cellWidget(0, 6).setValue(15.0)
        dialog.wheels_table.cellWidget(0, 11).setChecked(True)
        dialog.wheels_table.cellWidget(1, 12).setChecked(True)
        dialog.suspension_travel.setValue(0.35)
        dialog.max_steering.setValue(38.0)
        self.assertEqual(dialog.nodes[3]["transform"]["rotation"][2], 15.0)
        self.assertTrue(dialog.nodes[3]["vehicle"]["wheel"]["steerable"])
        self.assertTrue(dialog.nodes[4]["vehicle"]["wheel"]["driven"])
        self.assertEqual(dialog.meta["suspension"]["travel"], 0.35)
        self.assertEqual(dialog.meta["steering"]["max_angle"], 38.0)

    def test_locator_can_be_attached_to_component_and_uses_local_coordinates(self):
        nodes = sample_nodes()
        data = {"project": {"name": "Pickup"}, "vehicle_system": {"mount_points": [
            {"id": "loc-1", "name": "Mirror left", "type": "Mirror", "position": [0.2, 0, 0],
             "rotation": [0, 0, 0], "attached_node_id": "node-0"},
        ]}}
        dialog = VehicleSystemDialog(data, nodes, selected_mount_id="loc-1")
        self.assertEqual(dialog.mounts_table.currentRow(), 0)
        attached = dialog.mounts_table.cellWidget(0, 2)
        self.assertEqual(attached.currentData(), "node-0")
        np.testing.assert_allclose(mount_point_world_position(dialog.meta["mount_points"][0], nodes), [0.16, 0, 0.3])

    def test_changing_locator_attachment_preserves_its_world_position(self):
        nodes = sample_nodes()
        data = {"project": {"name": "Pickup"}, "vehicle_system": {"mount_points": [
            {"id": "loc-1", "name": "Front plate", "type": "Plate", "position": [1.0, 0.0, 0.5]},
        ]}}
        dialog = VehicleSystemDialog(data, nodes)
        target = dialog.mounts_table.cellWidget(0, 2)
        target.setCurrentIndex(target.findData("node-0"))
        point = dialog.meta["mount_points"][0]
        np.testing.assert_allclose(mount_point_world_position(point, dialog.nodes), [1.0, 0.0, 0.5])
        target.setCurrentIndex(target.findData(None))
        np.testing.assert_allclose(point["position"], [1.0, 0.0, 0.5])

    def test_preview_controls_are_temporary_and_reset_when_closed(self):
        class PreviewViewport:
            state = None

            def set_vehicle_preview(self, active, **values):
                self.state = {"active": active, **values}

        viewport = PreviewViewport()
        config = {"steering": {"max_angle": 40}, "suspension": {"travel": 0.3}}
        dialog = VehiclePreviewDialog(viewport, config)
        dialog.steering_slider.setValue(50)
        dialog.rotation_slider.setValue(180)
        dialog.suspension_slider.setValue(-50)
        self.assertEqual(viewport.state, {"active": True, "steering": 0.5, "wheel_rotation": 180.0, "suspension": -0.5})
        self.assertEqual(config, {"steering": {"max_angle": 40}, "suspension": {"travel": 0.3}})
        dialog.close()
        self.assertFalse(viewport.state["active"])


if __name__ == "__main__":
    unittest.main()
