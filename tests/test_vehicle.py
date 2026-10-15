"""Vehicle structure metadata and inspector behavior."""

import copy
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from puki3d.project import Project
from puki3d.vehicle import auto_classify, vehicle_diagnostics
from puki3d.vehicle_ui import VehicleSystemDialog


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
        self.assertEqual(dialog.mounts_table.cellWidget(0, 2).value(), 2.3)
        self.assertEqual(dialog.mounts_table.cellWidget(0, 4).value(), 0.4)
        dialog._suggest_classifications()
        self.assertEqual(sum(n.get("vehicle", {}).get("role") == "wheel" for n in dialog.nodes), 4)
        dialog.add_mount_point()
        self.assertEqual(dialog.meta["mount_points"][1]["position"], nodes[0]["transform"]["position"])
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


if __name__ == "__main__":
    unittest.main()
