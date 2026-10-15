"""Tests for SCS-informed ATS truck preparation and diagnostics."""

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from puki3d.ats_config import ATS_REQUIRED_FALLBACKS, ATSConfigDialog, ats_diagnostics, ensure_ats_config
from puki3d.project import Project
from puki3d.vehicle import auto_classify


def vehicle_fixture():
    names = (
        "Body shell", "Cabin roof", "Chassis frame", "Front left tire", "Front right tire",
        "Rear left tire", "Rear right tire", "Windshield glass", "Headlight",
    )
    positions = (
        [0, 0, 0.3], [0, 0, 1.4], [0, 0, -0.2], [1.4, 0.9, 0], [1.4, -0.9, 0],
        [-1.3, 0.9, 0], [-1.3, -0.9, 0], [0.5, 0, 1], [2.2, 0.6, 0.5],
    )
    nodes = [{"id": f"node-{i}", "name": name,
              "transform": {"position": pos, "rotation": [0, 0, 0], "scale": [0.8, 0.8, 0.2]}}
             for i, (name, pos) in enumerate(zip(names, positions))]
    auto_classify(nodes)
    data = {"project": {"name": "Pickup", "type": "vehicle"},
            "vehicle_system": {"name": "Pickup", "type": "Pickup", "chassis_node_id": "node-2",
                               "mount_points": [{"id": "lamp", "name": "headlight.left", "type": "Light"}]},
            "materials": []}
    return data, nodes


def complete_draft(data, nodes):
    cfg = ensure_ats_config(data, nodes)
    cfg["identity"].update({"brand_name": "Puki3D", "brand_id": "puki", "model_id": "pickup", "display_name": "Pickup"})
    cfg["chassis"].update({"drive_layout": "4x2", "axle_count": 2, "max_steering_angle_deg": 38,
                           "suspension_types": ["wishbone", "solidaxle"],
                           "residual_travel_m": [0.22, 0.2], "kerb_weight_kg": [1100, 1000]})
    cfg["engine"].update({"type": "petrol", "volume_l": 2.5, "torque_nm": 350,
                          "torque_peak_rpm": 2800, "rpm_idle": 600, "rpm_limit": 5500})
    cfg["transmission"].update({"ratios_forward": [3.7, 2.1, 1.3, 1.0],
                                "ratios_reverse": [-3.4], "differential_ratio": 3.55})
    cfg["lights"]["headlight_unit"] = "/def/vehicle/truck/puki.pickup/head_light/standard.sii"
    cfg["cameras"]["interior_camera"] = "camera.interior.puki.pickup"
    cfg["sounds"].update({"exterior": "/def/vehicle/truck/puki.pickup/sound/ext.sii",
                           "interior": "/def/vehicle/truck/puki.pickup/sound/int.sii"})
    cfg["fallbacks"] = {name: f"{name}|standard.sii" for name in ATS_REQUIRED_FALLBACKS}
    return cfg


class ATSConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_vehicle_configuration_defaults_derive_axle_dimensions_without_changing_schema(self):
        data, nodes = vehicle_fixture()
        before = data.get("schema_version")
        cfg = ensure_ats_config(data, nodes)
        self.assertEqual(cfg["vehicle_type"], "player_truck")
        self.assertEqual(cfg["chassis"]["axle_count"], 2)
        self.assertAlmostEqual(cfg["chassis"]["wheelbase_m"], 2.7)
        self.assertAlmostEqual(cfg["chassis"]["front_track_m"], 1.8)
        self.assertEqual(data.get("schema_version"), before)

    def test_validator_reports_scs_required_fallbacks_and_unverified_materials(self):
        data, nodes = vehicle_fixture()
        ensure_ats_config(data, nodes)
        data["materials"] = [{"name": "Paint", "metallic": 0.8}]
        messages = ats_diagnostics(data, nodes)
        codes = {item["code"] for item in messages}
        self.assertIn("fallbacks", codes)
        self.assertIn("materials", codes)
        self.assertIn("verification", codes)

    def test_validator_accepts_well_formed_preparation_but_never_marks_it_game_verified(self):
        data, nodes = vehicle_fixture()
        cfg = complete_draft(data, nodes)
        errors = [item for item in ats_diagnostics(data, nodes) if item["level"] == "error"]
        self.assertEqual(errors, [])
        self.assertEqual(cfg["status"], "prepared_unverified")

    def test_validator_detects_invalid_powertrain_fallback_and_unresolved_dependencies(self):
        data, nodes = vehicle_fixture()
        cfg = complete_draft(data, nodes)
        cfg["engine"]["volume_l"] = 0.2
        cfg["transmission"]["ratios_forward"] = [-2]
        cfg["fallbacks"]["chassis"] = "chassis|../private.sii"
        cfg["dlc_dependencies"] = ["dlc_not_installed"]
        cfg["external_references"] = ["/def/vehicle/shared/foo.sii"]
        messages = ats_diagnostics(data, nodes, installed_dlcs={"dlc_arizona"})
        codes = {item["code"] for item in messages}
        self.assertIn("engine_volume", codes)
        self.assertIn("forward_ratios", codes)
        self.assertIn("fallback_format", codes)
        self.assertIn("dlc_unresolved", codes)
        self.assertIn("external_references", codes)

    def test_dialog_commit_and_project_round_trip_preserve_ats_preparation(self):
        data, nodes = vehicle_fixture()
        complete_draft(data, nodes)
        dialog = ATSConfigDialog(data, nodes)
        dialog._commit()
        expected = dialog.project_data["ats_config"]
        with tempfile.TemporaryDirectory() as folder:
            project = Project.create(Path(folder), "Pickup", "vehicle")
            project.data["vehicle_system"] = data["vehicle_system"]
            project.data["ats_config"] = expected
            project.save()
            reopened = Project.load(project.path)
        self.assertEqual(reopened.data["ats_config"], expected)
        self.assertEqual(reopened.data["ats_config"]["status"], "prepared_unverified")


if __name__ == "__main__":
    unittest.main()
