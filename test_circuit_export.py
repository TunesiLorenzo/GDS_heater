"""Exercise geometry preparation, file exports, and the GUI export entry point."""
import json
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import circuit_export as ce
import heater_core as hc
from heater_app import App


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def layout():
    layers = {(11, 0): [rect(1, 1, 4, 2), rect(5, 1, 8, 2)],
              (12, 0): [rect(0, 1, 2, 2), rect(7, 1, 9, 2),
                        rect(3, 1, 6, 2), rect(4, 1, 5, 5)]}
    roles = {(11, 0): "Heater", (12, 0): "Routing"}
    ground = (4.1, 4.1, 4.9, 4.9)
    devices = [{"name": "mzi1", "plus": (0.1, 1.1, 0.9, 1.9), "minus": ground},
               {"name": "mzi2", "plus": (8.1, 1.1, 8.9, 1.9), "minus": ground}]
    return layers, roles, devices


def extract(**overrides):
    layers, roles, devices = layout()
    options = dict(rs_heater=100, rs_routing=10, h_fine=1, h_coarse=1)
    options.update(overrides)
    return ce.build_circuit(layers, roles, devices, **options)


class ExportTests(unittest.TestCase):
    def test_auto_region_preserves_common_return_and_device_names(self):
        model = extract(source={"cell": "TEST"})
        self.assertEqual([d["name"] for d in model["devices"]], ["mzi1", "mzi2"])
        self.assertEqual(len(model["resistors"]), 7)
        for device in model["devices"]:
            self.assertAlmostEqual(device["R_plus_ohm"], 10)
            self.assertAlmostEqual(device["R_heater_ohm"], 200)
        self.assertEqual(model["source"]["region_mode"], "auto")
        self.assertEqual(model["source"]["cell"], "TEST")

    def test_explicit_region_cannot_silently_exclude_a_pad(self):
        with self.assertRaisesRegex(ValueError, "every configured pad"):
            extract(region=(0, 0, 7, 6))
        self.assertEqual(extract(region=(0, 0, 9, 6))["source"]["region_mode"], "explicit")

    def test_via_role_without_vias_does_not_create_false_contacts(self):
        layers, roles, devices = layout()
        roles[(30, 0)] = "Via"
        with self.assertRaises(hc.SolveError):
            ce.build_circuit(layers, roles, devices, rs_heater=100, rs_routing=10,
                             h_fine=1, h_coarse=1)

    def test_reject_nonfinite_parameters(self):
        for name in ("rs_heater", "rs_routing", "h_fine", "h_coarse"):
            with self.subTest(parameter=name), self.assertRaises(ValueError):
                extract(**{name: float("nan")})

    def test_json_and_spice_are_complete_and_existing_file_survives_bad_data(self):
        model = extract()
        with tempfile.TemporaryDirectory() as folder:
            json_path, spice_path = Path(folder) / "model.json", Path(folder) / "model.cir"
            ce.write_circuit(model, json_path)
            ce.write_circuit(model, spice_path)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), model)
            self.assertIn(".subckt heater_circuit", spice_path.read_text(encoding="utf-8"))
            original = json_path.read_bytes()
            with self.assertRaises(ValueError):
                ce.write_circuit({"bad": float("nan")}, json_path)
            self.assertEqual(json_path.read_bytes(), original)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["model.cir", "model.json"])

    def test_saved_setup_loads_relative_gds_path_and_exports_without_gui(self):
        import gdstk
        layers, roles, devices = layout()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            lib = gdstk.Library()
            cell = lib.new_cell("TEST")
            for (layer, datatype), polys in layers.items():
                for poly in polys:
                    cell.add(gdstk.Polygon(poly, layer=layer, datatype=datatype))
            lib.write_gds(str(base / "test.gds"))
            setup = base / "setup.json"
            setup.write_text(json.dumps({"gds_file": "test.gds", "cell": "TEST", "devices": devices,
                                         "layer_roles": {f"{k[0]}/{k[1]}": v for k, v in roles.items()},
                                         "parameters": {"rs_h": "100", "rs_r": "10", "h": "1", "hmax": "1"}}),
                             encoding="utf-8")
            model = ce.circuit_from_setup(setup)
            self.assertEqual(len(model["resistors"]), 7)
            for device in model["devices"]:
                self.assertAlmostEqual(device["R_plus_ohm"], 10)
                self.assertAlmostEqual(device["R_heater_ohm"], 200)
            self.assertEqual(model["source"]["cell"], "TEST")


class GuiExportTests(unittest.TestCase):
    def make_app(self):
        app = App.__new__(App)
        app.layers, roles, app.devices = layout()
        app.role_vars = {k: Mock(get=Mock(return_value=v)) for k, v in roles.items()}
        app.params = {k: Mock(get=Mock(return_value=v)) for k, v in
                      {"rs_h": "100", "rs_r": "10", "h": "1", "hmax": "1"}.items()}
        app._role_keys = Mock(return_value=[(11, 0)])
        app.file_var = Mock(get=Mock(return_value="layout.gds"))
        app.cell_var = Mock(get=Mock(return_value="TEST"))
        app.gds_path = "layout.gds"
        app.region = None
        app.results = {"mzi1": "old isolated result"}
        app.q = queue.Queue()
        app.busy = False
        app.run_btn, app.circuit_btn, app.prog_var = Mock(), Mock(), Mock()
        return app

    def test_spice_is_default_and_worker_uses_fresh_joint_model(self):
        app = self.make_app()
        model = {"devices": app.devices, "terminal_order": ["t0", "t1", "t2"]}

        def start_thread(*, target, **kwargs):
            return Mock(start=target)

        with (patch("heater_app.filedialog.asksaveasfilename", return_value="circuit.cir") as dialog,
              patch("heater_app.threading.Thread", side_effect=start_thread),
              patch("heater_app.ce.build_circuit", return_value=model) as build,
              patch("heater_app.ce.write_circuit") as write):
            app.export_circuit()
        self.assertEqual(dialog.call_args.kwargs["defaultextension"], ".cir")
        self.assertEqual(len(build.call_args.args[2]), 2)
        write.assert_called_once_with(model, "circuit.cir")
        self.assertEqual(app.results, {"mzi1": "old isolated result"})
        self.assertEqual(app.q.get_nowait(), ("circuit_done", ("circuit.cir", None, (2, 3))))

    def test_export_rejects_missing_pads_before_file_dialog(self):
        app = self.make_app()
        app.devices[1]["minus"] = None
        with (patch("heater_app.messagebox.showerror") as error,
              patch("heater_app.filedialog.asksaveasfilename") as dialog):
            app.export_circuit()
        error.assert_called_once()
        dialog.assert_not_called()


if __name__ == "__main__":
    unittest.main()
