"""Setup file round-trip tests without a display server."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from heater_app import App


class Variable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class SetupTests(unittest.TestCase):
    def make_app(self, gds_path):
        app = App.__new__(App)
        app.gds_path = str(gds_path)
        app.file_var = Variable(gds_path.name)
        app.cell_var = Variable("HEATER")
        app.tops = [Mock(name="HEATER")]
        app.tops[0].name = "HEATER"
        app.pad_var = Variable("80x100")
        app.region = (0, 0, 300, 200)
        app.devices = [{"name": "heater 1", "plus": (1, 2, 3, 4),
                        "minus": (10, 20, 30, 40)}]
        app.role_vars = {(1, 0): Variable("Heater"), (2, 0): Variable("Routing")}
        app.vis_vars = {(1, 0): Variable(True), (2, 0): Variable(False)}
        app.params = {"rs_h": Variable("11"), "V": Variable("2.5")}
        app.status = Variable()
        app.ov_var = Variable("Potential")
        for name in ("show_overlay", "_refresh_dev_tree", "_draw_region", "_refresh_table",
                     "_show_results", "_schedule_render"):
            setattr(app, name, Mock())
        return app

    def test_save_and_load_complete_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            gds = base / "layout.gds"
            gds.touch()
            setup = base / "settings" / "layout_setup.json"
            setup.parent.mkdir()
            app = self.make_app(gds)
            with patch("heater_app.filedialog.asksaveasfilename", return_value=str(setup)):
                app.save_setup()
            data = json.loads(setup.read_text(encoding="utf-8"))
            self.assertEqual(data["gds_file"], os.path.relpath(gds, setup.parent))
            self.assertEqual(data["devices"][0]["plus"], [1, 2, 3, 4])
            self.assertFalse(data["layer_visibility"]["2/0"])

            loaded = self.make_app(gds)
            loaded.gds_path = None
            loaded.cell_var.set("OTHER")
            loaded.load_cell = Mock()
            loaded.devices = []
            loaded.params["V"].set("1")
            loaded.open_file = Mock(side_effect=lambda path: setattr(loaded, "gds_path", path) or True)
            with (patch("heater_app.filedialog.askopenfilename", return_value=str(setup)),
                  patch("heater_app.hc.load_gds", return_value=(None, loaded.tops))):
                loaded.load_setup()
            loaded.open_file.assert_called_once_with(str(gds))
            loaded.load_cell.assert_called_once_with("HEATER")
            self.assertEqual(loaded.devices, app.devices)
            self.assertEqual(loaded.region, app.region)
            self.assertEqual(loaded.pad_var.get(), "80x100")
            self.assertEqual(loaded.params["V"].get(), "2.5")
            self.assertEqual(loaded.role_vars[(2, 0)].get(), "Routing")
            self.assertFalse(loaded.vis_vars[(2, 0)].get())

    def test_legacy_device_file_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            gds = base / "layout.gds"
            gds.touch()
            setup = base / "old_devices.json"
            setup.write_text(json.dumps({"gds_file": "layout.gds", "cell": "HEATER",
                                         "devices": [{"name": "old", "plus": [1, 2, 3, 4],
                                                      "minus": None}]}), encoding="utf-8")
            app = self.make_app(gds)
            with patch("heater_app.filedialog.askopenfilename", return_value=str(setup)):
                app.load_setup()
            self.assertEqual(app.devices[0]["name"], "old")
            self.assertEqual(app.devices[0]["plus"], (1, 2, 3, 4))
            self.assertEqual(app.role_vars[(1, 0)].get(), "-")
            self.assertTrue(app.vis_vars[(2, 0)].get())


if __name__ == "__main__":
    unittest.main()
