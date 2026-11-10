"""Small analytic checks of the requested Rplus / Rh / ground equivalent."""

import unittest

import numpy as np

import heater_core as hc
from lumped_circuit import extract_lumped_circuit, to_spice


def fixture(count):
    grid = hc.Grid.from_bbox((0, 0, 9, 2 * count + 4), 1.0)
    heater = np.zeros((grid.ny, grid.nx), dtype=bool)
    routing = np.zeros_like(heater)
    ground_row = 2 * count + 1
    routing[1:ground_row + 1, 7] = True

    def box(row, col):
        return (col + 0.1, row + 0.1, col + 0.9, row + 0.9)

    devices = []
    for index in range(count):
        row = 1 + 2 * index
        heater[row, 3:6] = True  # Two 100-ohm heater edges.
        routing[row, 1:4] = True  # Two 10-ohm positive-feed edges.
        routing[row, 5:8] = True  # 20-ohm return arm to the shared bus.
        devices.append({"name": f"MZI {index + 1}", "plus": box(row, 1),
                        "minus": box(ground_row, 7)})
    return extract_lumped_circuit(grid, heater, routing, None, 100.0, 10.0, devices)


def netlist_solution(model):
    """Solve the emitted resistor circuit with its external grounds at zero."""
    rows = [line.split() for line in to_spice(model).splitlines()
            if line.strip() and line.lstrip()[0].lower() == "r"]
    nodes = sorted({node for _, a, b, _ in rows for node in (a, b)})
    indices = {node: i for i, node in enumerate(nodes)}
    laplacian = np.zeros((len(nodes), len(nodes)))
    for _, a, b, resistance in rows:
        i, j = indices[a], indices[b]
        conductance = 1.0 / float(resistance)
        laplacian[i, i] += conductance
        laplacian[j, j] += conductance
        laplacian[i, j] -= conductance
        laplacian[j, i] -= conductance
    count = len(model["devices"])
    basis = np.zeros((len(nodes), count))
    drive_indices = [indices[device["plus_terminal"]] for device in model["devices"]]
    basis[drive_indices, np.arange(count)] = 1.0
    fixed = {indices[node] for node in model["terminal_order"]}
    free = [index for index in range(len(nodes)) if index not in fixed]
    basis[free] = np.linalg.solve(laplacian[np.ix_(free, free)],
                                 -laplacian[np.ix_(free, sorted(fixed))] @ basis[sorted(fixed)])
    admittance = (laplacian @ basis)[drive_indices]
    return indices, basis, np.linalg.inv(admittance), rows


class LumpedCircuitTests(unittest.TestCase):
    def check_model(self, count, ground_impedance):
        model = fixture(count)
        self.assertEqual(model["schema"], "gds_heater.lumped_dc")
        positive = [resistor for resistor in model["resistors"] if resistor["kind"] == "positive"]
        heaters = [resistor for resistor in model["resistors"] if resistor["kind"] == "heater"]
        grounds = [resistor for resistor in model["resistors"] if resistor["kind"] == "ground"]
        self.assertEqual(len(positive), count)
        self.assertEqual(len(heaters), count)
        self.assertGreater(len(grounds), 0)
        self.assertLessEqual(len(grounds), count * (count + 1) // 2)
        self.assertEqual(len(model["resistors"]), 2 * count + len(grounds))
        allowed_ground_nodes = {device["return_node"] for device in model["devices"]}
        allowed_ground_nodes.update(device["minus_terminal"] for device in model["devices"])
        for resistor in grounds:
            self.assertTrue({resistor["a"], resistor["b"]} <= allowed_ground_nodes)
        for device in model["devices"]:
            supply = next(resistor for resistor in positive
                          if {resistor["a"], resistor["b"]} == {device["plus_terminal"], device["high_node"]})
            heater = next(resistor for resistor in heaters
                          if {resistor["a"], resistor["b"]} == {device["high_node"], device["return_node"]})
            self.assertTrue(supply["id"].lower().startswith("rplus"))
            self.assertTrue(heater["id"].lower().startswith("rh"))
            self.assertAlmostEqual(supply["resistance_ohm"], 20.0, places=7)
            self.assertAlmostEqual(heater["resistance_ohm"], 200.0, places=7)
            self.assertAlmostEqual(device["R_plus_ohm"], 20.0, places=7)
            self.assertAlmostEqual(device["R_heater_ohm"], 200.0, places=7)
        indices, basis, impedance, rows = netlist_solution(model)
        self.assertEqual(len(rows), len(model["resistors"]))
        self.assertTrue(all(float(row[3]) > 0.0 for row in rows))
        expected = np.asarray(ground_impedance) + np.eye(count) * 220.0
        np.testing.assert_allclose(impedance, expected, rtol=1e-9, atol=1e-8)
        currents = np.arange(1, count + 1) * 0.001
        potentials = basis @ (expected @ currents)
        returns = [potentials[indices[device["return_node"]]] for device in model["devices"]]
        np.testing.assert_allclose(returns, np.asarray(ground_impedance) @ currents,
                                   rtol=1e-9, atol=1e-10)

    def test_two_heaters_keep_one_supply_and_heater_resistor_each(self):
        self.check_model(2, [[60.0, 20.0], [20.0, 40.0]])

    def test_three_heaters_need_at_most_six_ground_resistors(self):
        self.check_model(3, [[80.0, 40.0, 20.0], [40.0, 60.0, 20.0], [20.0, 20.0, 40.0]])


if __name__ == "__main__":
    unittest.main()
