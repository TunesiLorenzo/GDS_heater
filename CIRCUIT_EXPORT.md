# Lumped resistor SPICE export

Load the setup and click **Export circuit…**. Each MZI has exactly two series
resistors: `Rplus_<name>` for its positive routing and `Rheater_<name>` for
its heater. Its return connects to the finite shared-ground network.

Only the heater returns and ground pad are retained in each ground group.
A group of two MZIs has at most three ground resistors; a group of three has
at most six. The twelve-device Benes setup therefore has 24 positive-route
and heater resistors, plus at most 21 ground resistors.

The export treats each heater/routing contact region as equipotential and
extracts resistances separately for the heater, positive routing, and ground
routing. This is a lumped model. No distributed contact probes or cross-path
resistors between positive routes are exported.

The file is a resistor-only `.subckt heater_circuit`. The comments identify
each device's positive and ground ports. Connect the ground ports to the
controller's external reference and apply the complete drive-voltage vector
to estimate heater voltage and common-ground drops.

Export from a saved setup without opening the GUI:

```powershell
python circuit_export.py Setups/Benes.json -o Setups/Benes_circuit.cir
```

Resistance values use the selected sheet resistances and grid settings.
External cable/package resistance is not inferred from the GDS.
