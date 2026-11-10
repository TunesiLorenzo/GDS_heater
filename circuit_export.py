"""Build a lumped resistor circuit from a layout or a saved analyzer setup.

Command line: python circuit_export.py Setups/Benes.json -o circuit.cir
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

import heater_core as hc
import lumped_circuit as circuit


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Every pad and region must have four coordinates.")
    box = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in box) or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("A pad or region has invalid coordinates.")
    return box


def _intersects(bounds, box):
    return ((bounds[:, 2] >= box[0]) & (bounds[:, 0] <= box[2]) &
            (bounds[:, 3] >= box[1]) & (bounds[:, 1] <= box[3]))


def build_circuit(layers, layer_roles, devices, *, rs_heater, rs_routing,
                  region=None, h_fine=None, h_coarse=None, source=None, progress=None):
    """Jointly extract all devices; all arguments are ordinary Python objects.

    In auto-region mode, polygon bounding-box connectivity conservatively selects
    metal reachable from any pad. Actual connectivity is checked on the mesh.
    Visibility, thickness, and the single-device applied voltage do not affect
    this linear sheet-resistance circuit.
    """
    say = progress or (lambda message: None)
    explicit_region = region is not None
    rs_heater = _positive(rs_heater, "Heater sheet resistance")
    rs_routing = _positive(rs_routing, "Routing sheet resistance")
    h_fine = None if h_fine is None else _positive(h_fine, "Heater grid cell")
    h_coarse = None if h_coarse is None else _positive(h_coarse, "Maximum grid cell")
    if not devices:
        raise ValueError("Add at least one device before exporting a circuit.")
    names = [str(d["name"]).strip() for d in devices]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Device names must be nonempty and unique.")
    devices = [{"name": name, "plus": _box(d.get("plus")), "minus": _box(d.get("minus"))}
               for name, d in zip(names, devices)]
    by_role = {role: [p for key, polys in layers.items() if layer_roles.get(key) == role
                      for p in polys] for role in ("Heater", "Routing", "Via")}
    heater_count = len(by_role["Heater"])
    if not heater_count:
        raise ValueError("Assign the role 'Heater' to at least one populated layer.")
    metal = by_role["Heater"] + by_role["Routing"]
    bounds = np.array([[*p.min(0), *p.max(0)] for p in metal])
    if region is None:
        # Include full conductors, so the crop cannot silently truncate a return.
        seeds = [d[k] for d in devices for k in ("plus", "minus")]
        selected = np.zeros(len(bounds), dtype=bool)
        frontier = seeds
        while len(frontier):
            hit = np.zeros(len(bounds), dtype=bool)
            for box in frontier:
                hit |= _intersects(bounds, box)
            fresh = hit & ~selected
            selected |= fresh
            frontier = bounds[fresh]
        all_bounds = np.concatenate([bounds[selected], np.asarray(seeds)])
        region = (*all_bounds[:, :2].min(0) - 2.0, *all_bounds[:, 2:].max(0) + 2.0)
    else:
        region = _box(region)
        if any(b[0] < region[0] or b[1] < region[1] or b[2] > region[2] or b[3] > region[3]
               for d in devices for b in (d["plus"], d["minus"])):
            raise ValueError("The analysis region must contain every configured pad. Clear or enlarge it.")
        selected = _intersects(bounds, region)
    heater = [p for p, keep in zip(by_role["Heater"], selected[:heater_count]) if keep]
    routing = [p for p, keep in zip(by_role["Routing"], selected[heater_count:]) if keep]
    if not heater:
        raise ValueError("No heater geometry touches the configured pad networks.")
    via = [p for p in by_role["Via"]
           if _intersects(np.array([[*p.min(0), *p.max(0)]]), region)[0]]
    fine_auto, coarse_auto = hc.auto_grid_sizes(heater, routing, region)
    fine = h_fine or fine_auto
    coarse = max(h_coarse or coarse_auto, fine)
    say("Building the joint circuit grid...")
    grid = hc.graded_grid(region, heater, routing + via, fine, coarse)
    if grid.nx * grid.ny > 60_000_000:
        raise ValueError(f"Joint grid would have {grid.nx:,} x {grid.ny:,} cells. "
                         "Use larger grid cells or export a smaller set of devices.")
    say(f"Rasterising the joint circuit ({grid.nx:,} x {grid.ny:,} cells)...")
    hm = hc.rasterize(heater, grid)
    rm = hc.rasterize(routing, grid) if routing else None
    # An assigned via layer gates contacts even when no vias occur in the crop.
    via_used = "Via" in layer_roles.values()
    vm = hc.rasterize(via, grid) if via_used else None
    model = circuit.extract_lumped_circuit(grid, hm, rm, vm, rs_heater, rs_routing, devices, progress=say)
    model["source"] = dict(source or {})
    model["source"].update({"layer_roles": {f"{k[0]}/{k[1]}": v for k, v in layer_roles.items()},
                            "analysis_region_um": [float(x) for x in region],
                            "region_mode": "explicit" if explicit_region else "auto"})
    return model


def circuit_from_setup(path, progress=None):
    """Load the same JSON setup used by the GUI without opening a Tk window."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    gds_path = (path.parent / data["gds_file"]).resolve()
    if progress:
        progress(f"Reading {gds_path.name}...")
    _, tops = hc.load_gds(str(gds_path))
    cell_name = data.get("cell")
    cell = next((c for c in tops if c.name == cell_name), None) if cell_name else tops[0]
    if cell is None:
        raise ValueError(f"Top cell '{cell_name}' was not found in {gds_path.name}.")
    roles = {tuple(int(x) for x in key.split("/")): role
             for key, role in data.get("layer_roles", {}).items()}
    if any(len(key) != 2 or role not in ("-", "Heater", "Routing", "Via")
           for key, role in roles.items()):
        raise ValueError("Invalid layer roles in setup file.")
    params = data.get("parameters", {})

    def spacing(key):
        value = str(params.get(key, "auto")).strip()
        return None if value.lower() in ("", "auto") else _positive(value, key)

    return build_circuit(hc.flatten_cell(cell), roles, data["devices"],
                         rs_heater=params.get("rs_h", 10), rs_routing=params.get("rs_r", 0.05),
                         region=data.get("region"), h_fine=spacing("h"), h_coarse=spacing("hmax"),
                         source={"gds_file": str(gds_path), "cell": cell.name,
                                 "setup_file": str(path)},
                         progress=progress)


def write_circuit(model, path):
    """Replace the destination only after the complete export has been written."""
    path = Path(path)
    is_spice = path.suffix.lower() in (".cir", ".sp", ".spice", ".ckt")
    if is_spice:
        text = circuit.to_spice(model)
    else:
        text = json.dumps(model, indent=2, allow_nan=False) + "\n"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.resolve().parent,
                                         prefix=".circuit-", suffix=".tmp", delete=False) as stream:
            temp_path = stream.name
            stream.write(text)
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    parser = argparse.ArgumentParser(description="Export a lumped heater/ground resistor circuit from an analyzer setup.")
    parser.add_argument("setup", help="Saved analyzer setup JSON")
    parser.add_argument("-o", "--output", required=True, help="Destination resistor SPICE .cir file")
    args = parser.parse_args()
    try:
        model = circuit_from_setup(args.setup, progress=print)
        write_circuit(model, args.output)
    except (OSError, ValueError, KeyError, TypeError, hc.SolveError, MemoryError) as exc:
        parser.exit(1, f"Circuit export failed: {exc}\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
