"""Small heater circuits: one feed and one heater resistor per device."""
import re

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from heater_core import SolveError, _layer_edges


def _sheet(g, mask, resistance):
    ids = np.full(mask.shape, -1, np.int64)
    ids[mask] = np.arange(int(mask.sum()))
    edges = _layer_edges(mask, ids, g)
    a = np.concatenate([edge[0] for edge in edges])
    b = np.concatenate([edge[1] for edge in edges])
    conductance = np.concatenate([edge[3] / resistance for edge in edges])
    rows, cols = np.concatenate((a, b, a, b)), np.concatenate((a, b, b, a))
    values = np.concatenate((conductance, conductance, -conductance, -conductance))
    return ids, sp.csr_matrix((values, (rows, cols)), shape=(int(mask.sum()),) * 2)


def _contract(matrix, electrodes):
    """Idealize each electrode; overlapping electrodes become aliases."""
    owner = np.full(matrix.shape[0], -1, np.int64)
    parent = list(range(len(electrodes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, nodes in enumerate(electrodes):
        for previous in np.unique(owner[nodes]):
            if previous >= 0:
                parent[root(int(previous))] = root(i)
        owner[nodes] = i
    groups = {}
    for i in range(len(electrodes)):
        groups.setdefault(root(i), len(groups))
    terminal = np.array([groups[root(i)] for i in range(len(electrodes))])
    touched = owner >= 0
    mapping = np.empty(matrix.shape[0], np.int64)
    mapping[touched] = terminal[owner[touched]]
    mapping[~touched] = len(groups) + np.arange(int((~touched).sum()))
    size = len(groups) + int((~touched).sum())
    contraction = sp.csr_matrix((np.ones(len(mapping)), (np.arange(len(mapping)), mapping)),
                                shape=(len(mapping), size))
    reduced = (contraction.T @ matrix @ contraction).tocsr()
    adjacency = reduced.copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    _, components = connected_components(adjacency, directed=False)
    return reduced, terminal, components


def _reduce(matrix, components, boundary):
    boundary = np.asarray(boundary, dtype=np.int64)
    if len(set(boundary)) != len(boundary) or np.any(components[boundary] != components[boundary[0]]):
        raise SolveError("A lumped circuit section has shorted or disconnected terminals.")
    interior = np.setdiff1d(np.flatnonzero(components == components[boundary[0]]), boundary)
    nodes = np.concatenate((boundary, interior))
    local = matrix[nodes][:, nodes]
    count = len(boundary)
    admittance = local[:count, :count].toarray()
    if len(interior):
        try:
            coupling = local[count:, :count].tocsc()
            factor = spla.splu(local[count:, count:].tocsc(), permc_spec="MMD_AT_PLUS_A")
            admittance += coupling.T @ factor.solve(-coupling.toarray())
        except (RuntimeError, ValueError) as exc:
            raise SolveError(f"Cannot reduce a lumped circuit section: {exc}") from exc
    admittance = 0.5 * (admittance + admittance.T)
    np.fill_diagonal(admittance, 0)
    scale = max(float(np.abs(admittance).max()), np.finfo(float).tiny)
    if not np.isfinite(admittance).all() or np.any(admittance > 1e-9 * scale):
        raise SolveError("The lumped circuit reduction produced invalid conductances.")
    admittance[admittance > 0] = 0
    np.fill_diagonal(admittance, -admittance.sum(axis=1))
    return admittance


def _two_terminal(matrix, components, a, b):
    conductance = -_reduce(matrix, components, [a, b])[0, 1]
    if conductance <= 0 or not np.isfinite(1 / conductance):
        raise SolveError("A lumped resistor section has no finite positive resistance.")
    return float(1 / conductance)


def extract_lumped_circuit(g, hm, rm, vm, rs_h, rs_r, devices, progress=None):
    """Extract separate feed/heater resistors and small shared-return networks.

    Each connected interlayer contact region becomes one equipotential node.
    Supported layouts have a separate two-terminal positive feed per device,
    two contacts per heater island, and a shared return for up to three heaters.
    A return group must have one shared external minus-pad electrode.
    """
    say = progress or (lambda message: None)
    if not np.isfinite(rs_h) or not np.isfinite(rs_r) or min(rs_h, rs_r) <= 0:
        raise SolveError("Sheet resistances must be finite and positive.")
    hm = np.asarray(hm, dtype=bool)
    rm = np.zeros_like(hm) if rm is None else np.asarray(rm, dtype=bool)
    if hm.shape != (g.ny, g.nx) or rm.shape != hm.shape or (vm is not None and np.shape(vm) != hm.shape):
        raise SolveError("Lumped circuit masks must match the grid.")
    if not hm.any() or not rm.any() or not devices:
        raise SolveError("The lumped circuit requires heaters, routing, and configured devices.")
    names = [d.get("name") for d in devices]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise SolveError("Device names must be nonempty and unique.")
    say("Finding heater islands and contact regions...")
    islands, _ = ndi.label(hm)
    contacts, contact_count = ndi.label(hm & rm & (True if vm is None else np.asarray(vm, dtype=bool)))
    if not contact_count:
        raise SolveError("No heater/routing contacts were found; check the selected via layer.")
    idx_r, routing = _sheet(g, rm, rs_r)
    idx_h, heater = _sheet(g, hm, rs_h)
    contact_cells = [np.nonzero(contacts == i) for i in range(1, contact_count + 1)]
    contact_island = [int(islands[rows[0], cols[0]]) for rows, cols in contact_cells]
    routing_electrodes = [idx_r[cells] for cells in contact_cells]
    heater_electrodes = [idx_h[cells] for cells in contact_cells]
    for device in devices:
        for polarity in ("plus", "minus"):
            try:
                box = np.asarray(device.get(polarity), dtype=float)
                valid = box.shape == (4,) and np.isfinite(box).all()
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise SolveError(f"{device['name']}: set a finite {polarity} pad box.")
            rows, cols = g.box_to_slices(box)
            nodes = np.unique(idx_r[rows, cols])
            nodes = nodes[nodes >= 0]
            if not nodes.size:
                raise SolveError(f"{device['name']}: the {polarity} pad must touch routing metal.")
            if np.any(hm[rows, cols] & ~rm[rows, cols]):
                raise SolveError(f"{device['name']}: pads touching bare heater sheet are unsupported by this lumped topology.")
            routing_electrodes.append(nodes)
    routing, rt, rc = _contract(routing, routing_electrodes)
    heater, ht, hc = _contract(heater, heater_electrodes)
    plus = rt[contact_count::2]
    minus = rt[contact_count + 1::2]
    if set(plus) & set(minus) or len(set(plus)) != len(plus):
        raise SolveError("Each MZI needs its own positive pad, separate from every ground pad.")
    assignment, used_islands = [], set()
    for i, name in enumerate(names):
        contact_indices = [c for c in range(contact_count) if rc[rt[c]] == rc[plus[i]]]
        if (len(contact_indices) != 1 or np.count_nonzero(rc[plus] == rc[plus[i]]) != 1 or
                np.any(rc[minus] == rc[plus[i]])):
            raise SolveError(f"{name}: the positive feed must connect only its pad and one heater contact.")
        high = contact_indices[0]
        island = contact_island[high]
        ends = [c for c, candidate in enumerate(contact_island) if candidate == island]
        if len(ends) != 2 or island in used_islands:
            raise SolveError(f"{name}: each configured MZI must map to one distinct heater island with exactly two contacts.")
        low = next(c for c in ends if c != high)
        if rc[rt[low]] != rc[minus[i]]:
            raise SolveError(f"{name}: its other heater contact is not connected to the configured ground pad.")
        used_islands.add(island)
        assignment.append((high, low))

    say("Reducing individual positive feeds, heaters, and shared returns...")
    model_devices, resistors, used_names = [], [], set()
    ground_groups = {}
    for i, (name, (high, low)) in enumerate(zip(names, assignment)):
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or f"mzi{i + 1}"
        if suffix.lower() in used_names:
            suffix += f"_{i + 1}"
        used_names.add(suffix.lower())
        group = int(rc[minus[i]])
        ground_groups.setdefault(group, []).append(i)
        rplus = _two_terminal(routing, rc, plus[i], rt[high])
        rheater = _two_terminal(heater, hc, ht[high], ht[low])
        device = {"name": name, "plus_terminal": f"t{i}", "minus_terminal": None,
                  "high_node": f"h_{i + 1}", "return_node": f"ret_{i + 1}",
                  "R_plus_ohm": rplus, "R_heater_ohm": rheater}
        model_devices.append(device)
        resistors.extend([
            {"id": f"Rplus_{suffix}", "a": device["plus_terminal"], "b": device["high_node"],
             "resistance_ohm": rplus, "kind": "positive"},
            {"id": f"Rheater_{suffix}", "a": device["high_node"], "b": device["return_node"],
             "resistance_ohm": rheater, "kind": "heater"}])
    terminals = [d["plus_terminal"] for d in model_devices]
    groups = []
    for number, (component, members) in enumerate(ground_groups.items(), 1):
        pads = np.unique(minus[members])
        lows = [assignment[i][1] for i in members]
        present = {c for c in range(contact_count) if rc[rt[c]] == component}
        if len(members) > 3 or len(pads) != 1 or present != set(lows):
            raise SolveError("Each shared-return group must contain at most three MZIs and one shared minus-pad box, with no other heater contacts.")
        port = f"t{len(devices) + number - 1}"
        terminals.append(port)
        for i in members:
            model_devices[i]["minus_terminal"] = port
        boundary = [int(rt[c]) for c in lows] + [int(pads[0])]
        node_names = [model_devices[i]["return_node"] for i in members] + [port]
        # A ground pad may already cover one or more low contact regions.
        unique_boundary, unique_names = [], []
        for node, node_name in zip(boundary, node_names):
            if node not in unique_boundary:
                unique_boundary.append(node)
                unique_names.append(node_name)
        if len(unique_boundary) != len(boundary):
            raise SolveError("A minus pad overlaps a heater contact; use a separate ground-pad box for this lumped topology.")
        y = _reduce(routing, rc, boundary)
        count = 0
        for a in range(len(boundary)):
            for b in range(a + 1, len(boundary)):
                if y[a, b] < 0:
                    resistance = float(-1 / y[a, b])
                    if not np.isfinite(resistance):
                        raise SolveError("A shared-return resistance exceeds the numerical range.")
                    count += 1
                    resistors.append({"id": f"Rground_{number}_{count}", "a": node_names[a], "b": node_names[b],
                                      "resistance_ohm": resistance, "kind": "ground"})
        groups.append({"devices": [names[i] for i in members], "ground_terminal": port, "resistor_count": count})
    return {"schema": "gds_heater.lumped_dc", "schema_version": 1,
            "terminal_order": terminals, "devices": model_devices, "resistors": resistors,
            "ground_groups": groups, "units": {"resistance": "ohm", "length": "um"},
            "grid": {"nx": g.nx, "ny": g.ny},
            "sheet_resistance_ohm_per_square": {"heater": float(rs_h), "routing": float(rs_r)},
            "boundary_conditions": "Pads and each contiguous interlayer contact region are ideal equipotential electrodes; external drivers define voltage and ground.",
            "warnings": ["Lumped approximation: contact regions are equipotential; finite spreading within those contacts is omitted.",
                         "One positive-feed and one heater resistor per MZI; only routing inside each shared-return group is Kron-reduced."]}


def to_spice(model):
    """Write the small named resistor network without sources or implicit ground."""
    lines = ["* Lumped MZI heater and shared-ground circuit; resistance in ohm.",
             "* One Rplus and one Rheater per MZI; shared-return groups reduced separately.",
             "* Approximation: each contiguous heater/routing contact is equipotential.",
             "* External sources connect positive ports and ground ports; no implicit ground."]
    for device in model["devices"]:
        name = " ".join(device["name"].split())
        lines.append(f"* {name}: + {device['plus_terminal']}, ground {device['minus_terminal']}, heater nodes {device['high_node']} {device['return_node']}")
    lines.append(".subckt heater_circuit " + " ".join(model["terminal_order"]))
    for resistor in model["resistors"]:
        lines.append(f"{resistor['id']} {resistor['a']} {resistor['b']} {resistor['resistance_ohm']:.12g}")
    lines.append(".ends heater_circuit")
    return "\n".join(lines) + "\n"
