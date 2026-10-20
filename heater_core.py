"""Geometry + field solver for GDS heater resistance / current density.

Model
-----
The selected layers are rasterised onto a square grid (cell size h, in um).
Each conductor (routing, heater) is a 2D sheet with its own sheet resistance.
Two adjacent square cells of the same sheet are joined by a conductance 1/Rs.
Where the two sheets are connected (overlap, or overlap AND via if via layers
are assigned) both sheets share one node, i.e. an ideal (zero-ohm) contact.
The + / - terminals are boxes: every node inside a box is held at V / 0 V.
The resulting Laplace problem is solved exactly with a sparse direct solver.

All lengths are in um, resistances in ohm, currents in A.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components


# ----------------------------------------------------------------------------
# GDS loading
# ----------------------------------------------------------------------------

def load_gds(path):
    """Return (library, list of top-level cells) with coordinates in um."""
    import gdstk
    lib = gdstk.read_gds(path, unit=1e-6)
    tops = lib.top_level()
    if not tops:
        tops = list(lib.cells)
    return lib, tops


def flatten_cell(cell):
    """Flatten a cell into {(layer, datatype): [ (N,2) float arrays ]}."""
    out = {}
    for p in cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
        pts = np.asarray(p.points, dtype=float)
        if len(pts) >= 3:
            out.setdefault((p.layer, p.datatype), []).append(pts)
    return out


def polys_bbox(polys):
    """Bounding box (xmin, ymin, xmax, ymax) of a list of point arrays."""
    if not polys:
        return None
    mins = np.min([p.min(0) for p in polys], axis=0)
    maxs = np.max([p.max(0) for p in polys], axis=0)
    return (mins[0], mins[1], maxs[0], maxs[1])


def polys_area(polys):
    a = 0.0
    for p in polys:
        x, y = p[:, 0], p[:, 1]
        a += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return a


# ----------------------------------------------------------------------------
# Rasterisation
# ----------------------------------------------------------------------------

@dataclass
class Grid:
    x0: float
    y0: float
    h: float
    nx: int
    ny: int

    @staticmethod
    def from_bbox(bbox, h):
        x0, y0, x1, y1 = bbox
        nx = max(1, int(np.ceil((x1 - x0) / h)))
        ny = max(1, int(np.ceil((y1 - y0) / h)))
        return Grid(x0, y0, h, nx, ny)

    @property
    def extent(self):
        return (self.x0, self.x0 + self.nx * self.h, self.y0, self.y0 + self.ny * self.h)

    def box_to_slices(self, box):
        """Cells whose centres lie inside box (xmin, ymin, xmax, ymax)."""
        xa, ya, xb, yb = box
        c0 = int(np.clip(np.ceil((min(xa, xb) - self.x0) / self.h - 0.5), 0, self.nx))
        c1 = int(np.clip(np.floor((max(xa, xb) - self.x0) / self.h - 0.5) + 1, 0, self.nx))
        r0 = int(np.clip(np.ceil((min(ya, yb) - self.y0) / self.h - 0.5), 0, self.ny))
        r1 = int(np.clip(np.floor((max(ya, yb) - self.y0) / self.h - 0.5) + 1, 0, self.ny))
        return slice(r0, r1), slice(c0, c1)


def _fill_polygon(mask, pts, g, row_chunk=512):
    """OR the interior of one simple polygon into mask (cell-centre sampling)."""
    # Coordinates in "cell index" space: cell (r, c) has its centre at (c, r).
    x = (pts[:, 0] - g.x0) / g.h - 0.5
    y = (pts[:, 1] - g.y0) / g.h - 0.5
    r_lo = max(int(np.ceil(y.min())), 0)
    r_hi = min(int(np.floor(y.max())), g.ny - 1)
    c_lo = max(int(np.ceil(x.min())), 0)
    c_hi = min(int(np.floor(x.max())), g.nx - 1)
    if r_hi < r_lo or c_hi < c_lo:
        return
    xa, ya = x, y
    xb, yb = np.roll(x, -1), np.roll(y, -1)
    dy = yb - ya
    dy[dy == 0] = 1.0  # horizontal edges never "cross" a row (masked below)
    width = c_hi - c_lo + 1
    for r0 in range(r_lo, r_hi + 1, row_chunk):
        r1 = min(r0 + row_chunk, r_hi + 1)
        rows = np.arange(r0, r1, dtype=float)[:, None]
        cross = (ya[None, :] <= rows) != (yb[None, :] <= rows)
        if not cross.any():
            continue
        xi = np.where(cross, xa + (rows - ya) / dy * (xb - xa), np.inf)
        xi.sort(axis=1)
        k = int(cross.sum(1).max())
        xi = xi[:, :k]
        starts, ends = xi[:, 0::2], xi[:, 1::2]
        valid = np.isfinite(ends)
        rr = np.broadcast_to(np.arange(r1 - r0)[:, None], ends.shape)[valid]
        c0 = np.clip(np.ceil(starts[valid]) - c_lo, 0, width).astype(np.int64)
        c1 = np.clip(np.ceil(ends[valid]) - c_lo, 0, width).astype(np.int64)
        diff = np.zeros((r1 - r0, width + 1), dtype=np.int32)
        np.add.at(diff, (rr, c0), 1)
        np.add.at(diff, (rr, c1), -1)
        mask[r0:r1, c_lo:c_hi + 1] |= np.cumsum(diff, axis=1)[:, :width] > 0


def rasterize(polys, g):
    mask = np.zeros((g.ny, g.nx), dtype=bool)
    for p in polys:
        _fill_polygon(mask, p, g)
    return mask


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------

@dataclass
class Result:
    grid: Grid
    voltage: float
    R: float                 # total resistance between terminals [ohm]
    I: float                 # total current [A]
    P: float                 # total power [W]
    R_heater: float          # share of R dissipated in heater sheet [ohm]
    R_routing: float         # share of R dissipated in routing sheet [ohm]
    J_heater: np.ndarray     # |J| in heater [mA/um^2], NaN outside (ny, nx)
    J_routing: np.ndarray    # |J| in routing [mA/um^2], NaN outside, or None if no thickness
    K_heater: np.ndarray     # |sheet current| in heater [mA/um]
    V_map: np.ndarray        # potential [V] (heater where present, else routing)
    stats: dict = field(default_factory=dict)
    n_nodes: int = 0


class SolveError(Exception):
    pass


def _layer_edges(mask, idx):
    """Return (a, b) node index arrays for horizontal and vertical neighbours."""
    mh = mask[:, :-1] & mask[:, 1:]
    mv = mask[:-1, :] & mask[1:, :]
    return (idx[:, :-1][mh], idx[:, 1:][mh], mh), (idx[:-1, :][mv], idx[1:, :][mv], mv)


def _cell_sheet_current(Ih, Iv, mh, mv, shape, h):
    """|K| per cell [A/um] from edge currents (horizontal Ih on mh, vertical Iv on mv)."""
    ny, nx = shape
    eh = np.zeros((ny, nx - 1)); eh[mh] = Ih
    ev = np.zeros((ny - 1, nx)); ev[mv] = Iv
    sx = np.zeros(shape); cx = np.zeros(shape)
    sx[:, :-1] += eh; sx[:, 1:] += eh
    cx[:, :-1] += mh; cx[:, 1:] += mh
    sy = np.zeros(shape); cy = np.zeros(shape)
    sy[:-1, :] += ev; sy[1:, :] += ev
    cy[:-1, :] += mv; cy[1:, :] += mv
    kx = np.divide(sx, cx, out=np.zeros(shape), where=cx > 0)
    ky = np.divide(sy, cy, out=np.zeros(shape), where=cy > 0)
    return np.hypot(kx, ky) / h


def solve(g, heater_mask, routing_mask, via_mask, rs_heater, rs_routing,
          plus_box, minus_box, voltage, t_heater_um, t_routing_um=None, progress=None):
    say = progress or (lambda s: None)
    shape = (g.ny, g.nx)
    hm = heater_mask
    rm = routing_mask if routing_mask is not None else np.zeros(shape, bool)
    if not hm.any():
        raise SolveError("No heater geometry inside the analysis region.")

    # --- node numbering: routing nodes first, heater nodes reuse routing ids where connected
    say("Numbering nodes...")
    idx_r = np.full(shape, -1, dtype=np.int64)
    n_r = int(rm.sum())
    idx_r[rm] = np.arange(n_r)
    connect = hm & rm
    if via_mask is not None:
        connect &= via_mask
    idx_h = np.full(shape, -1, dtype=np.int64)
    own = hm & ~connect
    n_h = int(own.sum())
    idx_h[own] = n_r + np.arange(n_h)
    idx_h[connect] = idx_r[connect]
    N = n_r + n_h

    # --- conductance matrix
    say(f"Assembling {N:,} nodes...")
    rows, cols, vals = [], [], []
    layers = []
    for mask, idx, rs in ((rm, idx_r, rs_routing), (hm, idx_h, rs_heater)):
        (ah, bh, mh), (av, bv, mv) = _layer_edges(mask, idx)
        layers.append((ah, bh, mh, av, bv, mv, 1.0 / rs))
        for a, b in ((ah, bh), (av, bv)):
            gval = np.full(a.size, 1.0 / rs)
            rows += [a, b, a, b]; cols += [b, a, a, b]; vals += [-gval, -gval, gval, gval]
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    L = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # --- terminals
    def term_nodes(box):
        rs_, cs_ = g.box_to_slices(box)
        ids = np.concatenate([idx_r[rs_, cs_].ravel(), idx_h[rs_, cs_].ravel()])
        return np.unique(ids[ids >= 0])

    plus = term_nodes(plus_box)
    minus = term_nodes(minus_box)
    if plus.size == 0:
        raise SolveError("The + terminal box does not touch any routing/heater metal.")
    if minus.size == 0:
        raise SolveError("The - terminal box does not touch any routing/heater metal.")
    if np.intersect1d(plus, minus).size:
        raise SolveError("The + and - terminal boxes overlap the same metal cells.")

    # --- keep only the part of the network that links + to -
    say("Checking connectivity...")
    adj = L.copy(); adj.setdiag(0); adj.eliminate_zeros()
    _, comp = connected_components(adj, directed=False)
    good = np.intersect1d(np.unique(comp[plus]), np.unique(comp[minus]))
    if good.size == 0:
        raise SolveError("The + and - terminals are not connected through the selected layers.\n"
                         "Check the layer roles, the via layer, and the analysis region.")
    active = np.isin(comp, good)

    fixed = np.zeros(N, bool); fixed[plus] = True; fixed[minus] = True
    v = np.zeros(N)
    v[plus] = voltage
    unk = np.flatnonzero(active & ~fixed)
    fx = np.flatnonzero(fixed)
    say(f"Solving {unk.size:,} unknowns...")
    if unk.size:
        Luu = L[unk][:, unk].tocsc()
        rhs = -(L[unk][:, fx] @ v[fx])
        v[unk] = spla.spsolve(Luu, rhs, permc_spec="MMD_AT_PLUS_A")
    v[~active] = np.nan

    # --- post-processing
    say("Computing current density...")
    vz = np.nan_to_num(v)
    inj = L @ vz
    I = float(inj[plus].sum())
    if not np.isfinite(I) or I <= 0:
        raise SolveError("Solver produced no current; check terminals and sheet resistances.")
    R = voltage / I

    K_maps, powers = [], []
    for (ah, bh, mh, av, bv, mv, gl) in layers:
        Ih = gl * (vz[ah] - vz[bh])
        Iv = gl * (vz[av] - vz[bv])
        powers.append(float((Ih ** 2).sum() / gl + (Iv ** 2).sum() / gl) if gl else 0.0)
        K_maps.append(_cell_sheet_current(Ih, Iv, mh, mv, shape, g.h))
    K_r, K_h = K_maps
    P_r, P_h = powers
    P = voltage * I

    act_h = (idx_h >= 0) & active[idx_h.clip(0)] if N else np.zeros(shape, bool)
    act_r = (idx_r >= 0) & active[idx_r.clip(0)] if N else np.zeros(shape, bool)

    K_heater = np.where(act_h, K_h * 1e3, np.nan)                 # mA/um
    J_heater = K_heater / t_heater_um                              # mA/um^2
    J_routing = None
    if t_routing_um:
        J_routing = np.where(act_r, K_r * 1e3 / t_routing_um, np.nan)

    V_map = np.full(shape, np.nan)
    V_map[act_r] = v[idx_r[act_r]]
    V_map[act_h] = v[idx_h[act_h]]

    # Typical density: heater cells not covered by routing (where it actually carries current)
    bare = act_h & ~rm
    sel = J_heater[bare] if bare.any() else J_heater[act_h]
    stats = {
        "J_heater_max": float(np.nanmax(J_heater)),
        "J_heater_p99": float(np.nanpercentile(sel, 99)),
        "J_heater_median": float(np.nanmedian(sel)),
        "heater_cells": int(act_h.sum()),
        "heater_width_um": _typical_width(bare if bare.any() else act_h, g.h),
    }
    if J_routing is not None and act_r.any():
        stats["J_routing_max"] = float(np.nanmax(J_routing))
        stats["J_routing_p99"] = float(np.nanpercentile(J_routing[act_r], 99))

    return Result(grid=g, voltage=voltage, R=R, I=I, P=P,
                  R_heater=P_h / I ** 2, R_routing=P_r / I ** 2,
                  J_heater=J_heater, J_routing=J_routing, K_heater=K_heater,
                  V_map=V_map, stats=stats, n_nodes=int(unk.size))


def _typical_width(m, h):
    """Trace width estimate 2*area/perimeter (exact for long straight traces)."""
    m8 = np.pad(m, 1).astype(np.int8)
    per = np.count_nonzero(np.diff(m8, axis=0)) + np.count_nonzero(np.diff(m8, axis=1))
    return float(2 * m.sum() / per * h) if per else None


def auto_cell_size(polys, region_bbox, target_cells=600_000, max_grid=30_000_000):
    """Pick h so roughly target_cells conductive cells result, capped by total grid size."""
    area = max(polys_area(polys), 1e-12)
    h = np.sqrt(area / target_cells)
    x0, y0, x1, y1 = region_bbox
    h = max(h, np.sqrt((x1 - x0) * (y1 - y0) / max_grid))
    # round to a "nice" number
    e = 10 ** np.floor(np.log10(h))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= h * 0.999:
            return float(m * e)
    return float(h)
