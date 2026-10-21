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

class Grid:
    """Rectilinear grid given by its cell edge coordinates (uniform or graded)."""

    def __init__(self, xe, ye):
        self.xe = np.asarray(xe, float)
        self.ye = np.asarray(ye, float)
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1])
        self.yc = 0.5 * (self.ye[1:] + self.ye[:-1])
        self.dx = np.diff(self.xe)
        self.dy = np.diff(self.ye)
        self.nx, self.ny = len(self.dx), len(self.dy)
        self.x0, self.y0 = float(self.xe[0]), float(self.ye[0])
        self.h = float(min(self.dx.min(), self.dy.min()))          # finest cell
        self.h_max = float(max(self.dx.max(), self.dy.max()))      # coarsest cell
        self.uniform = np.allclose(self.dx, self.dx[0]) and np.allclose(self.dy, self.dx[0])

    @staticmethod
    def from_bbox(bbox, h):
        x0, y0, x1, y1 = bbox
        nx = max(1, int(np.ceil((x1 - x0) / h)))
        ny = max(1, int(np.ceil((y1 - y0) / h)))
        return Grid(x0 + h * np.arange(nx + 1), y0 + h * np.arange(ny + 1))

    @property
    def extent(self):
        return (self.x0, float(self.xe[-1]), self.y0, float(self.ye[-1]))

    def cell_at(self, x, y):
        """(row, col) of the cell containing (x, y), or None outside the grid."""
        c = int(np.searchsorted(self.xe, x, "right")) - 1
        r = int(np.searchsorted(self.ye, y, "right")) - 1
        return (r, c) if 0 <= c < self.nx and 0 <= r < self.ny else None

    def box_to_slices(self, box):
        """Cells whose centres lie inside box (xmin, ymin, xmax, ymax)."""
        xa, ya, xb, yb = box
        c0 = int(np.searchsorted(self.xc, min(xa, xb), "left"))
        c1 = int(np.searchsorted(self.xc, max(xa, xb), "right"))
        r0 = int(np.searchsorted(self.yc, min(ya, yb), "left"))
        r1 = int(np.searchsorted(self.yc, max(ya, yb), "right"))
        return slice(r0, r1), slice(c0, c1)


def _merge_intervals(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _graded_axis(lo, hi, fine, h_fine, h_coarse, snaps=(), ratio=1.25):
    """Cell edges on [lo, hi]: h_fine inside the `fine` intervals, growing by `ratio` per
    cell away from them up to h_coarse, with edges placed exactly on every `snaps` value."""
    fine = [(max(a, lo), min(b, hi)) for a, b in _merge_intervals(fine) if b > lo and a < hi]
    A = np.array([a for a, _ in fine]) if fine else None
    B = np.array([b for _, b in fine]) if fine else None

    def spacing(x):
        if A is None:
            return h_coarse
        d = float(np.min(np.maximum(np.maximum(A - x, x - B), 0.0)))
        return min(h_coarse, h_fine + (ratio - 1.0) * d)

    bps = np.unique(np.concatenate([[lo, hi], np.ravel(fine) if fine else [],
                                    [v for v in snaps if lo < v < hi]]))
    keep = [bps[0]]
    for v in bps[1:-1]:  # drop breakpoints closer than a quarter fine cell
        if v - keep[-1] >= 0.25 * h_fine:
            keep.append(v)
    if hi - keep[-1] < 0.25 * h_fine and len(keep) > 1:
        keep.pop()
    keep.append(hi)

    edges = [keep[0]]
    for p, q in zip(keep[:-1], keep[1:]):
        pts, x = [p], p
        while x < q:
            step = spacing(x)
            for _ in range(3):  # use the spacing at the middle of the step
                step = spacing(min(x + 0.5 * step, q))
            x += step
            pts.append(x)
        if len(pts) > 2 and (q - pts[-2]) < 0.5 * (pts[-1] - pts[-2]):
            pts.pop()  # last step overshoots by more than half: merge it
        pts = np.asarray(pts)
        pts = p + (pts - p) * (q - p) / (pts[-1] - p)
        if len(pts) == 2 and q - p >= 2 * h_fine:
            pts = np.array([p, 0.5 * (p + q), q])  # at least two cells across a feature
        edges.extend(pts[1:])
    return np.asarray(edges)


def _axis_snaps(polys, max_count=4000):
    """x of vertical edges and y of horizontal edges (Manhattan features)."""
    if not polys:
        return np.array([]), np.array([])
    xs, ys = [], []
    for p in polys:
        q = np.roll(p, -1, axis=0)
        v = np.isclose(p[:, 0], q[:, 0])
        h = np.isclose(p[:, 1], q[:, 1])
        xs.append(p[v, 0]); ys.append(p[h, 1])
    xs = np.unique(np.round(np.concatenate(xs), 6))
    ys = np.unique(np.round(np.concatenate(ys), 6))
    return (xs if xs.size <= max_count else np.array([]),
            ys if ys.size <= max_count else np.array([]))


def graded_grid(region, heater_polys, other_polys, h_fine, h_coarse, margin=None, ratio=1.25):
    """Grid that is fine (h_fine) around the heater polygons and coarse (h_coarse) elsewhere."""
    x0, y0, x1, y1 = region
    h_coarse = max(h_coarse, h_fine)
    if margin is None:
        margin = max(10 * h_fine, 2.0)
    fx, fy = [], []
    for p in heater_polys:
        fx.append((p[:, 0].min() - margin, p[:, 0].max() + margin))
        fy.append((p[:, 1].min() - margin, p[:, 1].max() + margin))
    sx, sy = _axis_snaps(list(heater_polys) + list(other_polys))
    xe = _graded_axis(x0, x1, fx, h_fine, h_coarse, sx, ratio)
    ye = _graded_axis(y0, y1, fy, h_fine, h_coarse, sy, ratio)
    return Grid(xe, ye)


def _fill_polygon(mask, pts, g, row_chunk=512):
    """OR the interior of one simple polygon into mask (cell-centre sampling)."""
    xc, yc = g.xc, g.yc
    r_lo = int(np.searchsorted(yc, pts[:, 1].min(), "left"))
    r_hi = int(np.searchsorted(yc, pts[:, 1].max(), "right")) - 1
    c_lo = int(np.searchsorted(xc, pts[:, 0].min(), "left"))
    c_hi = int(np.searchsorted(xc, pts[:, 0].max(), "right")) - 1
    if r_hi < r_lo or c_hi < c_lo:
        return
    xa, ya = pts[:, 0], pts[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    dy = yb - ya
    dy[dy == 0] = 1.0  # horizontal edges never "cross" a row (masked below)
    width = c_hi - c_lo + 1
    for r0 in range(r_lo, r_hi + 1, row_chunk):
        r1 = min(r0 + row_chunk, r_hi + 1)
        rows = yc[r0:r1, None]
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
        # cells whose centre satisfies start <= xc < end
        c0 = np.clip(np.searchsorted(xc, starts[valid], "left") - c_lo, 0, width)
        c1 = np.clip(np.searchsorted(xc, ends[valid], "left") - c_lo, 0, width)
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


def _layer_edges(mask, idx, g):
    """Neighbour pairs (a, b), their mask and geometric factor (face length / centre distance)
    for horizontal and vertical neighbours. Conductance = factor / Rs."""
    mh = mask[:, :-1] & mask[:, 1:]
    mv = mask[:-1, :] & mask[1:, :]
    fh = g.dy[:, None] / (0.5 * (g.dx[:-1] + g.dx[1:]))[None, :]
    fv = g.dx[None, :] / (0.5 * (g.dy[:-1] + g.dy[1:]))[:, None]
    return ((idx[:, :-1][mh], idx[:, 1:][mh], mh, fh[mh]),
            (idx[:-1, :][mv], idx[1:, :][mv], mv, fv[mv]))


def _cell_sheet_current(Ih, Iv, mh, mv, g):
    """|K| per cell [A/um] from edge currents (horizontal Ih on mh, vertical Iv on mv)."""
    shape = (g.ny, g.nx)
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
    return np.hypot(kx / g.dy[:, None], ky / g.dx[None, :])


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
        (ah, bh, mh, fh), (av, bv, mv, fv) = _layer_edges(mask, idx, g)
        layers.append((ah, bh, mh, fh / rs, av, bv, mv, fv / rs))
        for a, b, gval in ((ah, bh, fh / rs), (av, bv, fv / rs)):
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
    for (ah, bh, mh, gh, av, bv, mv, gv) in layers:
        Ih = gh * (vz[ah] - vz[bh])
        Iv = gv * (vz[av] - vz[bv])
        powers.append(float((Ih ** 2 / gh).sum() + (Iv ** 2 / gv).sum()))
        K_maps.append(_cell_sheet_current(Ih, Iv, mh, mv, g))
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

    # Typical density: heater cells not covered by routing that actually carry current
    # (dead-end branches, e.g. other devices hanging off a shared ground, carry none)
    k_on = K_h[act_h].max() if act_h.any() else 0.0
    carrying = act_h & (K_h > 1e-3 * k_on)
    bare = carrying & ~rm
    sel = J_heater[bare] if bare.any() else J_heater[carrying]
    stats = {
        "J_heater_max": float(np.nanmax(J_heater)),
        "J_heater_p99": float(np.nanpercentile(sel, 99)),
        "J_heater_median": float(np.nanmedian(sel)),
        "heater_cells": int(act_h.sum()),
        "heater_width_um": _typical_width(bare if bare.any() else carrying, g),
    }
    if J_routing is not None and act_r.any():
        stats["J_routing_max"] = float(np.nanmax(J_routing))
        stats["J_routing_p99"] = float(np.nanpercentile(J_routing[act_r], 99))

    return Result(grid=g, voltage=voltage, R=R, I=I, P=P,
                  R_heater=P_h / I ** 2, R_routing=P_r / I ** 2,
                  J_heater=J_heater, J_routing=J_routing, K_heater=K_heater,
                  V_map=V_map, stats=stats, n_nodes=int(unk.size))


def _typical_width(m, g):
    """Trace width estimate 2*area/perimeter (exact for long straight traces)."""
    area = float((m * g.dy[:, None] * g.dx[None, :]).sum())
    vb = np.diff(np.pad(m, ((0, 0), (1, 1))).astype(np.int8), axis=1) != 0   # (ny, nx+1)
    hb = np.diff(np.pad(m, ((1, 1), (0, 0))).astype(np.int8), axis=0) != 0   # (ny+1, nx)
    per = float((vb * g.dy[:, None]).sum() + (hb * g.dx[None, :]).sum())
    return 2 * area / per if per else None


def polys_perimeter(polys):
    return float(sum(np.hypot(*(np.roll(p, -1, axis=0) - p).T).sum() for p in polys))


def _nice_down(h):
    """Largest 'nice' number (1, 2, 2.5, 5 x 10^n) not above h."""
    e = 10 ** np.floor(np.log10(h))
    for m in (10, 5, 2.5, 2, 1):
        if m * e <= h * 1.001:
            return float(m * e)
    return float(e)


def auto_grid_sizes(heater_polys, routing_polys, region, cells_across=10, max_fine_cells=600_000):
    """(h_fine, h_coarse): h_fine puts ~cells_across cells across the heater trace (capped so the
    heater has at most ~max_fine_cells cells); h_coarse resolves the routing with ~3 cells."""
    a_h, p_h = polys_area(heater_polys), polys_perimeter(heater_polys)
    w_h = 2 * a_h / p_h if p_h else 1.0
    h_fine = max(w_h / cells_across, np.sqrt(a_h / max_fine_cells))
    h_fine = _nice_down(h_fine)
    a_r, p_r = polys_area(routing_polys), polys_perimeter(routing_polys)
    w_r = 2 * a_r / p_r if p_r else w_h
    span = max(region[2] - region[0], region[3] - region[1])
    h_coarse = min(max(w_r / 3, h_fine), span / 40)
    return h_fine, max(_nice_down(h_coarse), h_fine)


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
