"""Fast screen rendering of large flattened layouts.

Instead of handing millions of vertices to matplotlib, each layer is drawn
into a screen-sized bitmap with PIL, only for polygons inside the view.
Polygons smaller than ~2 px are drawn as single pixels.
"""
import numpy as np
from PIL import Image, ImageDraw


class Cancelled(Exception):
    pass


class LayerGeom:
    """All polygons of one layer packed into flat arrays with per-polygon bboxes."""

    def __init__(self, polys):
        self.polys = polys
        self.n = len(polys)
        lens = np.fromiter((len(p) for p in polys), dtype=np.int64, count=self.n)
        self.off = np.concatenate([[0], np.cumsum(lens)])
        self.V = np.concatenate(polys) if polys else np.zeros((0, 2))
        s = self.off[:-1]
        self.bb = np.column_stack([np.minimum.reduceat(self.V[:, 0], s), np.minimum.reduceat(self.V[:, 1], s),
                                   np.maximum.reduceat(self.V[:, 0], s), np.maximum.reduceat(self.V[:, 1], s)]) \
            if self.n else np.zeros((0, 4))

    def in_box(self, box):
        """Indices of polygons whose bbox intersects box (xmin, ymin, xmax, ymax)."""
        b = self.bb
        return np.flatnonzero((b[:, 2] >= box[0]) & (b[:, 0] <= box[2]) & (b[:, 3] >= box[1]) & (b[:, 1] <= box[3]))

    def select(self, idx):
        return [self.polys[i] for i in idx]


def render_mask(geom, view, W, H, cancel=None):
    """uint8 (H, W) mask: 0 empty, 1 fill, 2 outline."""
    x0, x1, y0, y1 = view
    sx, sy = W / (x1 - x0), H / (y1 - y0)
    img = Image.new("L", (W, H), 0)
    idx = geom.in_box((x0, y0, x1, y1))
    if idx.size == 0:
        return np.zeros((H, W), np.uint8)
    bb = geom.bb[idx]
    tiny = ((bb[:, 2] - bb[:, 0]) * sx < 2) & ((bb[:, 3] - bb[:, 1]) * sy < 2)
    draw = ImageDraw.Draw(img)
    big = idx[~tiny]
    V, off = geom.V, geom.off
    for j, i in enumerate(big):
        if cancel is not None and j % 2000 == 0 and cancel():
            raise Cancelled
        p = V[off[i]:off[i + 1]]
        q = np.empty(p.shape, np.int32)
        q[:, 0] = np.rint((p[:, 0] - x0) * sx)
        q[:, 1] = np.rint((y1 - p[:, 1]) * sy)
        if len(q) > 16:  # drop vertices that land on the same pixel
            keep = np.any(q != np.roll(q, 1, axis=0), axis=1)
            q = q[keep]
            if len(q) < 2:
                q = q[:1]
        if len(q) >= 2:
            draw.polygon(q.ravel().tolist(), fill=1, outline=2)
        else:
            draw.point(q.ravel().tolist(), fill=2)
    m = np.asarray(img).copy()
    if tiny.any():
        tb = bb[tiny]
        cx = np.clip(((tb[:, 0] + tb[:, 2]) / 2 - x0) * sx, 0, W - 1).astype(np.int64)
        cy = np.clip((y1 - (tb[:, 1] + tb[:, 3]) / 2) * sy, 0, H - 1).astype(np.int64)
        m[cy, cx] = 2
    return m


def render(layers, view, W, H, background=(1.0, 1.0, 1.0), cancel=None):
    """layers: list of (LayerGeom, rgb, fill_alpha). Returns RGBA uint8 (H, W, 4)."""
    out = np.empty((H, W, 3), np.float32)
    out[:] = background
    for geom, rgb, a_fill in layers:
        m = render_mask(geom, view, W, H, cancel)
        c = np.asarray(rgb, np.float32)
        for val, a in ((1, a_fill), (2, 0.9)):
            sel = m == val
            if sel.any():
                out[sel] = out[sel] * (1 - a) + c * a
    rgba = np.empty((H, W, 4), np.uint8)
    rgba[..., :3] = np.clip(out * 255, 0, 255)
    rgba[..., 3] = 255
    return rgba
