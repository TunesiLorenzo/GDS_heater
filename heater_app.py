"""GDS Heater Analyzer.

Run:  python heater_app.py [layout.gds]

1. Open a GDS file and pick the top cell.
2. Assign layer roles: Heater, Routing, and optionally Via (if no via layer is
   assigned, routing and heater are connected wherever they overlap).
3. Draw the + and - terminal boxes over the pads (optionally an analysis region).
4. Enter sheet resistances, heater thickness and voltage, then Run.

Navigation: mouse wheel zooms at the cursor, left-drag pans (in Navigate mode),
right- or middle-drag always pans.
"""
import logging
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import heater_core as hc
import layout_render as lr

# equal-aspect axes re-adjust limits on every zoom; matplotlib logs that each time
logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)

ROLES = ["-", "Heater", "Routing", "Via"]
BOXES = {
    "plus":   dict(color="#d62728", text="+ terminal"),
    "minus":  dict(color="#1f77b4", text="- terminal"),
    "region": dict(color="#444444", text="analysis region"),
}
OVERLAYS = ["Layout", "|J| heater", "|J| routing", "Potential"]


class App:
    def __init__(self, root):
        self.root = root
        root.title("GDS Heater Analyzer")
        root.geometry("1400x880")
        self.lib = None
        self.tops = []
        self.layers = {}            # (layer, datatype) -> [arrays]
        self.geoms = {}             # (layer, datatype) -> layout_render.LayerGeom
        self.layout_im = None
        self.last_view = None
        self.render_job = None
        self.render_gen = 0
        self.vis_vars, self.role_vars = {}, {}
        self.boxes = {k: None for k in BOXES}
        self.box_artists = {k: [] for k in BOXES}
        self.mode = "nav"
        self.drag = None
        self.result = None
        self.overlay_im = None
        self.q = queue.Queue()
        self.busy = False
        self._build_ui()
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main, width=360)
        main.add(left, weight=1)
        main.add(right, weight=0)

        # --- figure
        self.fig = Figure(figsize=(9, 8))
        self.ax = self.fig.add_axes([0.07, 0.06, 0.80, 0.92])
        self.cax = self.fig.add_axes([0.89, 0.12, 0.022, 0.76])
        self.cax.set_visible(False)
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_xlabel("x [µm]")
        self.ax.set_ylabel("y [µm]")
        self.ax.text(0.5, 0.5, "Open a GDS file to start", transform=self.ax.transAxes,
                     ha="center", va="center", color="#888", fontsize=13)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.toolbar = NavigationToolbar2Tk(self.canvas, left, pack_toolbar=False)
        self.toolbar.update()
        self.status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        for ev, fn in (("scroll_event", self._on_scroll), ("button_press_event", self._on_press),
                       ("motion_notify_event", self._on_motion), ("button_release_event", self._on_release)):
            self.canvas.mpl_connect(ev, fn)

        pad = dict(padx=8, pady=4)

        # --- layout
        f = ttk.LabelFrame(right, text="Layout")
        f.pack(fill=tk.X, **pad)
        ttk.Button(f, text="Open GDS…", command=self.open_dialog).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.file_var = tk.StringVar(value="no file")
        ttk.Label(f, textvariable=self.file_var, width=30).grid(row=0, column=1, sticky="w")
        ttk.Label(f, text="Top cell").grid(row=1, column=0, sticky="w", padx=4)
        self.cell_var = tk.StringVar()
        self.cell_cb = ttk.Combobox(f, textvariable=self.cell_var, state="readonly", width=28)
        self.cell_cb.grid(row=1, column=1, sticky="we", pady=(0, 4))
        self.cell_cb.bind("<<ComboboxSelected>>", lambda e: self.load_cell(self.cell_var.get()))
        ttk.Button(f, text="Fit view", command=self.fit_view).grid(row=2, column=0, sticky="w", padx=4, pady=(0, 4))

        # --- layers (scrollable)
        f = ttk.LabelFrame(right, text="Layers  (visible · layer/datatype · role)")
        f.pack(fill=tk.BOTH, **pad)
        cv = tk.Canvas(f, height=190, highlightthickness=0)
        sb = ttk.Scrollbar(f, orient="vertical", command=cv.yview)
        self.layer_frame = ttk.Frame(cv)
        self.layer_frame.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=self.layer_frame, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # --- terminals
        f = ttk.LabelFrame(right, text="Terminals")
        f.pack(fill=tk.X, **pad)
        self.mode_var = tk.StringVar(value="nav")
        for i, (m, txt) in enumerate((("nav", "Navigate"), ("plus", "Draw +"),
                                      ("minus", "Draw −"), ("region", "Region"))):
            ttk.Radiobutton(f, text=txt, value=m, variable=self.mode_var, command=self._set_mode,
                            style="Toolbutton").grid(row=0, column=i, padx=2, pady=4, sticky="we")
        self.box_var = tk.StringVar(value=self._box_text())
        ttk.Label(f, textvariable=self.box_var, justify="left", font=("Consolas", 9)).grid(
            row=1, column=0, columnspan=4, sticky="w", padx=4)
        ttk.Button(f, text="Clear region", command=lambda: self._clear_box("region")).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        ttk.Label(f, text="Drag a box over each pad.", foreground="#666").grid(
            row=2, column=2, columnspan=2, sticky="w")

        # --- parameters
        f = ttk.LabelFrame(right, text="Parameters")
        f.pack(fill=tk.X, **pad)
        self.params = {}
        rows = [("rs_h", "Heater sheet resistance [Ω/□]", "10"),
                ("rs_r", "Routing sheet resistance [Ω/□]", "0.05"),
                ("t_h", "Heater thickness [nm]", "100"),
                ("t_r", "Routing thickness [nm] (optional)", "500"),
                ("V", "Applied voltage [V]", "1.0"),
                ("h", "Grid cell [µm] (auto = automatic)", "auto")]
        for i, (k, lbl, d) in enumerate(rows):
            ttk.Label(f, text=lbl).grid(row=i, column=0, sticky="w", padx=4, pady=1)
            var = tk.StringVar(value=d)
            ttk.Entry(f, textvariable=var, width=10).grid(row=i, column=1, sticky="e", padx=4, pady=1)
            self.params[k] = var
        ttk.Button(f, text="Suggest grid", command=self.suggest_grid).grid(row=len(rows), column=0, sticky="w", padx=4, pady=4)
        self.grid_info = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.grid_info, foreground="#666").grid(row=len(rows), column=1, sticky="e", padx=4)

        # --- run + results
        f = ttk.Frame(right)
        f.pack(fill=tk.X, **pad)
        self.run_btn = ttk.Button(f, text="Run analysis", command=self.run)
        self.run_btn.pack(side=tk.LEFT)
        self.prog_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.prog_var, foreground="#666").pack(side=tk.LEFT, padx=8)

        f = ttk.LabelFrame(right, text="Results")
        f.pack(fill=tk.BOTH, expand=True, **pad)
        self.res_text = tk.Text(f, height=14, width=44, font=("Consolas", 9), wrap="word",
                                relief="flat", background=self.root.cget("background"))
        self.res_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.res_text.configure(state="disabled")
        ov = ttk.Frame(f)
        ov.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Label(ov, text="Show:").pack(side=tk.LEFT)
        self.ov_var = tk.StringVar(value="Layout")
        for o in OVERLAYS:
            ttk.Radiobutton(ov, text=o, value=o, variable=self.ov_var, command=self.show_overlay).pack(side=tk.LEFT)

    # ------------------------------------------------------------------ loading
    def open_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("GDSII", "*.gds *.gds2 *.gdsii *.GDS"), ("All", "*.*")])
        if path:
            self.open_file(path)

    def open_file(self, path):
        self.status.set("Reading GDS…"); self.root.update_idletasks()
        try:
            self.lib, self.tops = hc.load_gds(path)
        except Exception as e:
            messagebox.showerror("Could not read GDS", str(e)); return
        import os
        self.file_var.set(os.path.basename(path))
        names = [c.name for c in self.tops]
        self.cell_cb["values"] = names
        self.cell_var.set(names[0])
        self.load_cell(names[0])

    def load_cell(self, name):
        cell = next(c for c in self.tops if c.name == name)
        self.status.set(f"Flattening {name}…"); self.root.update_idletasks()
        self.layers = hc.flatten_cell(cell)
        self.status.set("Indexing polygons…"); self.root.update_idletasks()
        self.geoms = {k: lr.LayerGeom(v) for k, v in self.layers.items()}
        self.result = None
        self.boxes = {k: None for k in BOXES}
        self.box_var.set(self._box_text())
        self._redraw_all()
        self._rebuild_layer_list()
        self._show_results(None)
        n = sum(len(v) for v in self.layers.values())
        self.status.set(f"{name}: {len(self.layers)} layers, {n:,} polygons")

    def _rebuild_layer_list(self):
        old_roles = {k: v.get() for k, v in self.role_vars.items()}
        for w in self.layer_frame.winfo_children():
            w.destroy()
        self.vis_vars, self.role_vars = {}, {}
        for i, key in enumerate(sorted(self.layers)):
            color = mcolors.to_hex(self._color(key))
            tk.Label(self.layer_frame, bg=color, width=2).grid(row=i, column=0, padx=(2, 4), pady=1)
            vv = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.layer_frame, variable=vv,
                            command=lambda k=key: self._toggle_layer(k)).grid(row=i, column=1)
            ttk.Label(self.layer_frame, text=f"{key[0]}/{key[1]}", width=7).grid(row=i, column=2, sticky="w")
            ttk.Label(self.layer_frame, text=f"{len(self.layers[key]):,} poly", foreground="#666",
                      width=10).grid(row=i, column=3, sticky="w")
            rv = tk.StringVar(value=old_roles.get(key, "-"))
            cb = ttk.Combobox(self.layer_frame, textvariable=rv, values=ROLES, state="readonly", width=8)
            cb.grid(row=i, column=4, padx=4)
            cb.bind("<<ComboboxSelected>>", lambda e, k=key: self._style_layer(k))
            self.vis_vars[key], self.role_vars[key] = vv, rv
            self._style_layer(key)

    # ------------------------------------------------------------------ drawing
    def _color(self, key):
        keys = sorted(self.layers)
        return cm.tab10(keys.index(key) % 10) if len(keys) <= 10 else cm.tab20(keys.index(key) % 20)

    def _redraw_all(self):
        self.ax.cla()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_xlabel("x [µm]"); self.ax.set_ylabel("y [µm]")
        self.overlay_im = None
        self.cax.cla(); self.cax.set_visible(False)
        # The layout is one screen-sized bitmap, re-rendered (in a thread) shortly after the
        # view stops changing. While panning/zooming the existing bitmap is just stretched.
        self.layout_im = self.ax.imshow(np.full((1, 1, 4), 255, np.uint8), extent=(0, 1, 0, 1),
                                        interpolation="nearest", zorder=1)
        self.ax.set_autoscale_on(False)  # updating the bitmap extent must not move the view
        self.last_view = None
        self.ax.callbacks.connect("xlim_changed", lambda ax: self._schedule_render())
        self.ax.callbacks.connect("ylim_changed", lambda ax: self._schedule_render())
        self.box_artists = {k: [] for k in BOXES}
        self.fit_view()

    def _style_layer(self, key):
        self._schedule_render(force=True)

    def _toggle_layer(self, key):
        self._schedule_render(force=True)

    def _schedule_render(self, force=False, delay=120):
        if force:
            self.last_view = None
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after(delay, self._start_render)

    def _start_render(self):
        self.render_job = None
        if not self.geoms or self.layout_im is None:
            return
        self.ax.apply_aspect()  # settle the equal-aspect limits before rendering for them
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        bb = self.ax.bbox
        W, H = max(int(bb.width), 2), max(int(bb.height), 2)
        view = (x0, x1, y0, y1)
        if self.last_view is not None:
            lv, lW, lH = self.last_view
            span = max(x1 - x0, y1 - y0)
            if (lW, lH) == (W, H) and max(abs(a - b) for a, b in zip(view, lv)) < 1e-6 * span:
                return
        self.last_view = (view, W, H)
        layers = []
        for key in sorted(self.geoms):
            if not self.vis_vars.get(key) or not self.vis_vars[key].get():
                continue
            role = self.role_vars[key].get()
            alpha = {"Heater": 0.55, "Routing": 0.35, "Via": 0.6}.get(role, 0.18)
            layers.append((self.geoms[key], self._color(key)[:3], alpha))
        self.render_gen += 1
        gen = self.render_gen
        cancel = lambda: self.render_gen != gen

        def work():
            try:
                img = lr.render(layers, view, W, H, cancel=cancel)
                self.q.put(("layout", (gen, img, view)))
            except lr.Cancelled:
                pass

        threading.Thread(target=work, daemon=True).start()

    def fit_view(self):
        bbs = [g.bb for g in self.geoms.values() if g.n]
        if bbs:
            b = np.concatenate(bbs)
            bb = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
            dx, dy = bb[2] - bb[0], bb[3] - bb[1]
            m = 0.04 * max(dx, dy, 1e-3)
            self.ax.set_xlim(bb[0] - m, bb[2] + m)
            self.ax.set_ylim(bb[1] - m, bb[3] + m)
        self.canvas.draw_idle()

    def _draw_box(self, kind):
        for a in self.box_artists[kind]:
            a.remove()
        self.box_artists[kind] = []
        b = self.boxes[kind]
        if b is None:
            return
        st = BOXES[kind]
        r = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=True,
                      facecolor=mcolors.to_rgba(st["color"], 0.10), edgecolor=st["color"],
                      lw=1.8, ls="--" if kind == "region" else "-", zorder=60)
        self.ax.add_patch(r)
        t = self.ax.text(b[0], b[3], " " + st["text"], color=st["color"], va="bottom", ha="left",
                         fontsize=9, fontweight="bold", zorder=61, clip_on=True)
        self.box_artists[kind] = [r, t]

    def _box_text(self):
        lines = []
        for k, st in BOXES.items():
            b = self.boxes[k]
            s = "not set" if b is None else f"({b[0]:.1f}, {b[1]:.1f}) – ({b[2]:.1f}, {b[3]:.1f}) µm"
            if k == "region" and b is None:
                s = "auto (metal net touching the terminals)"
            lines.append(f"{st['text']:<16}{s}")
        return "\n".join(lines)

    def _clear_box(self, kind):
        self.boxes[kind] = None
        self._draw_box(kind)
        self.box_var.set(self._box_text())
        self.canvas.draw_idle()

    def _set_mode(self):
        self.mode = self.mode_var.get()
        if self.mode != "nav" and self.toolbar.mode:
            # leave matplotlib's own pan/zoom so our box drawing receives the clicks
            (self.toolbar.pan if "pan" in str(self.toolbar.mode) else self.toolbar.zoom)()

    # ------------------------------------------------------------------ mouse
    def _on_scroll(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        f = 1 / 1.25 if ev.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(ev.xdata - (ev.xdata - x0) * f, ev.xdata + (x1 - ev.xdata) * f)
        self.ax.set_ylim(ev.ydata - (ev.ydata - y0) * f, ev.ydata + (y1 - ev.ydata) * f)
        self.canvas.draw_idle()

    def _on_press(self, ev):
        if self.toolbar.mode or ev.inaxes is not self.ax:
            return
        if ev.button in (2, 3) or (ev.button == 1 and self.mode == "nav"):
            self.drag = ("pan", ev.x, ev.y, self.ax.get_xlim(), self.ax.get_ylim())
        elif ev.button == 1:
            rect = Rectangle((ev.xdata, ev.ydata), 0, 0, fill=False, ec=BOXES[self.mode]["color"],
                             lw=1.5, ls=":", zorder=70)
            self.ax.add_patch(rect)
            self.drag = ("box", ev.xdata, ev.ydata, rect)

    def _on_motion(self, ev):
        if self.drag and self.drag[0] == "pan":
            _, px, py, xl, yl = self.drag
            bb = self.ax.bbox
            dx = (ev.x - px) * (xl[1] - xl[0]) / bb.width
            dy = (ev.y - py) * (yl[1] - yl[0]) / bb.height
            self.ax.set_xlim(xl[0] - dx, xl[1] - dx)
            self.ax.set_ylim(yl[0] - dy, yl[1] - dy)
            self.canvas.draw_idle()
            return
        if self.drag and self.drag[0] == "box" and ev.xdata is not None:
            _, x0, y0, rect = self.drag
            rect.set_bounds(min(x0, ev.xdata), min(y0, ev.ydata), abs(ev.xdata - x0), abs(ev.ydata - y0))
            self.canvas.draw_idle()
        if ev.inaxes is self.ax and ev.xdata is not None:
            self.status.set(self._probe(ev.xdata, ev.ydata))

    def _on_release(self, ev):
        if not self.drag:
            return
        d, self.drag = self.drag, None
        if d[0] != "box":
            return
        _, x0, y0, rect = d
        rect.remove()
        x1 = ev.xdata if ev.xdata is not None else x0
        y1 = ev.ydata if ev.ydata is not None else y0
        if abs(x1 - x0) > 0 and abs(y1 - y0) > 0:
            self.boxes[self.mode] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self._draw_box(self.mode)
            self.box_var.set(self._box_text())
        self.mode_var.set("nav"); self.mode = "nav"
        self.canvas.draw_idle()

    def _probe(self, x, y):
        s = f"x = {x:.3f} µm   y = {y:.3f} µm"
        r = self.result
        if r is None:
            return s
        g = r.grid
        c, rr = int((x - g.x0) // g.h), int((y - g.y0) // g.h)
        if 0 <= c < g.nx and 0 <= rr < g.ny:
            jh = r.J_heater[rr, c]
            if np.isfinite(jh):
                s += f"   |   J heater = {jh:.4g} mA/µm²  (K = {r.K_heater[rr, c]:.4g} mA/µm)"
            if r.J_routing is not None and np.isfinite(r.J_routing[rr, c]):
                s += f"   J routing = {r.J_routing[rr, c]:.4g} mA/µm²"
            if np.isfinite(r.V_map[rr, c]):
                s += f"   V = {r.V_map[rr, c]:.4g} V"
        return s

    # ------------------------------------------------------------------ analysis
    def _role_keys(self, role):
        return [k for k in self.geoms if self.role_vars[k].get() == role]

    def _role_polys(self, role, region):
        """Polygons of all layers with this role whose bbox touches region."""
        out = []
        for k in self._role_keys(role):
            g = self.geoms[k]
            out += g.select(g.in_box(region))
        return out

    def _region(self):
        """Analysis region: the drawn box, else the bbox of the metal net touching the terminals."""
        if self.boxes["region"]:
            return self.boxes["region"]
        keys = self._role_keys("Heater") + self._role_keys("Routing")
        bbs = [self.geoms[k].bb for k in keys if self.geoms[k].n]
        if not bbs:
            return None
        B = np.concatenate(bbs)
        seeds = [b for b in (self.boxes["plus"], self.boxes["minus"]) if b]

        def touching(boxes):
            hit = np.zeros(len(B), bool)
            for b in boxes:
                hit |= (B[:, 2] >= b[0]) & (B[:, 0] <= b[2]) & (B[:, 3] >= b[1]) & (B[:, 1] <= b[3])
            return hit

        # flood fill over polygon bounding boxes (a superset of the true connectivity)
        seen = touching(seeds)
        front = np.flatnonzero(seen)
        while front.size:
            new = np.zeros(len(B), bool)
            for i in range(0, front.size, 64):
                new |= touching(B[front[i:i + 64]])
            new &= ~seen
            seen |= new
            front = np.flatnonzero(new)
        pts = [B[seen]] + [np.array([b]) for b in seeds]
        A = np.concatenate(pts)
        m = 2.0
        return (A[:, 0].min() - m, A[:, 1].min() - m, A[:, 2].max() + m, A[:, 3].max() + m)

    def suggest_grid(self):
        if not self._role_keys("Heater"):
            messagebox.showinfo("Grid", "Assign at least one layer the role 'Heater' first."); return
        region = self._region()
        heater, routing = self._role_polys("Heater", region), self._role_polys("Routing", region)
        h = hc.auto_cell_size(heater + routing, region)
        self.params["h"].set(f"{h:g}")
        self._update_grid_info(h, region)

    def _update_grid_info(self, h, region):
        nx = int(np.ceil((region[2] - region[0]) / h)); ny = int(np.ceil((region[3] - region[1]) / h))
        self.grid_info.set(f"{nx}×{ny} cells")

    def _float(self, key, name, allow_empty=False):
        s = self.params[key].get().strip()
        if allow_empty and s == "":
            return None
        try:
            v = float(s)
        except ValueError:
            raise ValueError(f"{name}: '{s}' is not a number.")
        if v <= 0:
            raise ValueError(f"{name} must be positive.")
        return v

    def run(self):
        if self.busy:
            return
        if not self.layers:
            messagebox.showinfo("Run", "Open a GDS file first."); return
        try:
            if not self._role_keys("Heater"):
                raise ValueError("Assign the role 'Heater' to at least one layer.")
            if not self.boxes["plus"] or not self.boxes["minus"]:
                raise ValueError("Draw both the + and − terminal boxes (use 'Draw +' / 'Draw −').")
            region = self._region()
            heater, routing, via = (self._role_polys(r, region) for r in ("Heater", "Routing", "Via"))
            if not heater:
                raise ValueError("No heater polygons inside the analysis region.")
            rs_h = self._float("rs_h", "Heater sheet resistance")
            rs_r = self._float("rs_r", "Routing sheet resistance")
            t_h = self._float("t_h", "Heater thickness") / 1000.0
            t_r = self._float("t_r", "Routing thickness", allow_empty=True)
            t_r = t_r / 1000.0 if t_r else None
            V = self._float("V", "Voltage")
            hs = self.params["h"].get().strip().lower()
            h = hc.auto_cell_size(heater + routing, region) if hs in ("", "auto") else self._float("h", "Grid cell")
            g = hc.Grid.from_bbox(region, h)
            if g.nx * g.ny > 60_000_000:
                raise ValueError(f"Grid would be {g.nx}×{g.ny} cells. Increase the grid cell size "
                                 f"or draw a smaller analysis region.")
        except ValueError as e:
            messagebox.showerror("Cannot run", str(e)); return

        self._update_grid_info(h, region)
        boxes = dict(self.boxes)
        self.busy = True
        self.run_btn.state(["disabled"])
        say = lambda s: self.q.put(("progress", s))

        def work():
            try:
                say(f"Rasterising ({g.nx}×{g.ny}, h = {h:g} µm)…")
                hm = hc.rasterize(heater, g)
                rm = hc.rasterize(routing, g) if routing else None
                vm = hc.rasterize(via, g) if via else None
                res = hc.solve(g, hm, rm, vm, rs_h, rs_r, boxes["plus"], boxes["minus"], V, t_h, t_r, progress=say)
                res.stats["t_h_nm"] = t_h * 1e3
                res.stats["t_r_nm"] = t_r * 1e3 if t_r else None
                res.stats["via_used"] = bool(via)
                self.q.put(("done", res))
            except hc.SolveError as e:
                self.q.put(("error", str(e)))
            except MemoryError:
                self.q.put(("error", "Out of memory. Use a larger grid cell or a smaller analysis region."))
            except Exception as e:
                import traceback; traceback.print_exc()
                self.q.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "layout":
                    gen, img, (x0, x1, y0, y1) = payload
                    if gen == self.render_gen and self.layout_im is not None:
                        self.layout_im.set_data(img)
                        self.layout_im.set_extent((x0, x1, y0, y1))
                        self.canvas.draw_idle()
                elif kind == "progress":
                    self.prog_var.set(payload)
                elif kind == "done":
                    self.busy = False; self.run_btn.state(["!disabled"])
                    self.prog_var.set("Done.")
                    self.result = payload
                    self._show_results(payload)
                    self.ov_var.set("|J| heater")
                    self.show_overlay()
                elif kind == "error":
                    self.busy = False; self.run_btn.state(["!disabled"])
                    self.prog_var.set("Failed.")
                    messagebox.showerror("Analysis failed", payload)
        except queue.Empty:
            pass
        self.root.after(30, self._poll)

    def _show_results(self, r):
        self.res_text.configure(state="normal")
        self.res_text.delete("1.0", tk.END)
        if r is not None:
            s = r.stats
            I_mA = r.I * 1e3
            lines = [
                f"Total resistance   {r.R:,.4g} Ω",
                f"  in heater        {r.R_heater:,.4g} Ω",
                f"  in routing       {r.R_routing:,.4g} Ω",
                f"Current @ {r.voltage:g} V     {I_mA:.4g} mA",
                f"Power              {r.P * 1e3:.4g} mW",
                "",
                f"Heater |J|  (t = {s['t_h_nm']:g} nm)",
                f"  typical (median) {s['J_heater_median']:.4g} mA/µm²",
                f"  99th percentile  {s['J_heater_p99']:.4g} mA/µm²",
                f"  peak             {s['J_heater_max']:.4g} mA/µm²",
            ]
            if "J_routing_max" in s:
                lines += [f"Routing |J|  (t = {s['t_r_nm']:g} nm)",
                          f"  99th percentile  {s['J_routing_p99']:.4g} mA/µm²",
                          f"  peak             {s['J_routing_max']:.4g} mA/µm²"]
            w = s.get("heater_width_um")
            lines += ["", f"Grid {r.grid.nx}×{r.grid.ny}, h = {r.grid.h:g} µm, {r.n_nodes:,} unknowns",
                      "Contact: " + ("via layer" if s["via_used"] else "routing/heater overlap") + ", ideal (0 Ω)"]
            if w:
                cells = w / r.grid.h
                lines.append(f"Heater ≈ {w:.3g} µm wide ≈ {cells:.0f} cells")
                if cells < 6:
                    lines.append("⚠ Heater spans few cells: reduce the grid cell for accuracy "
                                 f"(error is roughly h/w ≈ {100 / cells:.0f} %).")
            lines.append("Peak J sits at inner corners (current crowding) and grows as the grid is refined.")
            self.res_text.insert("1.0", "\n".join(lines))
        self.res_text.configure(state="disabled")

    def show_overlay(self):
        if self.overlay_im is not None:
            self.overlay_im.remove(); self.overlay_im = None
        self.cax.cla(); self.cax.set_visible(False)
        r, which = self.result, self.ov_var.get()
        data, label, cmap = None, "", "inferno"
        if r is not None:
            if which == "|J| heater":
                data, label = r.J_heater, "|J| heater [mA/µm²]"
            elif which == "|J| routing":
                if r.J_routing is None:
                    messagebox.showinfo("Routing J", "Enter a routing thickness and run again.")
                else:
                    data, label = r.J_routing, "|J| routing [mA/µm²]"
            elif which == "Potential":
                data, label, cmap = r.V_map, "Potential [V]", "viridis"
        if data is not None and np.isfinite(data).any():
            fin = data[np.isfinite(data)]
            vmax = float(np.percentile(fin, 99.5)) if which != "Potential" else float(fin.max())
            vmin = 0.0 if which != "Potential" else float(fin.min())
            cmap_obj = matplotlib.colormaps[cmap].copy(); cmap_obj.set_bad(alpha=0)
            self.overlay_im = self.ax.imshow(np.ma.masked_invalid(data), extent=r.grid.extent, origin="lower",
                                             cmap=cmap_obj, vmin=vmin, vmax=max(vmax, vmin + 1e-30),
                                             interpolation="nearest", zorder=40)
            self.cax.set_visible(True)
            cb = self.fig.colorbar(self.overlay_im, cax=self.cax, extend="max" if which != "Potential" else "neither")
            cb.set_label(label)
        self.canvas.draw_idle()


def main():
    try:  # crisp rendering on high-DPI Windows displays
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    app = App(root)
    if len(sys.argv) > 1:
        root.after(200, lambda: app.open_file(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    main()
