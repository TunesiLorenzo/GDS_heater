"""GDS Heater Analyzer.

Run:  python heater_app.py [layout.gds]

1. Setup: open a GDS file, pick the top cell, assign layer roles (Heater, Routing,
   optionally Via; without a via layer, routing and heater connect wherever they
   overlap) and enter sheet resistances, thickness and voltage.
2. Devices: add one device per heater. For each, set its + and - pad: click inside
   a pad to snap to it, or drag a box. Devices may share pads (e.g. a common ground).
3. Run: every device is solved on its own (its + at V, its - at 0 V, all other pads
   floating). The table under the layout lists the results per device; select a row
   to see its current-density map.

Navigation: mouse wheel zooms at the cursor, left-drag pans (in Navigate mode),
right- or middle-drag always pans.
"""
import csv
import json
import logging
import os
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
from matplotlib.image import NonUniformImage
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import heater_core as hc
import layout_render as lr

# equal-aspect axes re-adjust limits on every zoom; matplotlib logs that each time
logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)

ROLES = ["-", "Heater", "Routing", "Via"]
COLORS = {"plus": "#d62728", "minus": "#1f77b4", "region": "#444444"}
SIGN = {"plus": "+", "minus": "−"}
OVERLAYS = ["Layout", "|J| heater", "|J| routing", "Potential"]
TABLE_COLS = [  # key, heading, width, anchor
    ("dev", "Device", 120, "w"),
    ("R", "R total [Ω]", 95, "e"),
    ("Rh", "R heater [Ω]", 95, "e"),
    ("Rr", "R routing [Ω]", 95, "e"),
    ("I", "I [mA]", 75, "e"),
    ("P", "P [mW]", 75, "e"),
    ("Jt", "J typ [mA/µm²]", 110, "e"),
    ("J99", "J p99 [mA/µm²]", 110, "e"),
    ("Jpk", "J peak [mA/µm²]", 110, "e"),
    ("Jr", "J routing pk", 95, "e"),
    ("note", "Note", 260, "w"),
]


def _fmt(v):
    return "" if v is None else f"{v:.4g}"


class App:
    def __init__(self, root):
        self.root = root
        root.title("GDS Heater Analyzer")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1400, sw - 40)}x{min(880, sh - 100)}+10+10")
        try:
            root.state("zoomed")  # maximise on Windows so the side panel fits small screens
        except tk.TclError:
            pass
        self.lib = None
        self.tops = []
        self.layers = {}            # (layer, datatype) -> [arrays]
        self.geoms = {}             # (layer, datatype) -> layout_render.LayerGeom
        self.layout_im = None
        self.last_view = None
        self.render_job = None
        self.render_gen = 0
        self.vis_vars, self.role_vars = {}, {}
        self.devices = []           # [{"name", "plus", "minus"}], boxes are (xmin, ymin, xmax, ymax)
        self.region = None          # optional analysis region shared by all devices
        self.results = {}           # device name -> Result, or error message (str)
        self.dev_artists = []
        self.region_artists = []
        self.mode = "nav"
        self.drag = None
        self.result = None          # Result currently shown (map, hover, detail panel)
        self.overlay_im = None
        self.q = queue.Queue()
        self.busy = False
        self._pad_miss = ""
        self._build_ui()
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.PanedWindow(main, orient=tk.VERTICAL)
        right = ttk.Frame(main, width=360)
        main.add(left, weight=1)
        main.add(right, weight=0)

        # --- figure (top of the left side)
        figf = ttk.Frame(left)
        left.add(figf, weight=1)
        self.fig = Figure(figsize=(9, 6))
        self.ax = self.fig.add_axes([0.07, 0.08, 0.80, 0.90])
        self.cax = self.fig.add_axes([0.89, 0.12, 0.022, 0.76])
        self.cax.set_visible(False)
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_xlabel("x [µm]")
        self.ax.set_ylabel("y [µm]")
        self.ax.text(0.5, 0.5, "Open a GDS file to start", transform=self.ax.transAxes,
                     ha="center", va="center", color="#888", fontsize=13)
        self.canvas = FigureCanvasTkAgg(self.fig, master=figf)
        self.toolbar = NavigationToolbar2Tk(self.canvas, figf, pack_toolbar=False)
        self.toolbar.update()
        self.status = tk.StringVar(value="")
        ttk.Label(figf, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        for ev, fn in (("scroll_event", self._on_scroll), ("button_press_event", self._on_press),
                       ("motion_notify_event", self._on_motion), ("button_release_event", self._on_release)):
            self.canvas.mpl_connect(ev, fn)

        # --- results table (bottom of the left side)
        tf = ttk.Frame(left)
        left.add(tf, weight=0)
        bar = ttk.Frame(tf)
        bar.pack(fill=tk.X, padx=6, pady=(4, 2))
        ttk.Label(bar, text="Results per device", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(bar, text="Save results…", command=self.export_csv).pack(side=tk.RIGHT)
        ttk.Button(bar, text="Copy table", command=self.copy_table).pack(side=tk.RIGHT, padx=4)
        self.table_info = tk.StringVar(value="Run the analysis to fill this table.")
        ttk.Label(bar, textvariable=self.table_info, foreground="#666").pack(side=tk.LEFT, padx=10)
        tw = ttk.Frame(tf)
        tw.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        self.table = ttk.Treeview(tw, columns=[c[0] for c in TABLE_COLS], show="headings",
                                  height=5, selectmode="browse")
        for key, head, width, anchor in TABLE_COLS:
            self.table.heading(key, text=head)
            self.table.column(key, width=width, anchor=anchor, stretch=key == "note")
        self.table.tag_configure("error", foreground="#b00020")
        xs = ttk.Scrollbar(tw, orient="horizontal", command=self.table.xview)
        ys = ttk.Scrollbar(tw, orient="vertical", command=self.table.yview)
        self.table.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tw.rowconfigure(0, weight=1); tw.columnconfigure(0, weight=1)
        self.table.bind("<<TreeviewSelect>>", self._on_table_select)

        pad = dict(padx=8, pady=4)

        # --- run bar: always visible below the tabs
        f = ttk.Frame(right)
        f.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=6)
        self.run_btn = ttk.Button(f, text="Run all devices", command=self.run)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(f, text="Fit view", command=self.fit_view).pack(side=tk.RIGHT)
        self.prog_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.prog_var, foreground="#666", wraplength=170).pack(side=tk.LEFT, padx=8)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))
        setup, devices, results = ttk.Frame(self.tabs), ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        self.tabs.add(setup, text=" Setup ")
        self.tabs.add(devices, text=" Devices ")
        self.tabs.add(results, text=" Device result ")

        # ===== Setup tab
        f = ttk.LabelFrame(setup, text="Layout")
        f.pack(fill=tk.X, **pad)
        ttk.Button(f, text="Open GDS…", command=self.open_dialog).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.file_var = tk.StringVar(value="no file")
        ttk.Label(f, textvariable=self.file_var, width=30).grid(row=0, column=1, sticky="w")
        ttk.Label(f, text="Top cell").grid(row=1, column=0, sticky="w", padx=4)
        self.cell_var = tk.StringVar()
        self.cell_cb = ttk.Combobox(f, textvariable=self.cell_var, state="readonly", width=28)
        self.cell_cb.grid(row=1, column=1, sticky="we", pady=(0, 4))
        self.cell_cb.bind("<<ComboboxSelected>>", lambda e: self.load_cell(self.cell_var.get()))

        f = ttk.LabelFrame(setup, text="Layers  (visible · layer/datatype · role)")
        f.pack(fill=tk.BOTH, expand=True, **pad)
        cv = tk.Canvas(f, height=110, highlightthickness=0)
        sb = ttk.Scrollbar(f, orient="vertical", command=cv.yview)
        self.layer_frame = ttk.Frame(cv)
        self.layer_frame.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=self.layer_frame, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        f = ttk.LabelFrame(setup, text="Parameters")
        f.pack(fill=tk.X, **pad)
        self.params = {}
        rows = [("rs_h", "Heater sheet resistance [Ω/□]", "10"),
                ("rs_r", "Routing sheet resistance [Ω/□]", "0.05"),
                ("t_h", "Heater thickness [nm]", "100"),
                ("t_r", "Routing thickness [nm] (optional)", "500"),
                ("V", "Applied voltage [V]", "1.0"),
                ("h", "Grid cell at heater [µm] (auto)", "auto"),
                ("hmax", "Max grid cell elsewhere [µm] (auto)", "auto")]
        for i, (k, lbl, d) in enumerate(rows):
            ttk.Label(f, text=lbl).grid(row=i, column=0, sticky="w", padx=4, pady=1)
            var = tk.StringVar(value=d)
            ttk.Entry(f, textvariable=var, width=10).grid(row=i, column=1, sticky="e", padx=4, pady=1)
            self.params[k] = var

        # ===== Devices tab
        f = ttk.LabelFrame(devices, text="Devices")
        f.pack(fill=tk.BOTH, expand=True, **pad)
        top = ttk.Frame(f)
        top.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(top, text="Name").pack(side=tk.LEFT)
        self.dev_name = tk.StringVar()
        e = ttk.Entry(top, textvariable=self.dev_name, width=14)
        e.pack(side=tk.LEFT, padx=4)
        e.bind("<Return>", lambda ev: self._dev_add())
        ttk.Button(top, text="Add", width=6, command=self._dev_add).pack(side=tk.LEFT)
        ttk.Button(top, text="Rename", width=7, command=self._dev_rename).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Delete", width=6, command=self._dev_delete).pack(side=tk.LEFT)
        io = ttk.Frame(f)
        io.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(io, text="Save devices…", command=self.save_devices).pack(side=tk.LEFT)
        ttk.Button(io, text="Load devices…", command=self.load_devices).pack(side=tk.LEFT, padx=4)
        dw = ttk.Frame(f)
        dw.pack(fill=tk.BOTH, expand=True, padx=4)
        self.dev_tree = ttk.Treeview(dw, columns=("name", "plus", "minus"), show="headings",
                                     height=7, selectmode="browse")
        for key, head, w in (("name", "Device", 100), ("plus", "+ pad (centre)", 110), ("minus", "− pad (centre)", 120)):
            self.dev_tree.heading(key, text=head)
            self.dev_tree.column(key, width=w, anchor="w")
        ys = ttk.Scrollbar(dw, orient="vertical", command=self.dev_tree.yview)
        self.dev_tree.configure(yscrollcommand=ys.set)
        self.dev_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ys.pack(side=tk.RIGHT, fill=tk.Y)
        self.dev_tree.bind("<<TreeviewSelect>>", lambda e: self._on_dev_select())
        self.dev_tree.bind("<Double-1>", lambda e: self._zoom_to_device())
        ttk.Label(f, text="Double-click a device to zoom to it. Devices can share a pad\n"
                          "(e.g. a common ground): just pick the same pad again.",
                  foreground="#666", justify="left").pack(fill=tk.X, padx=4, pady=(2, 4))

        f = ttk.LabelFrame(devices, text="Set pads of the selected device")
        f.pack(fill=tk.X, **pad)
        self.mode_var = tk.StringVar(value="nav")
        for i, (m, txt) in enumerate((("nav", "Navigate"), ("plus", "Set +"),
                                      ("minus", "Set −"), ("region", "Region"))):
            ttk.Radiobutton(f, text=txt, value=m, variable=self.mode_var, command=self._set_mode,
                            style="Toolbutton").grid(row=0, column=i, padx=2, pady=4, sticky="we")
        ttk.Label(f, text="Pad size [µm]  (e.g. 80 or 80x100)").grid(
            row=1, column=0, columnspan=3, sticky="w", padx=4, pady=(4, 0))
        self.pad_var = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self.pad_var, width=10).grid(row=1, column=3, sticky="e", padx=4, pady=(4, 0))
        ttk.Label(f, text="Click inside a pad to snap to it (blank size = whole\n"
                          "polygon under the click), or drag a box by hand.\n"
                          "After + is set, the next click sets −.",
                  foreground="#666", justify="left").grid(row=2, column=0, columnspan=4, sticky="w", padx=4)
        self.region_var = tk.StringVar(value=self._region_text())
        ttk.Label(f, textvariable=self.region_var, foreground="#444").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=4, pady=4)
        ttk.Button(f, text="Clear region", command=self._clear_region).grid(row=3, column=3, sticky="e", padx=4, pady=4)

        # ===== Device result tab: key numbers large, details below
        self.res_dev_var = tk.StringVar(value="No device selected")
        ttk.Label(results, textvariable=self.res_dev_var, font=("Segoe UI", 12, "bold")).pack(
            fill=tk.X, padx=10, pady=(6, 0))
        f = ttk.LabelFrame(results, text="Resistance")
        f.pack(fill=tk.X, **pad)
        big, mid = ("Segoe UI", 15, "bold"), ("Segoe UI", 11)
        self.res_vars = {}

        def row(parent, r, key, label, font):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=2)
            v = tk.StringVar(value="–")
            ttk.Label(parent, textvariable=v, font=font).grid(row=r, column=1, sticky="e", padx=6, pady=2)
            parent.columnconfigure(1, weight=1)
            self.res_vars[key] = v

        row(f, 0, "R", "Total", big)
        row(f, 1, "R_h", "Heater", mid)
        row(f, 2, "R_r", "Routing", mid)
        row(f, 3, "I", "Current", mid)
        row(f, 4, "P", "Power", mid)

        self.j_frame = f = ttk.LabelFrame(results, text="Heater current density")
        f.pack(fill=tk.X, **pad)
        row(f, 0, "J_med", "Typical (median)", big)
        row(f, 1, "J_p99", "99th percentile", mid)
        row(f, 2, "J_max", "Peak (inner corners)", mid)
        row(f, 3, "J_r", "Routing peak", mid)

        f = ttk.LabelFrame(results, text="Map on layout")
        f.pack(fill=tk.X, **pad)
        self.ov_var = tk.StringVar(value="Layout")
        for i, o in enumerate(OVERLAYS):
            ttk.Radiobutton(f, text=o, value=o, variable=self.ov_var, command=self.show_overlay).grid(
                row=i // 2, column=i % 2, sticky="w", padx=6, pady=1)

        f = ttk.LabelFrame(results, text="Details")
        f.pack(fill=tk.BOTH, expand=True, **pad)
        self.res_text = tk.Text(f, height=6, width=44, font=("Consolas", 9), wrap="word",
                                relief="flat", background=self.root.cget("background"))
        self.res_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.res_text.configure(state="disabled")

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
        self.file_var.set(os.path.basename(path))
        self.devices = []
        self.region = None
        self._refresh_dev_tree()
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
        self.results = {}
        self.result = None
        self._redraw_all()
        self._rebuild_layer_list()
        self._refresh_table()
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
        self.dev_artists, self.region_artists = [], []
        self._draw_devices()
        self._draw_region()
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
            self._zoom_to((b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()))
        self.canvas.draw_idle()

    def _zoom_to(self, bb, margin=0.04):
        dx, dy = bb[2] - bb[0], bb[3] - bb[1]
        m = margin * max(dx, dy, 1e-3)
        self.ax.set_xlim(bb[0] - m, bb[2] + m)
        self.ax.set_ylim(bb[1] - m, bb[3] + m)
        self.canvas.draw_idle()

    def _draw_devices(self):
        """Draw every device's pads; shared pads are drawn once with all their labels."""
        for a in self.dev_artists:
            a.remove()
        self.dev_artists = []
        sel = self._cur_dev()
        pads = {}  # rounded box -> {"box", "labels": [...], "kind", "selected"}
        for d in self.devices:
            for kind in ("plus", "minus"):
                b = d[kind]
                if b is None:
                    continue
                key = tuple(round(v, 3) for v in b)
                p = pads.setdefault(key, {"box": b, "labels": [], "kind": kind, "selected": False})
                p["labels"].append(d["name"] + SIGN[kind])
                p["selected"] |= d is sel
        for p in pads.values():
            b, c = p["box"], COLORS[p["kind"]]
            r = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=True,
                          facecolor=mcolors.to_rgba(c, 0.18 if p["selected"] else 0.06), edgecolor=c,
                          lw=2.6 if p["selected"] else 1.2, zorder=60)
            self.ax.add_patch(r)
            label = ", ".join(p["labels"])
            if len(label) > 48:
                label = label[:45] + "…"
            t = self.ax.text(b[0], b[3], " " + label, color=c, va="bottom", ha="left", fontsize=8.5,
                             fontweight="bold" if p["selected"] else "normal", zorder=61, clip_on=True)
            self.dev_artists += [r, t]
        self.canvas.draw_idle()

    def _draw_region(self):
        for a in self.region_artists:
            a.remove()
        self.region_artists = []
        b = self.region
        if b is not None:
            r = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=COLORS["region"],
                          lw=1.5, ls="--", zorder=59)
            self.ax.add_patch(r)
            t = self.ax.text(b[0], b[3], " analysis region", color=COLORS["region"], va="bottom",
                             fontsize=8.5, zorder=61, clip_on=True)
            self.region_artists = [r, t]
        self.region_var.set(self._region_text())
        self.canvas.draw_idle()

    def _region_text(self):
        if self.region is None:
            return "Region: auto (metal net of each device)"
        b = self.region
        return f"Region: ({b[0]:.0f}, {b[1]:.0f}) – ({b[2]:.0f}, {b[3]:.0f}) µm"

    def _clear_region(self):
        self.region = None
        self._draw_region()

    def _set_mode(self):
        self.mode = self.mode_var.get()
        if self.mode in ("plus", "minus") and self._cur_dev() is None:
            self._dev_add()  # first pad without a device: create one
        if self.mode != "nav" and self.toolbar.mode:
            # leave matplotlib's own pan/zoom so our box drawing receives the clicks
            (self.toolbar.pan if "pan" in str(self.toolbar.mode) else self.toolbar.zoom)()
        if self.mode in ("plus", "minus"):
            d = self._cur_dev()
            self.status.set(f"{d['name']}: click inside the {SIGN[self.mode]} pad, or drag a box.")

    def _set_mode_to(self, m):
        self.mode_var.set(m)
        self._set_mode()

    # ------------------------------------------------------------------ devices
    def _cur_dev(self):
        sel = self.dev_tree.selection()
        if not sel:
            return None
        return next((d for d in self.devices if d["name"] == sel[0]), None)

    def _next_name(self):
        names = {d["name"] for d in self.devices}
        i = len(self.devices) + 1
        while f"dev{i}" in names:
            i += 1
        return f"dev{i}"

    def _dev_add(self):
        name = self.dev_name.get().strip() or self._next_name()
        if any(d["name"] == name for d in self.devices):
            messagebox.showerror("Device name", f"A device called '{name}' already exists."); return
        self.devices.append({"name": name, "plus": None, "minus": None})
        self.dev_name.set("")
        self._refresh_dev_tree(select=name)

    def _dev_rename(self):
        d = self._cur_dev()
        new = self.dev_name.get().strip()
        if d is None or not new:
            messagebox.showinfo("Rename", "Select a device and type the new name in the Name box."); return
        if new != d["name"] and any(x["name"] == new for x in self.devices):
            messagebox.showerror("Device name", f"A device called '{new}' already exists."); return
        if d["name"] in self.results:
            self.results[new] = self.results.pop(d["name"])
        d["name"] = new
        self.dev_name.set("")
        self._refresh_dev_tree(select=new)
        self._refresh_table()

    def _dev_delete(self):
        d = self._cur_dev()
        if d is None:
            return
        i = self.devices.index(d)
        self.devices.remove(d)
        self.results.pop(d["name"], None)
        nxt = self.devices[min(i, len(self.devices) - 1)]["name"] if self.devices else None
        self._refresh_dev_tree(select=nxt)
        self._refresh_table()

    def _refresh_dev_tree(self, select=None):
        if select is None:
            cur = self._cur_dev()
            select = cur["name"] if cur else None
        self.dev_tree.delete(*self.dev_tree.get_children())
        counts = {}
        for d in self.devices:
            for k in ("plus", "minus"):
                if d[k]:
                    key = tuple(round(v, 3) for v in d[k])
                    counts[key] = counts.get(key, 0) + 1

        def cell(b):
            if b is None:
                return "not set"
            s = f"({(b[0] + b[2]) / 2:.0f}, {(b[1] + b[3]) / 2:.0f})"
            n = counts.get(tuple(round(v, 3) for v in b), 0)
            return s + (f"  shared ×{n}" if n > 1 else "")

        for d in self.devices:
            self.dev_tree.insert("", tk.END, iid=d["name"], values=(d["name"], cell(d["plus"]), cell(d["minus"])))
        if select is not None and self.dev_tree.exists(select):
            self.dev_tree.selection_set(select)
            self.dev_tree.see(select)
        self._draw_devices()

    def _on_dev_select(self):
        self._draw_devices()
        d = self._cur_dev()
        if d and self.table.exists(d["name"]) and self.table.selection() != (d["name"],):
            self.table.selection_set(d["name"])

    def _zoom_to_device(self):
        d = self._cur_dev()
        boxes = [b for b in (d["plus"], d["minus"]) if b] if d else []
        if not boxes:
            return
        if d["name"] in self.results and not isinstance(self.results[d["name"]], str):
            g = self.results[d["name"]].grid
            boxes.append((g.extent[0], g.extent[2], g.extent[1], g.extent[3]))
        B = np.array(boxes)
        self._zoom_to((B[:, 0].min(), B[:, 1].min(), B[:, 2].max(), B[:, 3].max()), margin=0.1)

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
            rect = Rectangle((ev.xdata, ev.ydata), 0, 0, fill=False, ec=COLORS[self.mode],
                             lw=1.5, ls=":", zorder=70)
            self.ax.add_patch(rect)
            self.drag = ("box", ev.xdata, ev.ydata, rect, ev.x, ev.y)

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
            _, x0, y0, rect = self.drag[:4]
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
        _, x0, y0, rect, px, py = d
        rect.remove()
        x1 = ev.xdata if ev.xdata is not None else x0
        y1 = ev.ydata if ev.ydata is not None else y0
        box = None
        if abs(ev.x - px) < 5 and abs(ev.y - py) < 5:
            # a click, not a drag: snap to the pad under the cursor
            if self.mode in ("plus", "minus"):
                try:
                    box = self._find_pad(x0, y0)
                except ValueError as e:
                    messagebox.showerror("Pad size", str(e))
                    return
                if box is None:
                    self.status.set(self._pad_miss)
                    self.canvas.draw_idle()
                    return  # stay in this mode so the user can click again
        elif abs(x1 - x0) > 0 and abs(y1 - y0) > 0:
            box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if box is None:
            self.canvas.draw_idle()
            return
        if self.mode == "region":
            self.region = box
            self._draw_region()
            self._set_mode_to("nav")
            return
        dev = self._cur_dev()
        if dev is None:
            self._dev_add(); dev = self._cur_dev()
        dev[self.mode] = self._same_as_existing(box)
        self.results.pop(dev["name"], None)
        self._refresh_dev_tree(select=dev["name"])
        self._refresh_table()
        if self.mode == "plus" and dev["minus"] is None:
            self._set_mode_to("minus")  # natural next step
        else:
            self._set_mode_to("nav")
            self.status.set(f"{dev['name']}: pads set.")

    def _same_as_existing(self, box):
        """Reuse an existing pad box that is essentially the same pad (shared pads must match exactly)."""
        def iou(a, b):
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / union if union > 0 else 0.0
        for d in self.devices:
            for k in ("plus", "minus"):
                if d[k] is not None and iou(d[k], box) > 0.9:
                    return d[k]
        return box

    # ------------------------------------------------------------------ pad picking
    def _pad_size(self):
        s = self.pad_var.get().strip().lower().replace("×", "x").replace("*", "x").replace(",", "x")
        if not s:
            return None
        try:
            parts = [float(p) for p in s.split("x") if p.strip()]
        except ValueError:
            raise ValueError(f"Pad size '{self.pad_var.get()}' is not a number. Use e.g. 80 or 80x100.")
        if len(parts) == 1:
            parts *= 2
        if len(parts) != 2 or min(parts) <= 0:
            raise ValueError("Pad size must be one or two positive numbers, e.g. 80 or 80x100.")
        return tuple(parts)

    def _pad_layers(self):
        keys = self._role_keys("Routing") or self._role_keys("Heater")
        return keys or [k for k in self.geoms if self.vis_vars[k].get()]

    def _find_pad(self, x, y):
        """Terminal box for a click at (x, y), or None (reason in self._pad_miss)."""
        from matplotlib.path import Path
        keys = self._pad_layers()
        size = self._pad_size()
        if size is None:
            # the "fattest" polygon under the click (largest short side): the pad,
            # not a narrow trace that runs into it
            best = None
            for k in keys:
                g = self.geoms[k]
                for i in g.in_box((x, y, x, y)):
                    if Path(g.polys[i]).contains_point((x, y)):
                        b = g.bb[i]
                        fat = min(b[2] - b[0], b[3] - b[1])
                        if best is None or fat > best[0]:
                            best = (fat, tuple(float(v) for v in b))
            self._pad_miss = "No metal under the click (on Routing layers, or Heater if none). Try again."
            return best[1] if best else None

        # Find the w×h square that is fully covered by metal and contains the click.
        w, h = size
        win = (x - 1.5 * w, y - 1.5 * h, x + 1.5 * w, y + 1.5 * h)
        polys = [p for k in keys for p in self.geoms[k].select(self.geoms[k].in_box(win))]
        res = min(w, h) / 60
        g = hc.Grid.from_bbox(win, res)
        m = hc.rasterize(polys, g)
        rc, cc = int((y - g.y0) / res), int((x - g.x0) / res)
        if not polys or not m[rc, cc]:
            self._pad_miss = "No metal under the click (on Routing layers, or Heater if none). Try again."
            return None
        nw, nh = int(round(w / res)), int(round(h / res))
        S = np.zeros((g.ny + 1, g.nx + 1))
        S[1:, 1:] = m.cumsum(0).cumsum(1)
        # metal count in every nh×nw window, indexed by its lower-left cell
        cnt = S[nh:, nw:] - S[:-nh, nw:] - S[nh:, :-nw] + S[:-nh, :-nw]
        full = cnt >= 0.99 * nw * nh  # small tolerance for edge pixels
        r_lo, c_lo = max(rc - nh + 1, 0), max(cc - nw + 1, 0)
        sub = full[r_lo:rc + 1, c_lo:cc + 1]
        if not sub.any():
            self._pad_miss = (f"No {w:g}×{h:g} µm square of metal fits around the click. "
                              "Check the pad size, or leave it blank to use the whole polygon.")
            return None
        rr, cc2 = np.nonzero(sub)
        # centre of all valid placements = centre of the pad
        r0, c0 = r_lo + rr.mean(), c_lo + cc2.mean()
        bx, by = g.x0 + c0 * res, g.y0 + r0 * res
        return (float(bx), float(by), float(bx + w), float(by + h))

    def _probe(self, x, y):
        s = f"x = {x:.3f} µm   y = {y:.3f} µm"
        r = self.result
        if r is None:
            return s
        cell = r.grid.cell_at(x, y)
        if cell is not None:
            rr, c = cell
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

    def _region_for(self, dev):
        """Analysis region: the drawn box, else the bbox of the metal net touching the device's pads."""
        if self.region:
            return self.region
        keys = self._role_keys("Heater") + self._role_keys("Routing")
        bbs = [self.geoms[k].bb for k in keys if self.geoms[k].n]
        if not bbs:
            return None
        B = np.concatenate(bbs)
        seeds = [b for b in (dev["plus"], dev["minus"]) if b]

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
        A = np.concatenate([B[seen]] + [np.array([b]) for b in seeds])
        m = 2.0
        return (A[:, 0].min() - m, A[:, 1].min() - m, A[:, 2].max() + m, A[:, 3].max() + m)

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
                raise ValueError("Assign the role 'Heater' to at least one layer (Setup tab).")
            if not self.devices:
                raise ValueError("Add at least one device and set its + and − pads (Devices tab).")
            rs_h = self._float("rs_h", "Heater sheet resistance")
            rs_r = self._float("rs_r", "Routing sheet resistance")
            t_h = self._float("t_h", "Heater thickness") / 1000.0
            t_r = self._float("t_r", "Routing thickness", allow_empty=True)
            t_r = t_r / 1000.0 if t_r else None
            V = self._float("V", "Voltage")
            auto = lambda k: self.params[k].get().strip().lower() in ("", "auto")
            h_fixed = None if auto("h") else self._float("h", "Grid cell at heater")
            hmax_fixed = None if auto("hmax") else self._float("hmax", "Max grid cell")
        except ValueError as e:
            messagebox.showerror("Cannot run", str(e)); return

        # Prepare one job per device; devices that cannot run get an error row.
        self.results = {}
        jobs = []
        for d in self.devices:
            if not d["plus"] or not d["minus"]:
                self.results[d["name"]] = "Pads not set: select the device and use Set + / Set −."
                continue
            region = self._region_for(d)
            heater = self._role_polys("Heater", region)
            if not heater:
                self.results[d["name"]] = "No heater polygons inside its analysis region."
                continue
            routing = self._role_polys("Routing", region)
            via = self._role_polys("Via", region)
            hf, hcz = hc.auto_grid_sizes(heater, routing, region)
            hf = h_fixed or hf
            hcz = max(hmax_fixed or hcz, hf)
            g = hc.graded_grid(region, heater, routing + via, hf, hcz)
            if g.nx * g.ny > 60_000_000:
                self.results[d["name"]] = (f"Grid would be {g.nx}×{g.ny} cells: use a larger grid cell "
                                           "or draw an analysis region.")
                continue
            jobs.append((d["name"], d["plus"], d["minus"], g, heater, routing, via))
        self._refresh_table()
        if not jobs:
            messagebox.showerror("Cannot run", "No device is ready to run. See the Note column in the table.")
            return

        self.busy = True
        self.run_btn.state(["disabled"])

        def work():
            for k, (name, plus, minus, g, heater, routing, via) in enumerate(jobs):
                say = lambda s, name=name, k=k: self.q.put(("progress", f"{name} ({k + 1}/{len(jobs)}): {s}"))
                try:
                    say(f"rasterising {g.nx}×{g.ny}…")
                    hm = hc.rasterize(heater, g)
                    rm = hc.rasterize(routing, g) if routing else None
                    vm = hc.rasterize(via, g) if via else None
                    res = hc.solve(g, hm, rm, vm, rs_h, rs_r, plus, minus, V, t_h, t_r, progress=say)
                    res.stats["t_h_nm"] = t_h * 1e3
                    res.stats["t_r_nm"] = t_r * 1e3 if t_r else None
                    res.stats["via_used"] = bool(via)
                    out = res
                except hc.SolveError as e:
                    out = str(e).replace("\n", " ")
                except MemoryError:
                    out = "Out of memory: use a larger grid cell or a smaller analysis region."
                except Exception as e:
                    import traceback; traceback.print_exc()
                    out = f"{type(e).__name__}: {e}"
                self.q.put(("dev_done", (name, out)))
            self.q.put(("all_done", len(jobs)))

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
                elif kind == "dev_done":
                    name, out = payload
                    self.results[name] = out
                    self._refresh_table()
                elif kind == "all_done":
                    self.busy = False
                    self.run_btn.state(["!disabled"])
                    ok = [d["name"] for d in self.devices if not isinstance(self.results.get(d["name"]), (str, type(None)))]
                    n_err = len(self.devices) - len(ok)
                    self.prog_var.set(f"Done: {len(ok)} solved" + (f", {n_err} with errors" if n_err else "") + ".")
                    cur = self._cur_dev()
                    pick = cur["name"] if cur and cur["name"] in ok else (ok[0] if ok else None)
                    if pick:
                        self.table.selection_set(pick)
                        self.tabs.select(2)
        except queue.Empty:
            pass
        self.root.after(30, self._poll)

    # ------------------------------------------------------------------ results
    def _row_values(self, name):
        r = self.results.get(name)
        if r is None:
            return (name,) + ("",) * (len(TABLE_COLS) - 2) + ("not run yet",), ()
        if isinstance(r, str):
            return (name,) + ("",) * (len(TABLE_COLS) - 2) + (r,), ("error",)
        s = r.stats
        note = []
        w = s.get("heater_width_um")
        if w and w / r.grid.h < 6:
            note.append(f"heater only ~{w / r.grid.h:.0f} cells wide: refine grid")
        vals = (name, _fmt(r.R), _fmt(r.R_heater), _fmt(r.R_routing), _fmt(r.I * 1e3), _fmt(r.P * 1e3),
                _fmt(s["J_heater_median"]), _fmt(s["J_heater_p99"]), _fmt(s["J_heater_max"]),
                _fmt(s.get("J_routing_max")), "; ".join(note) or f"grid {r.grid.h:.3g}–{r.grid.h_max:.3g} µm")
        return vals, ()

    def _refresh_table(self):
        sel = self.table.selection()
        self.table.delete(*self.table.get_children())
        for d in self.devices:
            vals, tags = self._row_values(d["name"])
            self.table.insert("", tk.END, iid=d["name"], values=vals, tags=tags)
        n = sum(1 for d in self.devices if not isinstance(self.results.get(d["name"]), (str, type(None))))
        self.table_info.set(f"{n} of {len(self.devices)} devices solved" if self.devices
                            else "Add devices in the Devices tab, then run.")
        if sel and self.table.exists(sel[0]):
            self.table.selection_set(sel[0])

    def _on_table_select(self, _=None):
        sel = self.table.selection()
        if not sel:
            return
        name = sel[0]
        r = self.results.get(name)
        if self.dev_tree.exists(name) and self.dev_tree.selection() != (name,):
            self.dev_tree.selection_set(name)
        self.result = r if r is not None and not isinstance(r, str) else None
        self._show_results(self.result, name, r if isinstance(r, str) else None)
        if self.result is not None and self.ov_var.get() == "Layout":
            self.ov_var.set("|J| heater")
        self.show_overlay()

    def _table_rows(self):
        head = [c[1] for c in TABLE_COLS] + ["+ pad box [µm]", "− pad box [µm]"]
        rows = []
        for d in self.devices:
            vals, _ = self._row_values(d["name"])
            boxes = ["" if d[k] is None else "(%.2f, %.2f, %.2f, %.2f)" % d[k] for k in ("plus", "minus")]
            rows.append(list(vals) + boxes)
        return head, rows

    def copy_table(self):
        head, rows = self._table_rows()
        text = "\n".join("\t".join(map(str, r)) for r in [head] + rows)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.table_info.set(f"Copied {len(rows)} rows: paste into Excel.")

    def export_csv(self):
        if not self.devices:
            messagebox.showinfo("Export", "There are no devices to export."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="heater_results.csv")
        if not path:
            return
        head, rows = self._table_rows()
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["# file", self.file_var.get(), "cell", self.cell_var.get()])
            w.writerow(["# Rs heater [ohm/sq]", self.params["rs_h"].get(), "Rs routing [ohm/sq]",
                        self.params["rs_r"].get(), "t heater [nm]", self.params["t_h"].get(),
                        "t routing [nm]", self.params["t_r"].get(), "V [V]", self.params["V"].get()])
            w.writerow(head)
            w.writerows(rows)
        self.table_info.set(f"Saved {os.path.basename(path)}")

    def save_devices(self):
        if not self.devices:
            messagebox.showinfo("Save devices", "There are no devices to save."); return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Device set", "*.json")],
                                            initialfile=os.path.splitext(self.file_var.get())[0] + "_devices.json")
        if not path:
            return
        data = {"gds_file": self.file_var.get(), "cell": self.cell_var.get(), "pad_size": self.pad_var.get(),
                "region": self.region,
                "layer_roles": {f"{k[0]}/{k[1]}": v.get() for k, v in self.role_vars.items() if v.get() != "-"},
                "parameters": {k: v.get() for k, v in self.params.items()},
                "devices": [{"name": d["name"], "plus": d["plus"], "minus": d["minus"]} for d in self.devices]}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        self.status.set(f"Saved {len(self.devices)} devices to {os.path.basename(path)}")

    def load_devices(self):
        path = filedialog.askopenfilename(filetypes=[("Device set", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            devs = [{"name": str(d["name"]),
                     "plus": tuple(d["plus"]) if d.get("plus") else None,
                     "minus": tuple(d["minus"]) if d.get("minus") else None} for d in data["devices"]]
        except Exception as e:
            messagebox.showerror("Load devices", f"Could not read {os.path.basename(path)}: {e}"); return
        self.devices = devs
        self.region = tuple(data["region"]) if data.get("region") else None
        self.pad_var.set(data.get("pad_size", ""))
        for k, v in data.get("parameters", {}).items():
            if k in self.params:
                self.params[k].set(v)
        for spec, role in data.get("layer_roles", {}).items():
            key = tuple(int(x) for x in spec.split("/"))
            if key in self.role_vars and role in ROLES:
                self.role_vars[key].set(role)
        self._schedule_render(force=True)
        self.results = {}
        self._refresh_dev_tree(select=devs[0]["name"] if devs else None)
        self._draw_region()
        self._refresh_table()
        self.status.set(f"Loaded {len(devs)} devices from {os.path.basename(path)}")

    def _show_results(self, r, name=None, error=None):
        self.res_dev_var.set(name or "No device selected")
        self.res_text.configure(state="normal")
        self.res_text.delete("1.0", tk.END)
        for v in self.res_vars.values():
            v.set("–")
        self.j_frame.configure(text="Heater current density")
        if error:
            self.res_text.insert("1.0", error)
        if r is not None:
            s = r.stats
            rv = self.res_vars
            rv["R"].set(f"{r.R:,.4g} Ω")
            rv["R_h"].set(f"{r.R_heater:,.4g} Ω  ({100 * r.R_heater / r.R:.1f} %)")
            rv["R_r"].set(f"{r.R_routing:,.4g} Ω  ({100 * r.R_routing / r.R:.1f} %)")
            rv["I"].set(f"{r.I * 1e3:.4g} mA  @ {r.voltage:g} V")
            rv["P"].set(f"{r.P * 1e3:.4g} mW")
            self.j_frame.configure(text=f"Heater current density  (t = {s['t_h_nm']:g} nm)")
            rv["J_med"].set(f"{s['J_heater_median']:.4g} mA/µm²")
            rv["J_p99"].set(f"{s['J_heater_p99']:.4g} mA/µm²")
            rv["J_max"].set(f"{s['J_heater_max']:.4g} mA/µm²")
            if "J_routing_max" in s:
                rv["J_r"].set(f"{s['J_routing_max']:.4g} mA/µm²  (t = {s['t_r_nm']:g} nm)")
            w = s.get("heater_width_um")
            g = r.grid
            lines = [f"Grid {g.nx}×{g.ny} = {g.nx * g.ny:,} cells, {g.h:.3g} µm at heater → {g.h_max:.3g} µm max, "
                     f"{r.n_nodes:,} unknowns",
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
            norm = mcolors.Normalize(vmin, max(vmax, vmin + 1e-30))
            cmap_obj = matplotlib.colormaps[cmap]
            rgba = cmap_obj(norm(np.nan_to_num(data)))
            rgba[~np.isfinite(data), 3] = 0.0  # no metal: transparent
            # the grid is graded (non-uniform), so use NonUniformImage instead of imshow
            self.overlay_im = NonUniformImage(self.ax, interpolation="nearest", extent=r.grid.extent, zorder=40)
            self.overlay_im.set_data(r.grid.xc, r.grid.yc, rgba)
            self.ax.add_image(self.overlay_im)
            self.cax.set_visible(True)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            cb = self.fig.colorbar(sm, cax=self.cax, extend="max" if which != "Potential" else "neither")
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
