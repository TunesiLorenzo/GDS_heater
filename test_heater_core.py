"""Validation of heater_core against cases with known answers. Run: python test_heater_core.py"""
import numpy as np

import heater_core as hc


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def run(heater, routing, plus, minus, h, rs_h=10.0, rs_r=0.05, V=1.0, t=0.1, via=None):
    allp = heater + routing
    bb = hc.polys_bbox(allp)
    g = hc.Grid.from_bbox((bb[0] - h, bb[1] - h, bb[2] + h, bb[3] + h), h)
    hm = hc.rasterize(heater, g)
    rm = hc.rasterize(routing, g) if routing else None
    vm = hc.rasterize(via, g) if via else None
    return hc.solve(g, hm, rm, vm, rs_h, rs_r, plus, minus, V, t)


def check(name, got, want, rel):
    ok = abs(got - want) <= rel * abs(want)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got:.4g}, expected {want:.4g} (±{rel:.1%})")
    return ok


def main():
    ok = True

    # 1) Straight bar 100 um long between terminals, 5 um wide, 10 ohm/sq -> 200 ohm
    r = run([rect(0, 0, 120, 5)], [], (-1, -1, 10, 6), (110, -1, 121, 6), h=0.25)
    ok &= check("bar resistance", r.R, 200.0, 0.005)
    # J = I / (w t) = 5 mA / (5 um * 0.1 um) = 10 mA/um^2
    ok &= check("bar current density (median)", r.stats["J_heater_median"], 10.0, 0.01)
    ok &= check("bar power split", r.R_heater + r.R_routing, r.R, 1e-6)

    # 2) Rotated geometry (non-Manhattan): bar at 30 degrees
    th = np.radians(30)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    bar = rect(0, 0, 120, 5) @ Rm.T
    end_p = np.array([[0, 2.5]]) @ Rm.T
    end_m = np.array([[120, 2.5]]) @ Rm.T
    # terminals: boxes around the first/last 10 um (approximate; compare to 200 * (L_eff / 100))
    r2 = run([bar], [], (end_p[0, 0] - 12, end_p[0, 1] - 8, end_p[0, 0] + 3, end_p[0, 1] + 8),
             (end_m[0, 0] - 3, end_m[0, 1] - 8, end_m[0, 0] + 12, end_m[0, 1] + 8), h=0.2)
    print(f"[info] 30-degree bar R = {r2.R:.1f} ohm (box-terminal ends make L_eff ~ 110 um -> ~220 ohm)")

    # 3) L-bend: two 5 um wide arms, 50 um from terminal to the corner square each.
    #    Known: a square corner counts as ~0.56 squares -> 10 + 10 + 0.56 = 20.56 sq
    w = 5.0
    L = [rect(0, 0, 55, 5), rect(50, 0, 55, 55)]
    r3 = run(L, [], (-1, -1, 0.1, 6), (49, 54.9, 56, 56), h=0.125)
    ok &= check("L-bend squares", r3.R / 10.0, 20.56, 0.01)

    # 4) Routing in series with heater (overlap contact), routing nearly ideal
    heater = [rect(40, 0, 140, 5)]
    routing = [rect(0, -2.5, 45, 7.5)]
    r4 = run(heater, routing, (-1, -3, 1, 8), (139, -1, 141, 6), h=0.25, rs_r=1e-4)
    # heater from x=45 (routing edge) to ~139.1: 94.1 um / 5 um = 18.8 sq * 10 ohm = 188 ohm + spreading
    ok &= check("routing+heater series", r4.R, 188.5, 0.01)
    small = r4.R_routing < 0.1
    print(f"[{'PASS' if small else 'FAIL'}] routing share small: {r4.R_routing:.4f} ohm")
    ok &= small

    # 5) Via layer gating: no via -> must refuse (not connected)
    try:
        run(heater, routing, (-1, -3, 1, 8), (139, -1, 141, 6), h=0.5, via=[rect(-50, -50, -40, -40)])
        print("[FAIL] via gating: expected a 'not connected' error"); ok = False
    except hc.SolveError:
        print("[PASS] via gating: disconnected without a via")

    # 6) Example GDS end-to-end
    import make_example_gds
    path = make_example_gds.build("example_heater.gds")
    lib, tops = hc.load_gds(path)
    layers = hc.flatten_cell(tops[0])
    print("[info] example layers:", {k: len(v) for k, v in layers.items()})
    heater, routing, via = layers[(10, 0)], layers[(20, 0)], layers[(30, 0)]
    region = hc.polys_bbox(heater + routing)
    h = 0.25
    g = hc.Grid.from_bbox(region, h)
    res = hc.solve(g, hc.rasterize(heater, g), hc.rasterize(routing, g), hc.rasterize(via, g),
                   10.0, 0.05, (-160, -40, -80, 40), (200, 8, 280, 88), 1.0, 0.1, 0.5)
    # squares estimate: 7 legs * 120/2 + 6 U-turns (2 corners each ~0.56 + 3 sq straight) + tabs
    print(f"[info] example: R={res.R:.1f} ohm (heater {res.R_heater:.1f}, routing {res.R_routing:.2f}), "
          f"I={res.I*1e3:.3f} mA, J median={res.stats['J_heater_median']:.2f} mA/um^2, "
          f"max={res.stats['J_heater_max']:.2f}, nodes={res.n_nodes:,}")
    ok &= check("example J median = I/(w t)", res.stats["J_heater_median"], res.I * 1e3 / (2 * 0.1), 0.03)

    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
