"""Write example_heater.gds: TiN meander heater (10/0), Al routing + pads (20/0), vias (30/0)."""
import gdstk

HEATER, ROUTING, VIA = (10, 0), (20, 0), (30, 0)


def build(path="example_heater.gds"):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)

    pad = lib.new_cell("PAD_80")
    pad.add(gdstk.rectangle((-40, -40), (40, 40), layer=ROUTING[0], datatype=ROUTING[1]))

    top = lib.new_cell("HEATER_TOP")

    # Meander: 7 legs of 120 um, 2 um wide, 8 um pitch (odd count so it ends on the right)
    w, leg, pitch, n = 2.0, 120.0, 8.0, 7
    pts = [(0, 0)]
    for i in range(n):
        x = leg if i % 2 == 0 else 0
        y = i * pitch
        pts.append((x, y))
        if i < n - 1:
            pts.append((x, y + pitch))
    top.add(gdstk.FlexPath(pts, w, layer=HEATER[0], datatype=HEATER[1]))
    end = pts[-1]

    # Heater landing tabs under the routing (5x5 um) at both meander ends
    for (x, y) in (pts[0], end):
        top.add(gdstk.rectangle((x - 2.5, y - 2.5), (x + 2.5, y + 2.5), layer=HEATER[0], datatype=HEATER[1]))
        top.add(gdstk.rectangle((x - 1.5, y - 1.5), (x + 1.5, y + 1.5), layer=VIA[0], datatype=VIA[1]))

    # Routing: 10 um wide traces from the tabs out to the pads
    left_pad, right_pad = (-120, 0), (240, end[1])
    top.add(gdstk.FlexPath([(pts[0][0] + 3, 0), (left_pad[0], 0)], 10, layer=ROUTING[0], datatype=ROUTING[1]))
    top.add(gdstk.FlexPath([(end[0] - 3, end[1]), (right_pad[0], end[1])], 10, layer=ROUTING[0], datatype=ROUTING[1]))
    top.add(gdstk.Reference(pad, left_pad))
    top.add(gdstk.Reference(pad, right_pad))
    top.add(gdstk.Label("+", left_pad, layer=63))
    top.add(gdstk.Label("-", right_pad, layer=63))

    lib.write_gds(path)
    return path


def build_multi(path="example_multi_heater.gds"):
    """Three MZI heaters (3, 5 and 7 legs) with their own signal pads and one shared ground pad."""
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    pad = lib.new_cell("PAD_80")
    pad.add(gdstk.rectangle((-40, -40), (40, 40), layer=ROUTING[0], datatype=ROUTING[1]))
    top = lib.new_cell("MZI_ARRAY")
    w, leg, pitch = 2.0, 120.0, 8.0
    gnd = (700, 300)
    top.add(gdstk.Reference(pad, gnd))
    for k, n in enumerate((3, 5, 7)):
        y0 = k * 300
        pts = [(0, y0)]
        for i in range(n):
            x = leg if i % 2 == 0 else 0
            pts.append((x, y0 + i * pitch))
            if i < n - 1:
                pts.append((x, y0 + (i + 1) * pitch))
        top.add(gdstk.FlexPath(pts, w, layer=HEATER[0], datatype=HEATER[1]))
        start, end = pts[0], pts[-1]
        for (x, y) in (start, end):
            top.add(gdstk.rectangle((x - 2.5, y - 2.5), (x + 2.5, y + 2.5), layer=HEATER[0], datatype=HEATER[1]))
            top.add(gdstk.rectangle((x - 1.5, y - 1.5), (x + 1.5, y + 1.5), layer=VIA[0], datatype=VIA[1]))
        sig = (-400, y0)
        top.add(gdstk.Reference(pad, sig))
        top.add(gdstk.FlexPath([(start[0] + 3, y0), sig], 10, layer=ROUTING[0], datatype=ROUTING[1]))
        # to the shared ground bus at x = 500, then to the ground pad
        top.add(gdstk.FlexPath([(end[0] - 3, end[1]), (500, end[1]), (500, gnd[1]), gnd], 10,
                               layer=ROUTING[0], datatype=ROUTING[1]))
        top.add(gdstk.Label(f"MZI_{k + 1}", sig, layer=63))
    lib.write_gds(path)
    return path


if __name__ == "__main__":
    print("wrote", build())
    print("wrote", build_multi())
