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


if __name__ == "__main__":
    print("wrote", build())
