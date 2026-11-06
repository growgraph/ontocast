"""Draw the home page figure into ``docs/index.md``.

Two sheets in perspective: the ontology (classes) above the facts. Documents
pass a render-and-critique loop into both; a dashed line ties each fact to its
class. The facts graph hints at Orion: real star positions, the belt as one
node, dot size by brightness. Each class types two facts far apart, and class
spots are searched so the dashed lines rarely cross or touch another fact.

Colours are CSS classes (``.oc-layers__*`` in ``docs/stylesheets/extra.css``),
so the figure follows the light and dark schemes.

Run from the repository root::

    uv run python docs/_build/hero_figure.py          # rewrite the figure
    uv run python docs/_build/hero_figure.py --check  # exit 1 if stale
"""

import itertools
import math
import re
import sys
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "index.md"

# ---------------------------------------------------------------- scene
W, D = 230.0, 120.0  # sheet width (x) and depth (z: 0 back .. D front)
Y_TOP, Y_BOT = 165.0, 0.0  # heights of the ontology and facts sheets
CAM = (757.0, 1280.0, 2036.0)  # far, about 30 degrees up, turned to the side
TGT = (115.0, 80.0, 60.0)
FOCAL = 900.0
BOX_X, BOX_W, BOX_Y = 206.0, 262.0, 26.0  # where the sheets land in the SVG

# name: (right ascension deg, declination deg, visual magnitude)
STARS = {
    "Meissa": (83.78, 9.93, 3.33),
    "Betelgeuse": (88.79, 7.41, 0.50),
    "Bellatrix": (81.28, 6.35, 1.64),
    "Belt": (84.05, -1.20, 1.0),
    "Saiph": (86.94, -9.67, 2.07),
    "Rigel": (78.63, -8.20, 0.13),
}
LINKS = [
    ("Meissa", "Betelgeuse"),
    ("Meissa", "Bellatrix"),
    ("Betelgeuse", "Belt"),
    ("Bellatrix", "Belt"),
    ("Belt", "Saiph"),
    ("Belt", "Rigel"),
    ("Betelgeuse", "Bellatrix"),
    ("Rigel", "Saiph"),
]
MARGIN_U, MARGIN_V = 14.0, 12.0
STRETCH_MAX = 1.3  # a hint, not a chart: the fit may stretch one axis this much
EXTRA_CLASSES = 1  # classes typing no fact (a parent)


# ---------------------------------------------------------------- projection
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    n = math.sqrt(_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


FWD = _norm(_sub(TGT, CAM))
RIGHT = _norm(_cross3(FWD, (0.0, 1.0, 0.0)))
UP = _cross3(RIGHT, FWD)


def _raw(p):
    d = _sub(p, CAM)
    zc = _dot(d, FWD)
    return (FOCAL * _dot(d, RIGHT) / zc, -FOCAL * _dot(d, UP) / zc)


_CORNERS = [_raw((x, y, z)) for y in (Y_TOP, Y_BOT) for x in (0, W) for z in (0, D)]
_MINX = min(p[0] for p in _CORNERS)
_MAXX = max(p[0] for p in _CORNERS)
_MINY = min(p[1] for p in _CORNERS)
_SCALE = BOX_W / (_MAXX - _MINX)


def P(u, v, y):
    """Sheet point (u across, v depth from the back) at height y, in SVG units."""
    rx, ry = _raw((u, y, v))
    return (BOX_X + (rx - _MINX) * _SCALE, BOX_Y + (ry - _MINY) * _SCALE)


def f(p):
    return f"{p[0]:.1f},{p[1]:.1f}"


def quad(y, u0=0.0, v0=0.0, u1=W, v1=D):
    a, b, c, d = P(u0, v0, y), P(u1, v0, y), P(u1, v1, y), P(u0, v1, y)
    return f"M{f(a)} L{f(b)} L{f(c)} L{f(d)} Z"


def disc(u, v, y, r, n=20):
    ps = [
        P(
            u + r * math.cos(2 * math.pi * k / n),
            v + r * math.sin(2 * math.pi * k / n),
            y,
        )
        for k in range(n)
    ]
    return "M" + " L".join(f(p) for p in ps) + " Z"


def slab(y, t=4.0):
    """The sheet's visible thickness: its outline, shifted down."""
    a, b, c, d = P(0, 0, y), P(W, 0, y), P(W, D, y), P(0, D, y)
    return (
        f"M{a[0]:.1f},{a[1] + t:.1f} L{b[0]:.1f},{b[1] + t:.1f} "
        f"L{c[0]:.1f},{c[1] + t:.1f} L{d[0]:.1f},{d[1] + t:.1f} Z"
    )


def seg_cross(p1, p2, p3, p4):
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    return (o(p1, p2, p3) * o(p1, p2, p4) < 0) and (o(p3, p4, p1) * o(p3, p4, p2) < 0)


def seg_dist(p, a, b):
    ax, ay = b[0] - a[0], b[1] - a[1]
    t = max(
        0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / (ax * ax + ay * ay))
    )
    return math.hypot(p[0] - (a[0] + t * ax), p[1] - (a[1] + t * ay))


# ---------------------------------------------------------------- layout
def lay_out():
    """Sky to sheet: the best-filling turn, with a mild stretch."""
    ra0 = sum(s[0] for s in STARS.values()) / len(STARS)
    dec0 = sum(s[1] for s in STARS.values()) / len(STARS)
    cosd = math.cos(math.radians(dec0))
    base = {k: (-(dec - dec0), (ra - ra0) * cosd) for k, (ra, dec, _) in STARS.items()}
    best = None
    for phi in range(0, 180, 3):
        c, s = math.cos(math.radians(phi)), math.sin(math.radians(phi))
        rot = {k: (x * c - y * s, x * s + y * c) for k, (x, y) in base.items()}
        xs, ys = [p[0] for p in rot.values()], [p[1] for p in rot.values()]
        su = (W - 2 * MARGIN_U) / (max(xs) - min(xs))
        sv = (D - 2 * MARGIN_V) / (max(ys) - min(ys))
        if best is None or min(su, sv) > best[0]:
            best = (min(su, sv), rot, su, sv, xs, ys)
    uni, rot, su, sv, xs, ys = best
    su, sv = min(su, uni * STRETCH_MAX), min(sv, uni * STRETCH_MAX)
    ou = (W - su * (max(xs) - min(xs))) / 2 - su * min(xs)
    ov = (D - sv * (max(ys) - min(ys))) / 2 - sv * min(ys)
    return {k: (ou + su * x, ov + sv * y) for k, (x, y) in rot.items()}


def radius(mag):
    """Brighter star, bigger dot."""
    return max(5.4, min(8.6, 5.4 + 1.1 * (2.9 - mag)))


def _pairings(items):
    if not items:
        yield []
        return
    a = items[0]
    for i in range(1, len(items)):
        for rest in _pairings(items[1:i] + items[i + 1 :]):
            yield [(a, items[i])] + rest


def place_classes(uv):
    """Pair the facts into classes and place each class on the ontology sheet.

    Prefers pairs far apart and not linked, classes spread over the sheet, and
    dashed lines that neither cross nor pass through another fact.
    """
    pts = {k: P(*uv[k], Y_BOT) for k in uv}
    rad = {k: radius(STARS[k][2]) for k in uv}
    linked = {frozenset(link) for link in LINKS}
    best = None
    for pairs in _pairings(list(STARS)):
        cands = []
        for a, b in pairs:
            mu = (uv[a][0] + uv[b][0]) / 2
            spots = []
            for cu in range(24, int(W) - 23, 12):
                for cv in range(18, int(D) - 17, 10):
                    cp = P(cu, cv, Y_TOP)
                    touch = sum(
                        1
                        for k in (a, b)
                        for o in uv
                        if o != k and seg_dist(pts[o], pts[k], cp) < rad[o] + 3
                    )
                    spots.append(
                        (-80 * touch - 0.5 * abs(cu - mu), (float(cu), float(cv)), cp)
                    )
            spots.sort(key=lambda s: s[0], reverse=True)
            cands.append(spots[:14])
        wide = min(math.dist(pts[a], pts[b]) for a, b in pairs)
        unlinked = sum(1 for a, b in pairs if frozenset((a, b)) not in linked)
        for combo in itertools.product(*cands):
            ps = [c[1] for c in combo]
            if any(
                abs(p[0] - q[0]) < 52 and abs(p[1] - q[1]) < 36
                for i, p in enumerate(ps)
                for q in ps[i + 1 :]
            ):
                continue
            segs = [
                (i, pts[k], combo[i][2])
                for i, (a, b) in enumerate(pairs)
                for k in (a, b)
            ]
            crosses = sum(
                1
                for i, s in enumerate(segs)
                for t in segs[i + 1 :]
                if s[0] != t[0] and seg_cross(s[1], s[2], t[1], t[2])
            )
            sp = [c[2] for c in combo]
            spread = (max(p[0] for p in sp) - min(p[0] for p in sp)) + 1.5 * (
                max(p[1] for p in sp) - min(p[1] for p in sp)
            )
            score = (
                sum(c[0] for c in combo)
                + 0.3 * wide
                + 12 * unlinked
                + 1.6 * spread
                - 60 * crosses
            )
            if best is None or score > best[0]:
                best = (score, pairs, ps)
    return best[1], best[2]


def fill_classes(cls_pos):
    """Add classes that type no fact, each in the emptiest spot of the sheet."""
    out = list(cls_pos)
    grid = [
        (float(u), float(v))
        for u in range(24, int(W) - 23, 6)
        for v in range(16, int(D) - 15, 6)
    ]
    for _ in range(EXTRA_CLASSES):
        free = [
            p
            for p in grid
            if all(abs(p[0] - q[0]) >= 46 or abs(p[1] - q[1]) >= 32 for q in out)
        ]
        out.append(
            max(
                free,
                key=lambda p: min(math.dist(P(*p, Y_TOP), P(*q, Y_TOP)) for q in out),
            )
        )
    return out


def schema_edges(cls_pos):
    """A spanning tree over the classes, nearest first, plus one closing edge."""
    n = len(cls_pos)
    sp = [P(*p, Y_TOP) for p in cls_pos]
    order = sorted(
        (math.dist(sp[i], sp[j]), i, j) for i in range(n) for j in range(i + 1, n)
    )
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    tree, rest = [], []
    for _, i, j in order:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            tree.append((i, j))
        else:
            rest.append((i, j))
    return tree + rest[:1]


# ---------------------------------------------------------------- SVG
def svg() -> str:
    uv = lay_out()
    pairs, cls_pos = place_classes(uv)
    cls_of = {k: i for i, (a, b) in enumerate(pairs) for k in (a, b)}
    all_cls = fill_classes(cls_pos)

    top_front = max(P(0, D, Y_TOP)[1], P(W, D, Y_TOP)[1])
    bot_back = min(P(0, 0, Y_BOT)[1], P(W, 0, Y_BOT)[1])
    bot_front = max(P(0, D, Y_BOT)[1], P(W, D, Y_BOT)[1])
    top_back = min(P(0, 0, Y_TOP)[1], P(W, 0, Y_TOP)[1])
    y0 = math.floor(top_back - 22)
    height = round(bot_front + 8 - y0)

    o = [
        f'<svg viewBox="0 {y0} 480 {height}" role="img" aria-labelledby="oc-layers-title" '
        'xmlns="http://www.w3.org/2000/svg">',
        '<title id="oc-layers-title">Documents pass through a render-and-critique loop and '
        "become two layers of one graph: an ontology of classes above, and facts below, "
        "each fact tied to its class.</title>",
        "<defs>",
        '<marker id="oc-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto"><path class="oc-layers__arrowhead" '
        'd="M0,1 L9,5 L0,9 z"/></marker>',
        '<marker id="oc-arrow-loop" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4.5" '
        'markerHeight="4.5" orient="auto"><path class="oc-layers__arrowhead '
        'oc-layers__arrowhead--loop" d="M0,1 L9,5 L0,9 z"/></marker>',
        "</defs>",
    ]

    # documents and the render-and-critique loop
    cy = round((top_front + bot_back) / 2) - 2
    o.append(
        '<g class="oc-layers__doc">'
        f'<rect x="32" y="{cy - 38}" width="48" height="64" rx="4"/>'
        f'<rect x="24" y="{cy - 30}" width="48" height="64" rx="4"/>'
        f'<rect x="16" y="{cy - 22}" width="48" height="64" rx="4"/></g>'
    )
    o.append(
        '<g class="oc-layers__text-line">'
        + "".join(
            f'<path d="M25,{cy - 7 + 9 * i} H{[55, 50, 55, 46, 53][i]}"/>'
            for i in range(5)
        )
        + "</g>"
    )
    o.append(
        f'<text class="oc-layers__caption" x="48" y="{cy + 62}" text-anchor="middle">Documents</text>'
    )
    o.append(
        f'<path class="oc-layers__flow" d="M88,{cy + 1} H104" marker-end="url(#oc-arrow)"/>'
    )
    lx, r = 136, 22

    def arc(a):
        return (
            lx + r * math.cos(math.radians(a)),
            cy + 1 + r * math.sin(math.radians(a)),
        )

    o.append(
        f'<path class="oc-layers__loop" d="M{f(arc(190))} A{r},{r} 0 0 1 {f(arc(350))}" '
        'marker-end="url(#oc-arrow-loop)"/>'
    )
    o.append(
        f'<path class="oc-layers__loop oc-layers__loop--back" d="M{f(arc(10))} A{r},{r} 0 0 1 '
        f'{f(arc(170))}" marker-end="url(#oc-arrow-loop)"/>'
    )
    o.append(
        f'<text class="oc-layers__small" x="{lx}" y="{cy - 30}" text-anchor="middle">render</text>'
    )
    o.append(
        f'<text class="oc-layers__small" x="{lx}" y="{cy + 40}" text-anchor="middle">critique</text>'
    )
    t_end, b_end = P(0, 0.72 * D, Y_TOP), P(0, 0.30 * D, Y_BOT)
    o.append(
        f'<path class="oc-layers__flow" d="M166,{cy - 8} Q180,{t_end[1] + 12:.1f} '
        f'{t_end[0] - 7:.1f},{t_end[1]:.1f}" marker-end="url(#oc-arrow)"/>'
    )
    o.append(
        f'<path class="oc-layers__flow" d="M166,{cy + 10} Q180,{b_end[1] - 12:.1f} '
        f'{b_end[0] - 7:.1f},{b_end[1]:.1f}" marker-end="url(#oc-arrow)"/>'
    )

    # facts sheet; the dashed lines go under the dots and under the ontology sheet
    o.append(
        f'<path class="oc-layers__slab oc-layers__slab--facts" d="{slab(Y_BOT)}"/>'
    )
    o.append(
        f'<path class="oc-layers__sheet oc-layers__sheet--facts" d="{quad(Y_BOT)}"/>'
    )
    o.append(
        '<g class="oc-layers__fact-edge">'
        + "".join(
            f'<path d="M{f(P(*uv[a], Y_BOT))} L{f(P(*uv[b], Y_BOT))}"/>'
            for a, b in LINKS
        )
        + "</g>"
    )
    o.append(
        '<g class="oc-layers__typed">'
        + "".join(
            f'<path d="M{f(P(*uv[k], Y_BOT))} L{f(P(*cls_pos[cls_of[k]], Y_TOP))}"/>'
            for k in STARS
        )
        + "</g>"
    )
    o.append(
        '<g class="oc-layers__fact">'
        + "".join(
            f'<path d="{disc(*uv[k], Y_BOT, radius(STARS[k][2]))}"/>' for k in STARS
        )
        + "</g>"
    )

    # ontology sheet
    o.append(f'<path class="oc-layers__slab oc-layers__slab--onto" d="{slab(Y_TOP)}"/>')
    o.append(
        f'<path class="oc-layers__sheet oc-layers__sheet--onto" d="{quad(Y_TOP)}"/>'
    )
    o.append(
        '<g class="oc-layers__schema-edge">'
        + "".join(
            f'<path d="M{f(P(*all_cls[i], Y_TOP))} L{f(P(*all_cls[j], Y_TOP))}"/>'
            for i, j in schema_edges(all_cls)
        )
        + "</g>"
    )
    o.append(
        '<g class="oc-layers__class">'
        + "".join(
            f'<path d="{quad(Y_TOP, u - 16, v - 12, u + 16, v + 12)}"/>'
            for u, v in all_cls
        )
        + "</g>"
    )

    tl, bl = P(0, 0, Y_TOP), P(0, D, Y_BOT)
    o.append(
        f'<text class="oc-layers__label" x="{tl[0] - 14:.1f}" y="{tl[1] + 6:.1f}" '
        'text-anchor="end">Ontology</text>'
    )
    o.append(
        f'<text class="oc-layers__label" x="{bl[0] - 14:.1f}" y="{bl[1] + 6:.1f}" '
        'text-anchor="end">Facts</text>'
    )
    o.append("</svg>")
    return "\n".join(o)


FIGURE = re.compile(r'(<figure class="oc-layers">\n)(.*?)(\n</figure>)', re.S)


def main() -> int:
    page = INDEX.read_text()
    if not FIGURE.search(page):
        print(f'{INDEX}: no <figure class="oc-layers"> block', file=sys.stderr)
        return 1
    updated = FIGURE.sub(lambda m: m.group(1) + svg() + m.group(3), page, count=1)
    if "--check" in sys.argv[1:]:
        if updated != page:
            print(
                f"{INDEX}: figure is stale; run docs/_build/hero_figure.py",
                file=sys.stderr,
            )
            return 1
        return 0
    if updated != page:
        INDEX.write_text(updated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
