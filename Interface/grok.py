import scipy.io as sio
import numpy as np
import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = "browser"


# ================================
# CONFIG
# ================================
MAT_FILE_PATH = r"D:\\major project\\FLorexFinal\\InterfaceFinal\\static\\343.mat"

wall_height = 25
wall_thickness = 0.4

stretch_x = 24
stretch_y = 10

floor_color = "rgb(210,210,210)"


# =====================================================
# HELPER: ORDER POLYGON POINTS (FIX SCRAMBLED ROOMS)
# =====================================================
def order_polygon(xs, ys):
    cx = np.mean(xs)
    cy = np.mean(ys)
    angles = np.arctan2(ys - cy, xs - cx)
    order = np.argsort(angles)
    return xs[order], ys[order]


# =====================================================
# LOAD MAT FILE
# =====================================================
data = sio.loadmat(MAT_FILE_PATH)
record = data["data"][0,0]
rBound = record["rBoundary"]
polys = [rBound[0,i] for i in range(rBound.shape[1])]


fig = go.Figure()


# =====================================================
# GLOBAL SHADOW BASE
# =====================================================
allx = []
ally = []
for poly in polys:
    poly = np.array(poly)

    # Normalize shapes
    if poly.ndim == 3:
        poly = poly.reshape(-1,2)
    if poly.ndim == 1:
        poly = poly.reshape(-1,2)

    if poly.shape[1] != 2:
        continue

    xs = poly[:,0] * stretch_x
    ys = poly[:,1] * stretch_y

    allx.extend(xs)
    ally.extend(ys)


if len(allx)>0 and len(ally)>0:
    xmin, xmax = min(allx), max(allx)
    ymin, ymax = min(ally), max(ally)

    fig.add_trace(go.Mesh3d(
        x=[xmin-10, xmax+10, xmax+10, xmin-10],
        y=[ymin-10, ymin-10, ymax+10, ymax+10],
        z=[-4,-4,-4,-4],
        color='lightgray',
        opacity=0.20
    ))


# =====================================================
# DRAW ROOMS
# -------- ROOMS --------
for poly in polys:
    poly = np.array(poly)

    # Normalize shape (Nx2 always)
    if poly.ndim == 3:
        poly = poly.reshape(-1,2)
    if poly.ndim == 1:
        poly = poly.reshape(-1,2)
    if poly.shape[1] != 2:
        continue

    xs = poly[:,0] * stretch_x
    ys = poly[:,1] * stretch_y

    # ==== Skip degenerate triangles ====
    if len(xs) < 3:
        continue

    # ==== Skip areas that are too tiny ====
    area = abs(np.sum(xs*np.roll(ys,-1) - np.roll(xs,-1)*ys)) / 2
    if area < 30:   # tweakable threshold
        continue

    # Convert to list for plotly
    xs = xs.tolist()
    ys = ys.tolist()
    zs = [0]*len(xs)

    # ---------- FLOOR ----------
    fig.add_trace(go.Scatter3d(
        x=xs + [xs[0]],
        y=ys + [ys[0]],
        z=zs + [0],
        mode="lines",
        line=dict(color="black", width=3)
    ))

    fig.add_trace(go.Mesh3d(
        x=xs,
        y=ys,
        z=np.zeros_like(xs),
        color=floor_color,
        opacity=0.95
    ))

    # ---------- WALLS ----------
   # ---------- WALLS (solid blocks) ----------
    for i in range(len(xs)):
        j = (i+1) % len(xs)

        # outward normal extrusion
        vx = ys[j] - ys[i]
        vy = xs[i] - xs[j]
        norm = np.sqrt(vx*vx + vy*vy)
        if norm > 0:
            vx /= norm
            vy /= norm

        ix1 = xs[i] + vx * wall_thickness
        iy1 = ys[i] + vy * wall_thickness
        ix2 = xs[j] + vx * wall_thickness
        iy2 = ys[j] + vy * wall_thickness

        # solid wall panel
        fig.add_trace(go.Mesh3d(
            x=[xs[i], xs[j], ix2, ix1],
            y=[ys[i], ys[j], iy2, iy1],
            z=[0, 0, wall_height, wall_height],
            color="white",
            opacity=1        # <- full solid wall
        ))

        # crisp top outline
        fig.add_trace(go.Scatter3d(
            x=[xs[i], xs[j], ix2, ix1, xs[i]],
            y=[ys[i], ys[j], iy2, iy1, ys[i]],
            z=[wall_height]*5,
            mode="lines",
            line=dict(color="black", width=3)
        ))


# =====================================================
# VIEW SETTINGS
# =====================================================
fig.update_layout(
    scene=dict(
        aspectratio=dict(x=1, y=1, z=0.20),
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
    ),
    paper_bgcolor='white'
)

fig.show()
