import scipy.io as sio
import numpy as np
import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = "browser"


# ===========================
# CONFIG
# ===========================
MAT_FILE_PATH = r"D:\\major project\\FLorexFinal\\InterfaceFinal\\static\\343.mat"
wall_height = 25
wall_thickness = 0.4
stretch_x = 24
stretch_y = 10

grey_floor = "rgb(210,210,210)"
white_wall = "rgb(255,255,255)"


# ===========================
# LOAD & PARSE MAT
# ===========================
data = sio.loadmat(MAT_FILE_PATH)
record = data["data"][0,0]
rBound = record["rBoundary"]
polys = [rBound[0,i] for i in range(rBound.shape[1])]


# ===========================
# AREA FOR ROOM DETECTION
# ===========================
def polygon_area(xs, ys):
    return 0.5*np.abs(np.dot(xs,np.roll(ys,1))-np.dot(ys,np.roll(xs,1)))

areas = []
scaled_polys = []

for poly in polys:
    xs = poly[:,0]*stretch_x
    ys = poly[:,1]*stretch_y
    area = polygon_area(xs,ys)
    areas.append(area)
    scaled_polys.append((xs,ys))

# Room ranking
order = np.argsort(areas)[::-1]


# ===========================
# ROOM ROLE MAPPING
# ===========================
room_role = {}
if len(order) > 0: room_role[order[0]] = "living"
if len(order) > 1: room_role[order[1]] = "bedroom"
if len(order) > 2: room_role[order[2]] = "dining"
if len(order) > 3: room_role[order[3]] = "kitchen"


# ===========================
# FURNITURE GENERATORS
# ===========================
def centroid(xs, ys):
    return np.mean(xs), np.mean(ys)

def furniture_box(cx, cy, w, d, h, color):
    # centered 3D block
    x = [cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2]
    y = [cy-d/2, cy-d/2, cy+d/2, cy+d/2, cy-d/2]
    z = [0,0,0,0,0]
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=[0,h,h,0,0],
        color=color,
        opacity=1
    ))


# ===========================
# SHADOW BASE / GROUND
# ===========================
allx=[]; ally=[]
for xs,ys in scaled_polys:
    allx.extend(xs); ally.extend(ys)

xmin, xmax = min(allx), max(allx)
ymin, ymax = min(ally), max(ally)

fig = go.Figure()

fig.add_trace(go.Mesh3d(
    x=[xmin-10, xmax+10, xmax+10, xmin-10],
    y=[ymin-10, ymin-10, ymax+10, ymax+10],
    z=[-4,-4,-4,-4],
    opacity=0.22,
    color="lightgrey"
))

# ===========================
# RENDER ROOMS + WALLS
# ===========================
for idx, (xs,ys) in enumerate(scaled_polys):

    # FLOOR
    fig.add_trace(go.Mesh3d(
        x=xs, y=ys, z=np.zeros_like(xs),
        color=grey_floor, opacity=0.95
    ))

    # WALLS
    n = len(xs)
    for i in range(n):
        j = (i+1) % n

        vx = ys[j] - ys[i]
        vy = xs[i] - xs[j]
        norm = np.sqrt(vx*vx + vy*vy)
        if norm>0:
            vx/=norm; vy/=norm

        ix1 = xs[i] + vx * wall_thickness
        iy1 = ys[i] + vy * wall_thickness
        ix2 = xs[j] + vx * wall_thickness
        iy2 = ys[j] + vy * wall_thickness

        fig.add_trace(go.Mesh3d(
            x=[xs[i], xs[j], ix2, ix1],
            y=[ys[i], ys[j], iy2, iy1],
            z=[0,0,wall_height,wall_height],
            color=white_wall,
            opacity=1
        ))


# ===========================
# PLACE FURNITURE
# ===========================
for idx,(xs,ys) in enumerate(scaled_polys):
    cx,cy = centroid(xs,ys)

    role = room_role.get(idx,"none")

    if role=="living":
        furniture_box(cx,cy, 15,10, 6, "rgb(180,180,180)")  # sofa
        furniture_box(cx+12,cy, 6,6, 4, "rgb(160,160,160)") # coffee table

    if role=="bedroom":
        furniture_box(cx,cy, 18,12, 6, "rgb(180,180,180)")  # bed
        furniture_box(cx+14,cy, 10,4, 7, "rgb(160,160,160)")# wardrobe

    if role=="dining":
        furniture_box(cx,cy, 14,8, 5, "rgb(180,180,180)")   # dining table

    if role=="kitchen":
        furniture_box(cx,cy, 20,4, 6, "rgb(150,150,150)")   # kitchen counter


# ===========================
# VISUAL SETTINGS
# ===========================
fig.update_layout(
    title="3D Floorplan with Minimalist Furniture (Auto)",
    scene=dict(
        aspectratio=dict(x=1,y=1,z=0.25),
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
    ),
    paper_bgcolor='white'
)

fig.show()
