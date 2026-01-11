import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


# ---------- INPUT FILE ----------
MAT_FILE_PATH = r"D:\\major project\\FLorexFinal\\InterfaceFinal\\static\\343.mat"
# ---------------------------------------------------

# --------- SET DEFAULT EXPORT FOLDER HERE ----------
EXPORT_PATH = r"D:\major project\FLorexFinal\InterfaceFinal\output_3d"
os.makedirs(EXPORT_PATH, exist_ok=True)


# ------------ PROCEDURAL MATERIALS ------------

def wood(x, y):
    """Smooth wood-like grain"""
    return 0.55 + 0.10*np.sin(0.07*x) + 0.08*np.cos(0.06*y)


def wall_plaster(x, y):
    """Soft off-white plaster feel"""
    return 0.78 + 0.03*np.sin(0.1*(x+y))


def shadow_shade(z):
    """Vertical shadow for realism"""
    return 0.75 + 0.25*(z/30)


# ----------------------------------------------


def generate_3d_realistic(height=25):

    data = sio.loadmat(MAT_FILE_PATH)
    record = data["data"][0,0]
    rBound = record["rBoundary"]

    polys = [rBound[0,i] for i in range(rBound.shape[1])]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    allx=[]; ally=[]

    for poly in polys:
        xs, ys = poly[:,0], poly[:,1]
        allx.extend(xs); ally.extend(ys)

        # -------- FLOOR --------
        verts = [(x,y,0) for x,y in zip(xs,ys)]
        fc = wood(xs.mean(), ys.mean())  # single smooth tone

        floor = Poly3DCollection([verts])
        floor.set_facecolor((fc, fc*0.9, fc*0.7))
        floor.set_edgecolor((0.2,0.2,0.2))
        floor.set_alpha(0.98)
        ax.add_collection3d(floor)

        # -------- WALLS --------
        n = len(xs)
        for i in range(n):
            j = (i+1)%n

            quad = [
                (xs[i],ys[i],0),
                (xs[j],ys[j],0),
                (xs[j],ys[j],height),
                (xs[i],ys[i],height)
            ]

            wc = wall_plaster(xs[i], ys[i])
            wall = Poly3DCollection([quad])
            wall.set_facecolor((wc, wc, wc))
            wall.set_edgecolor((0.25,0.25,0.25))
            wall.set_alpha(0.92)
            ax.add_collection3d(wall)


    # ---------- CAMERA ----------
    # Slight tilt → slight angle → see walls + roofless room
    ax.view_init(
        elev=66,    # 90 = flat top, 78 = slight tilt
        azim=-50    # rotate
    )
    ax.dist = 9     # realistic zoom

    # ---------- BOUNDING ----------
    xmin, xmax = min(allx), max(allx)
    ymin, ymax = min(ally), max(ally)
    ax.set_xlim(xmin-10, xmax+10)
    ax.set_ylim(ymin-10, ymax+10)
    ax.set_zlim(0, height+3)

    ax.set_box_aspect((xmax-xmin, ymax-ymin, height))

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_facecolor((1,1,1))

    # ---------- SAVE ----------
    base = os.path.basename(MAT_FILE_PATH).replace(".mat","")
    out = os.path.join(EXPORT_PATH, f"{base}_realistic_angled.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n📌 DONE! Realistic angled PNG created:\n", out)
    return out


# -------- RUN --------
generate_3d_realistic()
