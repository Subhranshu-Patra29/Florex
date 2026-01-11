import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from g2p.plotcopy import plot_fp  # existing Graph2Plan visualization function

# === Load .mat file ===
mat_path = r"D:\\major project\\FLorexFinal\\InterfaceFinal\\static\\343.mat"
mat_data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
data = mat_data['data']  # adjust key if needed

# === Extract Data ===
boundary = np.array(data.boundary)
newBox = np.array(data.newBox)
order = np.array(data.order, dtype=int) - 1  # 0-based index for Python
rType = np.array(data.rType)
doors = np.array(data.doors)
windows = np.array(data.windows)

# === Visualize with plot_fp ===
ax = plot_fp(boundary, newBox[order], rType[order], doors, windows)

# === Save Output ===
fig = plt.gcf()

fig.canvas.draw()
fig.canvas.print_figure('test.png')
plt.show()