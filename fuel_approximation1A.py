import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
x0, y0 = 10.0, 10.0
R = 1.0

T = np.linspace(10, 100, 200)
w = np.linspace(0.4, 0.8, 200)
TT, WW = np.meshgrid(T, w)

# Fully expanded formula
term1 = (12 / TT**3) * (
    R**2 + x0**2 + y0**2
    - 2*R*(x0*np.cos(WW*TT) + y0*np.sin(WW*TT))
)

term2 = (4 * WW**2 * R**2) / TT

term3 = (12 * WW * R / TT**2) * (
    y0*np.cos(WW*TT) - x0*np.sin(WW*TT)
)

J = term1 + term2 + term3

# =====================================================
# TWO SURFACES IN SAME FIGURE
# =====================================================

fig = plt.figure(figsize=(18,8))

# ---------------------------
# Surface 1: Fuel vs T and w
# ---------------------------
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(TT, WW, J, cmap='plasma')
ax1.set_xlabel("Docking Time T")
ax1.set_ylabel("Port Rotation w")
ax1.set_zlabel("Fuel J")
ax1.set_title("Fuel Surface: J(T, w)")

# ---------------------------
# Surface 2: Fuel vs x0 and y0
# ---------------------------
# Fix T and w
T_fixed = 40.0
w_fixed = 0.6
x_vals = np.linspace(-15, 15, 200)
y_vals = np.linspace(-15, 15, 200)
X, Y = np.meshgrid(x_vals, y_vals)

term1_xy = (12 / T_fixed**3) * (
    R**2 + X**2 + Y**2
    - 2*R*(X*np.cos(w_fixed*T_fixed) + Y*np.sin(w_fixed*T_fixed))
)
term2_xy = (4 * w_fixed**2 * R**2) / T_fixed
term3_xy = (12 * w_fixed * R / T_fixed**2) * (
    Y*np.cos(w_fixed*T_fixed) - X*np.sin(w_fixed*T_fixed)
)
J_xy = term1_xy + term2_xy + term3_xy

ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(X, Y, J_xy, cmap='viridis')
ax2.set_xlabel("Initial Position x0")
ax2.set_ylabel("Initial Position y0")
ax2.set_zlabel("Fuel J")
ax2.set_title(f"Fuel Surface: J(x0, y0) | T={T_fixed}, w={w_fixed}")

plt.tight_layout()
plt.show()