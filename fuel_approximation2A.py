import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================================================
# PARAMETERS
# =====================================================

R = 1.0
PORT_W = 0.5

# Initial state
x0, y0 = 10.0, 10.0
vx0, vy0 = 10.0, 0.0

# =====================================================
# THRUSTER MATRIX (D + L)
# =====================================================

D = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]])

L = 0.5*np.array([[0,0,1],
                  [0,-1,0],
                  [1,0,0]])

M = D + L

# Effective 2D force mapping (use first two rows only)
M2 = M[0:2,0:2]

# Compute Q = M^{-T} M^{-1}
Minv = np.linalg.inv(M2)
Q = Minv.T @ Minv

# =====================================================
# FUEL FUNCTION
# =====================================================

def fuel_cost(T, w, x0, y0, vx0, vy0):
    
    pf = np.array([R*np.cos(w*T),
                   R*np.sin(w*T)])
    
    vf = np.array([-w*R*np.sin(w*T),
                    w*R*np.cos(w*T)])
    
    p0 = np.array([x0, y0])
    v0 = np.array([vx0, vy0])
    
    delta_p = pf - p0 - v0*T
    delta_v = vf - v0
    
    term1 = (12 / T**3) * (delta_p.T @ Q @ delta_p)
    term2 = (4 / T) * (delta_v.T @ Q @ delta_v)
    term3 = (12 / T**2) * (delta_p.T @ Q @ delta_v)
    
    J = term1 + term2 - term3
    return J

# =====================================================
# SURFACE 1: Fuel vs T and w
# =====================================================

T_vals = np.linspace(10, 100, 150)
w_vals = np.linspace(0.3, 0.8, 150)

TT, WW = np.meshgrid(T_vals, w_vals)
J_surface = np.zeros_like(TT)

for i in range(TT.shape[0]):
    for j in range(TT.shape[1]):
        J_surface[i,j] = fuel_cost(
            TT[i,j], WW[i,j],
            x0, y0, vx0, vy0
        )

# =====================================================
# SURFACE 2: Fuel vs x0 and y0
# =====================================================

T_fixed = 40.0
w_fixed = PORT_W

x_vals = np.linspace(-15, 15, 150)
y_vals = np.linspace(-15, 15, 150)

X, Y = np.meshgrid(x_vals, y_vals)
J_xy = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        J_xy[i,j] = fuel_cost(
            T_fixed, w_fixed,
            X[i,j], Y[i,j],
            vx0, vy0
        )

# =====================================================
# PLOTTING BOTH IN SAME FIGURE
# =====================================================

fig = plt.figure(figsize=(18,8))

# Surface 1
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(TT, WW, J_surface, cmap='plasma')
ax1.set_xlabel("Docking Time T")
ax1.set_ylabel("Port Rotation w")
ax1.set_zlabel("Fuel J")
ax1.set_title("Fuel Surface: J(T, w) with Thruster Matrix")

# Surface 2
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(X, Y, J_xy, cmap='viridis')
ax2.set_xlabel("Initial Position x0")
ax2.set_ylabel("Initial Position y0")
ax2.set_zlabel("Fuel J")
ax2.set_title(f"Fuel Surface: J(x0, y0) | T={T_fixed}, w={w_fixed}")

plt.tight_layout()
plt.show()