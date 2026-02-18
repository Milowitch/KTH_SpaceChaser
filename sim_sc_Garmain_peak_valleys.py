import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
# PARAMETERS 
n = 0.0                 # orbital mean motion (rad/s)
R_orbit = 1.0           # docking port orbit radius
PORT_W = 0.5
T_min, T_max = 10, 100
num_points = 50         # num of time sampling
SPHERE_RADIUS = R_orbit
REPULSION_K = 0.0       #Repulsion zone out gain
steps = 5000
x0 = np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0])

# CW MATRICES 
def cw_A(n):
    return np.array([
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
        [3*n**2,0,0,0,2*n,0],
        [0,0,0,-2*n,0,0],
        [0,0,-n**2,0,0,0]
    ])

def cw_B():
    return np.array([
        [0,0,0],
        [0,0,0],
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

A = cw_A(n)
B = cw_B()

# Docking Target State 
def docking_state(T, R=R_orbit, w=PORT_W):
    x = R*np.cos(w*T)
    y = R*np.sin(w*T)
    z = 0
    vx = -w*R*np.sin(w*T)
    vy = w*R*np.cos(w*T)
    vz = 0
    return np.array([x, y, z, vx, vy, vz])

def gramian(A, B, T, steps=200):
    dt = T / steps
    W = np.zeros((6,6))
    for i in range(steps):
        t = i*dt
        Phi = expm(A*t)
        W += Phi @ B @ B.T @ Phi.T * dt
    return W

#Repulsion (Keep-Out)
def repulsion_force(pos):
    r = np.linalg.norm(pos)
    if r < SPHERE_RADIUS and r > 1e-6:
        return REPULSION_K * (SPHERE_RADIUS - r) * pos/r
    else:
        return np.zeros(3)

def cw_dynamics_with_repulsion(x, u):
    f = A @ x + B @ u
    f[:3] += repulsion_force(x[:3])
    return f

def rk4_step(f, x, u, dt):
    k1 = f(x,u)
    k2 = f(x + dt/2*k1, u)
    k3 = f(x + dt/2*k2, u)
    k4 = f(x + dt*k3, u)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def trajectory_with_keepout(A,B,x0,xf,T,steps=5000):
    dt = T / steps
    W_T = gramian(A,B,T)
    W_inv = np.linalg.inv(W_T)
    Phi_T = expm(A*T)

    traj = np.zeros((steps+1,6))
    u_hist = np.zeros((steps,3))
    t_hist = np.linspace(0,T,steps)
    traj[0] = x0

    for i in range(steps):
        t = i*dt
        Phi = expm(A*(T-t))
        # Minimum energy control law
        u = B.T @ Phi.T @ W_inv @ (xf - Phi_T @ x0)
        u_hist[i] = u
        traj[i+1] = rk4_step(cw_dynamics_with_repulsion, traj[i], u, dt)

    traj[-1] = xf
    return traj, u_hist, t_hist

Ts = np.linspace(T_min,T_max,num_points)
Jvals = []

for T in Ts:
    W = gramian(A,B,T)
    if np.linalg.cond(W) > 1e12:
        Jvals.append(np.nan)
        continue
    Phi = expm(A*T)
    xf = docking_state(T)
    d = xf - Phi @ x0
    J = d.T @ np.linalg.inv(W) @ d
    Jvals.append(J)

Jvals = np.array(Jvals)
valid = ~np.isnan(Jvals)
Ts_valid = Ts[valid]
J_valid = Jvals[valid]

#  Peaks & Valleys 


peaks,_ = find_peaks(J_valid)
valleys,_ = find_peaks(-J_valid)

first_peak_idx = peaks[np.argmax(J_valid[peaks])]
valleys_after_peak = valleys[valleys > first_peak_idx]
first6_valleys_idx = valleys_after_peak[:6]

print("First high peak time:", Ts_valid[first_peak_idx])
print("Fuel at peak:", J_valid[first_peak_idx])
print("First 6 valley times:", Ts_valid[first6_valleys_idx])
print("Fuel at valleys:", J_valid[first6_valleys_idx])

plt.figure(figsize=(10,5))
plt.plot(Ts_valid,J_valid,label='Fuel Cost')
plt.plot(Ts_valid[first_peak_idx],J_valid[first_peak_idx],'ro')
plt.plot(Ts_valid[first6_valleys_idx],J_valid[first6_valleys_idx],'go')
plt.xlabel("Docking Time (s)")
plt.ylabel("Fuel")
plt.title("Fuel vs Docking Time")
plt.grid(True)
plt.legend()


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
colors = ['r','g','b','c','m','y','k']

# First Peak
T_peak = Ts_valid[first_peak_idx]
xf_peak = docking_state(T_peak)
traj_peak, u_peak, t_peak = trajectory_with_keepout(A,B,x0,xf_peak,T_peak)

print("\n==== THRUSTER (Peak Case) ====")
print("Max thrust per axis:", np.max(np.abs(u_peak),axis=0))
print("Peak total thrust:", np.max(np.linalg.norm(u_peak,axis=1)))

ax.plot(traj_peak[:,0], traj_peak[:,1], traj_peak[:,2],
        color=colors[0],
        label=f"Peak T={T_peak:.1f}s")

# Plot thruster profile for peak
plt.figure(figsize=(10,5))
plt.plot(t_peak, u_peak[:,0], label='Ux')
plt.plot(t_peak, u_peak[:,1], label='Uy')
plt.plot(t_peak, u_peak[:,2], label='Uz')
plt.xlabel("Time (s)")
plt.ylabel("Thruster Input")
plt.title(f"Thruster Profile vs Time (Peak, T={T_peak:.2f}s)")
plt.grid(True)
plt.legend()

#  Valleys 
for i, idx in enumerate(first6_valleys_idx):
    T_val = Ts_valid[idx]
    xf_val = docking_state(T_val)
    traj_val, u_val, t_val = trajectory_with_keepout(A,B,x0,xf_val,T_val)

    print(f"\n==== THRUSTER (Valley {i+1}) ====")
    print("Max thrust per axis:", np.max(np.abs(u_val),axis=0))
    print("Peak total thrust:", np.max(np.linalg.norm(u_val,axis=1)))

    ax.plot(traj_val[:,0], traj_val[:,1], traj_val[:,2],
            color=colors[i+1],
            label=f"Valley {i+1} T={T_val:.1f}s")

    # Plot thruster profile per valley
    plt.figure(figsize=(10,5))
    plt.plot(t_val, u_val[:,0], label='Ux')
    plt.plot(t_val, u_val[:,1], label='Uy')
    plt.plot(t_val, u_val[:,2], label='Uz')
    plt.xlabel("Time (s)")
    plt.ylabel("Thruster Input")
    plt.title(f"Thruster Profile vs Time (Valley {i+1}, T={T_val:.2f}s)")
    plt.grid(True)
    plt.legend()

u,v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
xs = SPHERE_RADIUS*np.cos(u)*np.sin(v)
ys = SPHERE_RADIUS*np.sin(u)*np.sin(v)
zs = SPHERE_RADIUS*np.cos(v)
ax.plot_wireframe(xs, ys, zs, color='k', alpha=0.3)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Docking Trajectories")
ax.legend()
ax.set_box_aspect([1,1,1])

plt.show()
