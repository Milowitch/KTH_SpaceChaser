import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from matplotlib.animation import FuncAnimation

# parameters
n = 0.011
R_orbit = 1.0
SPHERE_RADIUS = R_orbit
PORT_W = 0.5
T_min, T_max = 10, 100
num_points = 50
steps = 200
slope_threshold = 0.01

# init positions
x0_list = [
    np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([12.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
    np.array([12.0, 12.0, -0.5, 0.0, 0.0, 0.0])
]

phases = [0, 2*np.pi/3, 4*np.pi/3]

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

def docking_state(t, phase=0.0):
    theta = PORT_W*t + phase
    x = R_orbit*np.cos(theta)
    y = R_orbit*np.sin(theta)
    z = 0
    vx = -PORT_W*R_orbit*np.sin(theta)
    vy = PORT_W*R_orbit*np.cos(theta)
    vz = 0
    return np.array([x,y,z,vx,vy,vz])

def gramian(A,B,T,steps=250):
    dt = T/steps
    W = np.zeros((6,6))
    for i in range(steps):
        t = i*dt
        Phi = expm(A*t)
        W += Phi @ B @ B.T @ Phi.T * dt
    return W

def dynamics_thruster(x, U):
    D = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    L = np.array([[0,0,0,1],[0,0,-1,0],[1,-1,0,0]])*0.5
    F_total = np.zeros(3)
    for i in range(3):
        F_total += D@U[:,i] + L@U[:,i]

    dxdt = np.zeros(6)
    dxdt[0:3] = x[3:6]
    dxdt[3] = 3*n**2*x[0] + 2*n*x[4] + F_total[0]
    dxdt[4] = -2*n*x[3] + F_total[1]
    dxdt[5] = -n**2*x[2] + F_total[2]
    return dxdt, F_total

def rk4_thruster(x,U,dt):
    k1,_ = dynamics_thruster(x,U)
    k2,_ = dynamics_thruster(x + dt/2*k1,U)
    k3,_ = dynamics_thruster(x + dt/2*k2,U)
    k4,_ = dynamics_thruster(x + dt*k3,U)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def trajectory_SC_thruster(A,B,x0,phase,T,steps=200):
    dt = T/steps
    traj = np.zeros((steps+1,6))
    traj[0] = x0

    thruster_history = np.zeros((steps,3))
    u_history = np.zeros((steps,3))

    # NEW: store all 12 physical thrusters
    U_all_history = np.zeros((steps,4,3))

    for i in range(steps):
        t_remaining = T - i*dt
        xf = docking_state(T, phase)

        W = gramian(A,B,t_remaining)
        W_inv = np.linalg.inv(W)
        Phi = expm(A*t_remaining)

        u_desired = B.T @ Phi.T @ W_inv @ (xf - Phi@traj[i])
        u_history[i] = u_desired

        # 4 thrusters per axis × 3 axes
        U = np.tile(np.append(u_desired/3,0).reshape(4,1),3)
        U_all_history[i] = U   # <-- SAVE ALL THRUSTERS

        traj[i+1] = rk4_thruster(traj[i],U,dt)

        _, F_total = dynamics_thruster(traj[i],U)
        thruster_history[i] = F_total

    return traj, thruster_history, u_history, U_all_history


#  Fuel vs Time 
Ts = np.linspace(T_min,T_max,num_points)
total_fuel_list = []
best_assignments = []

for T in Ts:
    cost_matrix = np.zeros((3,3))
    W = gramian(A,B,T)
    W_inv = np.linalg.inv(W)
    Phi_T = expm(A*T)

    for i, sc in enumerate(x0_list):
        for j, phase in enumerate(phases):
            xf = docking_state(T, phase)
            d = xf - Phi_T @ sc
            cost_matrix[i,j] = d.T @ W_inv @ d

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_fuel_list.append(cost_matrix[row_ind, col_ind].sum())
    best_assignments.append(col_ind)

total_fuel_list = np.array(total_fuel_list)

# select docking time by slope
slope = np.gradient(total_fuel_list, Ts)
practical_idx = np.where(np.abs(slope) < slope_threshold)[0]
best_T_idx = practical_idx[0] if len(practical_idx) > 0 else np.argmin(total_fuel_list)
best_T = Ts[best_T_idx]
best_assignment = best_assignments[best_T_idx]

print("Practical docking time:", best_T)
print("Total fuel:", total_fuel_list[best_T_idx])
# ---------- Fuel vs Time for each SC ----------
Ts = np.linspace(T_min,T_max,num_points)
total_fuel_list = []
best_assignments = []

# Keep per-spacecraft fuel
fuel_per_sc_over_time = np.zeros((len(x0_list), len(Ts)))

for t_idx, T in enumerate(Ts):
    cost_matrix = np.zeros((3,3))
    W = gramian(A,B,T)
    W_inv = np.linalg.inv(W)
    Phi_T = expm(A*T)

    for i, sc in enumerate(x0_list):
        for j, phase in enumerate(phases):
            xf = docking_state(T, phase)
            d = xf - Phi_T @ sc
            cost_matrix[i,j] = d.T @ W_inv @ d

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_fuel = cost_matrix[row_ind, col_ind].sum()
    total_fuel_list.append(total_fuel)
    best_assignments.append(col_ind)

    # store individual SC fuel
    for sc_idx, dp_idx in enumerate(col_ind):
        fuel_per_sc_over_time[sc_idx, t_idx] = cost_matrix[sc_idx, dp_idx]

total_fuel_list = np.array(total_fuel_list)

# Select docking time with slope
slope = np.gradient(total_fuel_list, Ts)
practical_idx = np.where(np.abs(slope) < slope_threshold)[0]
best_T_idx = practical_idx[0] if len(practical_idx) > 0 else np.argmin(total_fuel_list)
best_T = Ts[best_T_idx]
best_assignment = best_assignments[best_T_idx]

print("Practical docking time:", best_T)
print("Total fuel at that time:", total_fuel_list[best_T_idx])
for sc_idx in range(len(x0_list)):
    print(f"SC{sc_idx+1} fuel: {fuel_per_sc_over_time[sc_idx,best_T_idx]:.3f}")
plt.figure(figsize=(10,5))

# per spacecraft
colors = ['r','g','b']
for sc_idx in range(len(x0_list)):
    plt.plot(Ts, fuel_per_sc_over_time[sc_idx], color=colors[sc_idx], label=f'SC{sc_idx+1}')

plt.plot(Ts, total_fuel_list, 'k--', label='Total Fuel', linewidth=2)

plt.axvline(best_T, color='purple', linestyle=':', label=f'Selected T={best_T:.2f}s')

plt.xlabel("Docking Time (s)")
plt.ylabel("Fuel (J or norm² of control)")
plt.title("Fuel vs Docking Time per Spacecraft and Total")
plt.grid(True)
plt.legend()

#  Heatmap 
cost_matrix = np.zeros((3,3))
W = gramian(A,B,best_T)
W_inv = np.linalg.inv(W)
Phi_T = expm(A*best_T)

for i, sc in enumerate(x0_list):
    for j, phase in enumerate(phases):
        xf = docking_state(best_T, phase)
        d = xf - Phi_T@sc
        cost_matrix[i,j] = d.T@W_inv@d

plt.figure(figsize=(6,5))
sns.heatmap(cost_matrix, annot=True, fmt=".2f",
            xticklabels=[f"DP{j+1}" for j in range(3)],
            yticklabels=[f"SC{i+1}" for i in range(3)],
            cmap="viridis")
plt.title(f"Cost Matrix Heatmap (T={best_T:.2f}s)")

#  Trajectories + u(t) 
colors = ['r','g','b']
trajectories = []
thruster_histories = []
u_histories = [] #thruster X Y Z
U_histories = [] #thruster as a individual 4 thrusters × 3 axes = 12 thrusters 


for sc_idx, dp_idx in enumerate(best_assignment):
    traj, thrusters, u_hist , U_all= trajectory_SC_thruster(
        A,B,x0_list[sc_idx], phases[dp_idx], best_T
    )
    trajectories.append(traj)
    thruster_histories.append(thrusters)
    u_histories.append(u_hist)
    
    U_histories.append(U_all)


    t_array = np.linspace(0,best_T,len(U_all))

    plt.figure(figsize=(10,6))

    for axis in range(3):        # X,Y,Z groups
        for thr in range(4):     # 4 thrusters per axis
            plt.plot(t_array, U_all[:,thr,axis],
                     label=f'A{axis+1}-T{thr+1}')

    plt.title(f"All 12 Thrusters SC{sc_idx+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Thruster Force")
    plt.grid(True)
    plt.legend(ncol=4, fontsize=8)
    # u plot
    plt.figure(figsize=(8,4))
    t_array = np.linspace(0,best_T,len(thrusters))
    plt.plot(t_array, thrusters[:,0],label='Fx')
    plt.plot(t_array, thrusters[:,1],label='Fy')
    plt.plot(t_array, thrusters[:,2],label='Fz')

    plt.title(f"Thruster Forces (x y z) in SC{sc_idx+1}")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(8,4))
    t_array = np.linspace(0,best_T,len(u_hist))
    plt.plot(t_array, u_hist[:,0],label='ux')
    plt.plot(t_array, u_hist[:,1],label='uy')
    plt.plot(t_array, u_hist[:,2],label='uz')
    plt.title(f"u(t) SC{sc_idx+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Control")
    plt.grid(True)
    plt.legend()

    # # magnitude
    # plt.figure(figsize=(6,4))
    # u_norm = np.linalg.norm(u_hist,axis=1)
    # plt.plot(t_array,u_norm)
    # plt.title(f"|u(t)| SC{sc_idx+1}")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Control magnitude")
    # plt.grid(True)

# ---------- 2D Top view ----------
plt.figure(figsize=(8,8))
for sc_idx, traj in enumerate(trajectories):
    plt.plot(traj[:,0], traj[:,1], color=colors[sc_idx], label=f'SC{sc_idx+1}')
    plt.scatter(traj[-1,0], traj[-1,1], marker='*', s=150, color=colors[sc_idx])

for phase in phases:
    dp = docking_state(best_T, phase)
    plt.scatter(dp[0], dp[1], marker='x', s=100, color='k')

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("2D Top-Down Trajectories")
plt.grid(True)
plt.legend()



#  Animation 
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
lines = [ax.plot([],[],[],color=colors[i])[0] for i in range(3)]
points = [ax.plot([],[],[],'o',color=colors[i])[0] for i in range(3)]
port_points = [ax.plot([],[],[],'o',color=colors[i],markersize=3)[0] for i in range(3)]

port_trajs = []
frames = steps
ts_anim = np.linspace(0,best_T,frames)
for phase in phases:
    traj_dp = np.zeros((frames,3))
    for i,t in enumerate(ts_anim):
        xf = docking_state(t, phase)
        traj_dp[i] = xf[:3]
    port_trajs.append(traj_dp)

all_thrusters = np.concatenate(thruster_histories)
thruster_norm = np.linalg.norm(all_thrusters,axis=1)
thruster_norm /= thruster_norm.max()


def animate(frame):
    for i in range(3):
        traj = trajectories[i]
        lines[i].set_data(traj[:frame,0], traj[:frame,1])
        lines[i].set_3d_properties(traj[:frame,2])
        color = colors[i]
        lines[i].set_color(color)
        points[i].set_data([traj[frame-1,0]],[traj[frame-1,1]])
        points[i].set_3d_properties([traj[frame-1,2]])
        port_points[i].set_data([port_trajs[i][frame-1,0]],[port_trajs[i][frame-1,1]])
        port_points[i].set_3d_properties([port_trajs[i][frame-1,2]])
    return lines + points + port_points

ani = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
# Keep-out sphere
u,v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
xs = SPHERE_RADIUS*np.cos(u)*np.sin(v)
ys = SPHERE_RADIUS*np.sin(u)*np.sin(v)
zs = SPHERE_RADIUS*np.cos(v)
ax.plot_wireframe(xs, ys, zs, color='k', alpha=0.3, linewidth=0.5)
ax.set_xlim(-2,15)
ax.set_ylim(-2,15)
ax.set_zlim(-2,5)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("3 SCs Docking")
plt.show()





