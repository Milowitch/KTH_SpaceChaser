#!/usr/bin/env python3
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Torque-Free Euler Equations
def euler_eq(t, w, I):
    I1, I2, I3 = I
    omega1, omega2, omega3 = w
    domega1 = (I2 - I3)/I1 * omega2*omega3
    domega2 = (I3 - I1)/I2 * omega3*omega1
    domega3 = (I1 - I2)/I3 * omega1*omega2
    return [domega1, domega2, domega3]

# Simulation parameters
t_span = (0, 20)
t_eval = np.linspace(*t_span, 500)

# Symmetric and asymmetric bodies
I_sym = [2.0, 2.0, 1.0]; omega0_sym = [1.0, 0.0, 0.5]
I_asym = [3.0, 2.0, 1.0]; omega0_asym = [1.0, 0.2, 0.5]

# Solve Euler equations for omega vs t plots

sol_sym = solve_ivp(euler_eq, t_span, omega0_sym, t_eval=t_eval, args=(I_sym,))
sol_asym = solve_ivp(euler_eq, t_span, omega0_asym, t_eval=t_eval, args=(I_asym,))

# Plot Angular Velocity Components
plt.figure(figsize=(12,5))
# Asymmetric
plt.subplot(1,2,1)
plt.plot(sol_asym.t, sol_asym.y[0], label=r'$\omega_1$')
plt.plot(sol_asym.t, sol_asym.y[1], label=r'$\omega_2$')
plt.plot(sol_asym.t, sol_asym.y[2], label=r'$\omega_3$')
plt.title('Torque-Free Rotation: Asymmetric Body')
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity [rad/s]')
plt.legend()
# Symmetric
plt.subplot(1,2,2)
plt.plot(sol_sym.t, sol_sym.y[0], label=r'$\omega_1$')
plt.plot(sol_sym.t, sol_sym.y[1], label=r'$\omega_2$')
plt.plot(sol_sym.t, sol_sym.y[2], label=r'$\omega_3$')
plt.title('Torque-Free Rotation: Symmetric Body')
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity [rad/s]')
plt.legend()
plt.tight_layout()
plt.show()

# 3D Trajectory of Angular Velocity
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_sym.y[0], sol_sym.y[1], sol_sym.y[2], label='Symmetric body ω trajectory')
ax.set_xlabel(r'$\omega_1$'); ax.set_ylabel(r'$\omega_2$'); ax.set_zlabel(r'$\omega_3$')
ax.set_title('3D Trajectory of Angular Velocity (Symmetric)')
ax.legend(); plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_asym.y[0], sol_asym.y[1], sol_asym.y[2], label='Asymmetric body ω trajectory')
ax.set_xlabel(r'$\omega_1$'); ax.set_ylabel(r'$\omega_2$'); ax.set_zlabel(r'$\omega_3$')
ax.set_title('3D Trajectory of Angular Velocity (Asymmetric)')
ax.legend(); plt.show()

# Integrate rotation matrix + Euler for animation
def integrate_rotation(I, omega0):
    R0 = np.eye(3).flatten()
    y0 = list(omega0) + list(R0)

    def combined_eq(t, y, I):
        omega = y[:3]
        R = y[3:].reshape((3,3))
        I1, I2, I3 = I
        omega1, omega2, omega3 = omega
        domega = [(I2 - I3)/I1 * omega2*omega3,
                  (I3 - I1)/I2 * omega3*omega1,
                  (I1 - I2)/I3 * omega1*omega2]
        omega_skew = np.array([[0, -omega3, omega2],
                               [omega3, 0, -omega1],
                               [-omega2, omega1, 0]])
        dR = R @ omega_skew
        return domega + list(dR.flatten())

    sol = solve_ivp(combined_eq, t_span, y0, t_eval=t_eval, args=(I,))
    R_all = sol.y[3:].T.reshape(-1,3,3)
    return R_all

R_sym_all = integrate_rotation(I_sym, omega0_sym)
R_asym_all = integrate_rotation(I_asym, omega0_asym)

# Cube for animation
cube_vertices = np.array([[-0.5,-0.5,-0.5],
                          [ 0.5,-0.5,-0.5],
                          [ 0.5, 0.5,-0.5],
                          [-0.5, 0.5,-0.5],
                          [-0.5,-0.5, 0.5],
                          [ 0.5,-0.5, 0.5],
                          [ 0.5, 0.5, 0.5],
                          [-0.5, 0.5, 0.5]])
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

# Animation: cubes + body axes
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
max_val = 1.5
ax.set_xlim([-max_val,max_val]); ax.set_ylim([-max_val,max_val]); ax.set_zlim([-max_val,max_val])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('Symmetric vs Asymmetric Torque-Free Rotation with Body Axes')

# cube edges
lines_sym = [ax.plot([],[],[], 'b')[0] for _ in edges]
lines_asym = [ax.plot([],[],[], 'r')[0] for _ in edges]

# body axes
axis_sym = [ax.plot([],[],[], 'r')[0], ax.plot([],[],[], 'g')[0], ax.plot([],[],[], 'b')[0]]
axis_asym = [ax.plot([],[],[], 'r')[0], ax.plot([],[],[], 'g')[0], ax.plot([],[],[], 'b')[0]]

def update(num):
    R_sym = R_sym_all[num]; R_asym = R_asym_all[num]
    shift_sym = np.array([0.8,0,0]); shift_asym = np.array([-0.8,0,0])
    # Rotate cubes
    rotated_sym = cube_vertices @ R_sym.T + shift_sym
    rotated_asym = cube_vertices @ R_asym.T + shift_asym
    # edges
    for i, (start,end) in enumerate(edges):
        lines_sym[i].set_data(rotated_sym[[start,end],0], rotated_sym[[start,end],1])
        lines_sym[i].set_3d_properties(rotated_sym[[start,end],2])
        lines_asym[i].set_data(rotated_asym[[start,end],0], rotated_asym[[start,end],1])
        lines_asym[i].set_3d_properties(rotated_asym[[start,end],2])
    # axes
    origin_sym = shift_sym; origin_asym = shift_asym
    for i in range(3):
        axis_sym[i].set_data([origin_sym[0], origin_sym[0]+R_sym[0,i]],
                             [origin_sym[1], origin_sym[1]+R_sym[1,i]])
        axis_sym[i].set_3d_properties([origin_sym[2], origin_sym[2]+R_sym[2,i]])
        axis_asym[i].set_data([origin_asym[0], origin_asym[0]+R_asym[0,i]],
                              [origin_asym[1], origin_asym[1]+R_asym[1,i]])
        axis_asym[i].set_3d_properties([origin_asym[2], origin_asym[2]+R_asym[2,i]])
    return lines_sym + lines_asym + axis_sym + axis_asym

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=50, blit=True)
plt.show()