# KTH_SpaceChaser- Collaborative Capture of Tumbling Space Assets in Microgravity Environments
This project simulates **multi-chaser spacecraft docking** in a circular orbit using **Clohessy-Wiltshire (CW) linear dynamics** and **minimum-energy control**.  
The simulation computes optimal docking times, fuel usage, trajectories, and control inputs for multiple spacecraft docking at rotating ports.
---
Keywords: Multi-robot systems, distributed control, MPC, ROS2, Gazebo, ATMOS

## Features

- Compute **optimal docking assignments** using the **Hungarian method**.
- Plot **fuel vs docking time** for each spacecraft and total fuel.
- Visualize **trajectories** in 2D (top view) and 3D.
- Animate spacecraft approaching docking ports in 3D.
- Compute and plot **optimal control input** `u(t)` for each spacecraft.
- Generate **cost matrix heatmaps** for assignment visualization.

---

## Dependencies

- Python 3.8+ , Compiled with Python 3.12.3
- [NumPy](https://numpy.org/)  
- [SciPy](https://www.scipy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Seaborn](https://seaborn.pydata.org/)

Install dependencies via pip:

```bash
pip install numpy scipy matplotlib seaborn
