import numpy as np
import matplotlib.pyplot as plt

def plot(trajectories, final_success, final_distances, tpe):
    STEP_INCREMENT = 0.005 / np.sqrt(2)

    plt.figure(figsize=(10, 8))
    plt.title(f'XY Trajectories of 10 {tpe}-based Sequences')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.grid(True)

    for i, trajectory in enumerate(trajectories):
        x = 0.0
        y = 0.0
        xy_trajectory = [(x, y)] 

        for action in trajectory:
            if action == 0:
                x += STEP_INCREMENT
                y += STEP_INCREMENT
            elif action == 1:
                x += STEP_INCREMENT
            elif action == 7:
                y += STEP_INCREMENT
            
            xy_trajectory.append((x, y))

        xy_trajectory = np.array(xy_trajectory)
        
        plt.plot(xy_trajectory[:, 0], xy_trajectory[:, 1], 
                label=f'Traj {i+1} (Dist: {final_distances[i]:.4f})', 
                linewidth=1.5)
                
        end_x = xy_trajectory[-1, 0]
        end_y = xy_trajectory[-1, 1]
        if final_success[i]:
            plt.plot(end_x, end_y, 'gx', markersize=10, markeredgewidth=2) 
        else:
            plt.plot(end_x, end_y, 'rx', markersize=6, markeredgewidth=2) 
        
    plt.plot(0, 0, 'go', label='Start Point (0,0)', markersize=8)
    plt.plot([], [], 'rx', markersize=8, markeredgewidth=2, label='Failure - End Point')
    plt.plot([], [], 'gx', markersize=8, markeredgewidth=2, label='Success - End Point')

    plt.legend(loc='lower left', fontsize='small')
    plt.axis('equal') 
    plt.tight_layout()
    plt.savefig(f'{tpe}_xy_trajectories.png')

heuristic_trajectories = [
    [0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 7, 1, 1, 1, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 7, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 1, 1, 7, 7, 7, 7, 7, 7, 7],
    [1, 1, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 1, 7, 7, 7, 7],
    [0, 1, 0, 0, 0, 7, 7, 0, 0, 7, 7, 0, 1, 1, 0, 0, 7, 7, 7, 0],
    [0, 1, 1, 0, 0, 7, 7, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 7, 7, 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 7, 1, 1],
    [0, 1, 1, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 1, 1, 7, 7, 7, 7, 7],
    [0, 1, 1, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0, 1, 0, 0, 7, 7, 0],
    [0, 1, 0, 0, 0, 7, 7, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 1, 1, 7, 7, 7, 7]
]
heur_final_distances = np.array([0.027209, 0.027097, 0.021027, 0.030321, 0.033732, 0.033591, 0.021805, 0.021737, 0.028688, 0.021798])
heur_final_rotations = np.array([-118.564222, -126.417348, -132.145140, -135.344996, -135.004125, -132.330546, -132.435077, -123.363036, -132.879937, -127.612327])
heur_final_success = np.array([False, False, True, True, True, True, True, False, True, False])

print(np.mean(heur_final_distances)) # 0.0267005 
print(np.mean(heur_final_rotations)) # -129.6096754
plot(heuristic_trajectories, heur_final_success, heur_final_distances, "Heuristic")

MPC_trajectories = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
MPC_final_distances = np.array([0.043395, 0.026091, 0.052342, 0.060236, 0.065582, 0.053214, 0.052754, 0.063309, 0.057625, 0.055981])
MPC_final_rotations = np.array([-93.840110, -107.938943, -79.847266, -58.891743, -59.938147, -76.488972, -73.989139, -59.931695, -63.699810, -68.464727])
MPC_final_success = np.array([False, False, False, False, False, False, False, False, False, False])

print(np.mean(MPC_final_distances)) # 0.0530529
print(np.mean(MPC_final_rotations)) # -74.30305519999999
plot(MPC_trajectories, MPC_final_success, MPC_final_distances, "MPC")