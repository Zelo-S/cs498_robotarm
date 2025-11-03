import numpy as np
from math import cos, sin, radians, degrees, atan2, sqrt, acos
from scipy.optimize import minimize

# --- Fixed Link Transformation Matrices ---
T_base_w_fixed = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.058],
    [0, 0, 0, 1]
])

T_w_s_fixed = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0.0485],
    [0, 0, 0, 1]
])

T_s_e_fixed = np.array([
    [1, 0, 0, -0.045182],
    [0, -1, 0, -0.129473],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

T_e_r1_fixed = np.array([
    [0, -1, 0, 0.113375],
    [-1, 0, 0, -0.127157],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

T_r1_r2_fixed = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

T_r2_ee_fixed = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.05],
    [0, 0, 0, 1]
])

fixed_transforms = [
    T_base_w_fixed,
    T_w_s_fixed,
    T_s_e_fixed,
    T_e_r1_fixed,
    T_r1_r2_fixed
]

# --- Helper Functions ---
def rotz(theta):
    """Homogeneous transformation matrix for rotation around Z-axis."""
    c = cos(theta)
    s = sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def forward_kinematics(thetas):
    """
    Compute forward kinematics given joint angles.
    
    Args:
        thetas: List of 5 joint angles in radians
    
    Returns:
        4x4 transformation matrix from base to end-effector
    """
    T_chain = np.identity(4)
    
    for i in range(5):
        T_link_fixed = fixed_transforms[i]
        T_variable_joint = rotz(thetas[i])
        T_i_to_iplus1 = T_link_fixed @ T_variable_joint
        T_chain = T_chain @ T_i_to_iplus1
    
    T_base_ee = T_chain @ T_r2_ee_fixed
    return T_base_ee

def rotation_error(R_current, R_desired):
    """
    Calculate rotation error between two rotation matrices.
    Returns a scalar error metric.
    """
    R_error = R_desired.T @ R_current
    # Trace-based error metric
    trace = np.trace(R_error)
    error = 1 - (trace - 1) / 2  # Normalized to [0, 2]
    return error

def inverse_kinematics_numerical(target_x, target_y, fixed_z_k, target_rot=None, 
                                  initial_guess=None, 
                                  position_weight=1.0, orientation_weight=1.0):
    """
    Numerical inverse kinematics with FIXED z-coordinate constraint.
    
    Args:
        target_x: Desired end-effector x position
        target_y: Desired end-effector y position
        fixed_z_k: FIXED z-coordinate (constraint)
        target_rot: Desired 3x3 rotation matrix (optional)
        initial_guess: Initial joint angles (5 values in radians)
        position_weight: Weight for position error (x, y only)
        orientation_weight: Weight for orientation error
    
    Returns:
        solution: Dictionary containing joint angles and error metrics
    """
    if initial_guess is None:
        initial_guess = np.zeros(5)
    
    target_pos = np.array([target_x, target_y, fixed_z_k])
    
    def objective(thetas):
        """Objective function to minimize."""
        T = forward_kinematics(thetas)
        current_pos = T[:3, 3]
        
        # Position error - EMPHASIZE z-constraint heavily
        x_error = (current_pos[0] - target_x)**2
        y_error = (current_pos[1] - target_y)**2
        z_error = (current_pos[2] - fixed_z_k)**2 * 100  # Heavy penalty for z deviation
        
        pos_error = sqrt(x_error + y_error + z_error)
        
        # Orientation error (if target rotation is provided)
        if target_rot is not None:
            current_rot = T[:3, :3]
            rot_error = rotation_error(current_rot, target_rot)
            total_error = position_weight * pos_error + orientation_weight * rot_error
        else:
            total_error = pos_error
        
        return total_error
    
    # Joint limits (adjust based on your robot's specs)
    bounds = [(-np.pi, np.pi) for _ in range(5)]
    
    # Add constraint that z must equal fixed_z_k
    def z_constraint(thetas):
        T = forward_kinematics(thetas)
        return T[2, 3] - fixed_z_k
    
    constraints = {'type': 'eq', 'fun': z_constraint}
    
    # Optimize
    result = minimize(objective, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Verify solution
    T_result = forward_kinematics(result.x)
    final_pos = T_result[:3, 3]
    
    # Calculate errors
    xy_error = sqrt((final_pos[0] - target_x)**2 + (final_pos[1] - target_y)**2)
    z_error = abs(final_pos[2] - fixed_z_k)
    total_pos_error = np.linalg.norm(final_pos - target_pos)
    
    solution = {
        'joint_angles_rad': result.x,
        'joint_angles_deg': np.degrees(result.x),
        'success': result.success,
        'xy_error': xy_error,
        'z_error': z_error,
        'total_position_error': total_pos_error,
        'final_position': final_pos,
        'target_position': target_pos,
        'iterations': result.nit,
        'transformation_matrix': T_result,
        'z_constraint_satisfied': z_error < 0.001  # 1mm tolerance
    }
    
    if target_rot is not None:
        solution['rotation_error'] = rotation_error(T_result[:3, :3], target_rot)
    
    return solution

def inverse_kinematics_analytical_partial(target_x, target_y):
    """
    Analytical solution for the first joint (waist rotation).
    This is a partial solution - remaining joints solved numerically.
    
    Args:
        target_x: Desired end-effector x position
        target_y: Desired end-effector y position
    
    Returns:
        theta_1: Waist rotation angle in radians
    """
    # For a typical waist joint, theta_1 orients the arm toward the target
    theta_1 = atan2(target_y, target_x)
    
    return theta_1

def inverse_kinematics_hybrid(target_x, target_y, fixed_z_k, target_rot=None):
    """
    Hybrid approach: Solve theta_1 analytically, then use numerical for others.
    Z-coordinate is FIXED at fixed_z_k.
    
    Args:
        target_x: Desired end-effector x position
        target_y: Desired end-effector y position
        fixed_z_k: FIXED z-coordinate (constraint)
        target_rot: Desired 3x3 rotation matrix (optional)
    
    Returns:
        solution: Dictionary containing joint angles and error metrics
    """
    # Solve first joint analytically
    theta_1 = inverse_kinematics_analytical_partial(target_x, target_y)
    
    # Use this as initial guess for numerical solver
    initial_guess = np.array([theta_1, 0, 0, 0, 0])
    
    return inverse_kinematics_numerical(target_x, target_y, fixed_z_k, 
                                       target_rot, initial_guess)

def inverse_kinematics_planar(target_x, target_y, fixed_z_k, target_rot=None,
                              num_solutions=5):
    """
    Find multiple IK solutions for planar (fixed z) motion.
    Returns the best solution from multiple random initial guesses.
    
    Args:
        target_x: Desired end-effector x position
        target_y: Desired end-effector y position
        fixed_z_k: FIXED z-coordinate (constraint)
        target_rot: Desired 3x3 rotation matrix (optional)
        num_solutions: Number of random initial guesses to try
    
    Returns:
        best_solution: Dictionary containing the best joint angles and error metrics
    """
    solutions = []
    
    # Try hybrid approach first
    sol_hybrid = inverse_kinematics_hybrid(target_x, target_y, fixed_z_k, target_rot)
    solutions.append(sol_hybrid)
    
    # Try multiple random initial guesses
    for _ in range(num_solutions - 1):
        initial_guess = np.random.uniform(-np.pi, np.pi, 5)
        sol = inverse_kinematics_numerical(target_x, target_y, fixed_z_k, 
                                          target_rot, initial_guess)
        solutions.append(sol)
    
    # Find best solution (lowest error)
    best_solution = min(solutions, key=lambda s: s['total_position_error'])
    best_solution['num_attempts'] = num_solutions
    
    return best_solution

# --- Example Usage ---
if __name__ == "__main__":
    print("="*60)
    print("5-DOF Robot IK with FIXED Z-Coordinate Constraint")
    print("="*60)
    
    # Define the fixed z-coordinate
    K = 0.20  # Fixed z-coordinate in meters
    
    # Example 1: Forward kinematics test
    print(f"\n--- Example 1: Forward Kinematics Test ---")
    print(f"Fixed Z-coordinate (k): {K} m")
    test_angles = [radians(30), radians(45), radians(-20), radians(10), radians(15)]
    T_fk = forward_kinematics(test_angles)
    fk_position = T_fk[:3, 3]
    target_rotation = T_fk[:3, :3]
    
    print(f"Test joint angles (deg): {[round(degrees(a), 2) for a in test_angles]}")
    print(f"FK result position: {fk_position.round(4)}")
    print(f"FK result z: {fk_position[2]:.4f} m (not necessarily equal to k)")
    
    # Example 2: IK with fixed z-coordinate (position only)
    print(f"\n--- Example 2: IK with Fixed Z = {K} m (Position Only) ---")
    target_x, target_y = 0.15, 0.10
    
    ik_solution = inverse_kinematics_numerical(target_x, target_y, K)
    
    print(f"Target: x={target_x}, y={target_y}, z={K} (FIXED)")
    print(f"IK Success: {ik_solution['success']}")
    print(f"Z-constraint satisfied: {ik_solution['z_constraint_satisfied']}")
    print(f"Solution angles (deg): {ik_solution['joint_angles_deg'].round(2)}")
    print(f"XY error: {ik_solution['xy_error']:.6f} m")
    print(f"Z error: {ik_solution['z_error']:.6f} m")
    print(f"Total position error: {ik_solution['total_position_error']:.6f} m")
    print(f"Final position: {ik_solution['final_position'].round(4)}")
    
    # Example 3: IK with orientation
    print(f"\n--- Example 3: IK with Fixed Z + Orientation ---")
    
    # Define a desired orientation (e.g., pointing downward)
    desired_rotation = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    ik_solution_rot = inverse_kinematics_numerical(
        target_x, target_y, K,
        target_rot=desired_rotation,
        orientation_weight=0.5
    )
    
    print(f"Target: x={target_x}, y={target_y}, z={K} (FIXED)")
    print(f"IK Success: {ik_solution_rot['success']}")
    print(f"Z-constraint satisfied: {ik_solution_rot['z_constraint_satisfied']}")
    print(f"Solution angles (deg): {ik_solution_rot['joint_angles_deg'].round(2)}")
    print(f"XY error: {ik_solution_rot['xy_error']:.6f} m")
    print(f"Z error: {ik_solution_rot['z_error']:.6f} m")
    print(f"Rotation error: {ik_solution_rot['rotation_error']:.6f}")
    
    # Example 4: Hybrid approach
    print(f"\n--- Example 4: Hybrid IK Approach ---")
    ik_hybrid = inverse_kinematics_hybrid(target_x, target_y, K)
    
    print(f"Target: x={target_x}, y={target_y}, z={K} (FIXED)")
    print(f"IK Success: {ik_hybrid['success']}")
    print(f"Z-constraint satisfied: {ik_hybrid['z_constraint_satisfied']}")
    print(f"Solution angles (deg): {ik_hybrid['joint_angles_deg'].round(2)}")
    print(f"XY error: {ik_hybrid['xy_error']:.6f} m")
    print(f"Z error: {ik_hybrid['z_error']:.6f} m")
    
    # Example 5: Planar motion (multiple solutions)
    print(f"\n--- Example 5: Find Best Solution (Multiple Attempts) ---")
    target_x2, target_y2 = 0.12, 0.08
    
    ik_planar = inverse_kinematics_planar(target_x2, target_y2, K, num_solutions=10)
    
    print(f"Target: x={target_x2}, y={target_y2}, z={K} (FIXED)")
    print(f"Number of attempts: {ik_planar['num_attempts']}")
    print(f"Best IK Success: {ik_planar['success']}")
    print(f"Z-constraint satisfied: {ik_planar['z_constraint_satisfied']}")
    print(f"Best solution angles (deg): {ik_planar['joint_angles_deg'].round(2)}")
    print(f"XY error: {ik_planar['xy_error']:.6f} m")
    print(f"Z error: {ik_planar['z_error']:.6f} m")
    print(f"Final position: {ik_planar['final_position'].round(4)}")
    
    # Example 6: Workspace exploration at fixed z
    print(f"\n--- Example 6: Workspace Points at Fixed Z = {K} m ---")
    test_points = [
        (0.10, 0.05),
        (0.15, 0.00),
        (0.12, -0.08),
        (0.08, 0.10)
    ]
    
    for x, y in test_points:
        sol = inverse_kinematics_hybrid(x, y, K)
        status = "✓" if sol['z_constraint_satisfied'] else "✗"
        print(f"{status} Target ({x:.2f}, {y:.2f}, {K:.2f}): " +
              f"Error = {sol['total_position_error']:.6f} m, " +
              f"Z-error = {sol['z_error']:.6f} m")
    
    print("\n" + "="*60)