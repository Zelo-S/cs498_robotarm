import numpy as np
from math import cos, sin, radians

# --- 1. Define Joint Variables (Thetas) ---
# We assume 5 revolute joints, and the transformation T_r2_ee is fixed.
# Assign initial dummy values (e.g., all 0 degrees) for demonstration.
theta_1 = radians(0)  # Joint 1: Waist rotation
theta_2 = radians(-90) # Joint 2: Shoulder pitch
theta_3 = radians(0)  # Joint 3: Elbow pitch
theta_4 = radians(0)  # Joint 4: Wrist roll
theta_5 = radians(0) # Joint 5: Wrist pitch

# A list of thetas for easier iteration
thetas = [theta_1, theta_2, theta_3, theta_4, theta_5]

# --- 2. Define General Z-Rotation Function ---
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

# --- 3. Fixed Link Transformation Matrices (from your image) ---
# These are the geometric constants (link lengths, offsets, and frame alignments).
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

# The last link (r2 -> ee) is usually fixed to the tool, so no variable joint is needed here.
T_r2_ee_fixed = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.05],
    [0, 0, 0, 1]
])

# List of fixed transformations for iteration
fixed_transforms = [
    T_base_w_fixed,
    T_w_s_fixed,
    T_s_e_fixed,
    T_e_r1_fixed,
    T_r1_r2_fixed
]

# --- 4. Combine Variable and Fixed Transformations ---
T_chain = np.identity(4)

for i in range(5):
    # 1. Create the variable rotation matrix for the current joint angle
    T_variable_joint = rotz(thetas[i])
    
    # 2. Get the fixed transformation for the link
    T_link_fixed = fixed_transforms[i]
    
    # 3. The transformation for the current link is:
    # T_i-1_to_i = (Variable Rotation) @ (Fixed Link Geometry)
    T_i_to_iplus1 =  T_link_fixed @ T_variable_joint
    
    # 4. Multiply with the running chain transformation
    T_chain = T_chain @ T_i_to_iplus1

# --- 5. Add the Final End-Effector Fixed Transform ---
T_base_ee = T_chain @ T_r2_ee_fixed


# --- 6. Print Results ---
print(f"--- 5-DOF Forward Kinematics Solution ---")
print(f"Joint Angles (Degrees): {[round(np.degrees(t), 2) for t in thetas]}")
print("="*40)

print("\nOverall Transformation Matrix (T_base_ee):\n")
print(T_base_ee.round(4))

# Extract final position vector (P) and Rotation Matrix (R)
P = T_base_ee[:3, 3]

print("\nEnd-Effector Position (P) in Base Frame (x, y, z):\n", P.round(4))
print("\nEnd-Effector Orientation R Matrix in Base Frame (Top-Left 3x3):\n", T_base_ee[:3, :3].round(4))