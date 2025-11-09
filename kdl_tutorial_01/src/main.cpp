#include <stdio.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "kdl/tree.hpp"
#include "kdl/chain.hpp"
#include "kdl/frames.hpp"
#include "kdl/jntarray.hpp"
#include "kdl/chainiksolverpos_lma.hpp"
#include "kdl_parser/kdl_parser.hpp"
#include "kdl/chainfksolverpos_recursive.hpp" // For Forward Kinematics

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using std::placeholders::_1;
using namespace std::chrono_literals;

class MyRobotIK : public rclcpp::Node{
public:
	MyRobotIK() : Node("leg_inverse_kinematics_example"){
        // put joint offsets if any needed
        joint_offset_ = {0, 0, 0, 0, 0};
        
        center_x_ = 0.160;
        center_y_ = 0.000;
        center_z_ = 0.125;
        step_size_ = 0.01; // 10mm steps 
        step_index_ = 0;
        
		subscription_ = this->create_subscription<std_msgs::msg::String>(
		"robot_description",
		rclcpp::QoS(rclcpp::KeepLast(1))
			.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL),
		std::bind(&MyRobotIK::robotDescriptionCallback, this, _1));

        publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/forward_position_controller/commands", 
            10
        );
        RCLCPP_INFO(this->get_logger(), "MyRobotIK Node Initialized with0Joint Command Publisher.");
	}

private:
	void robotDescriptionCallback(const std_msgs::msg::String& msg){
        RCLCPP_INFO(this->get_logger(), "Received robot_description. Building KDL model...");

		// Construct KDL tree from URDF
		const std::string urdf = msg.data;
		if (!kdl_parser::treeFromString(urdf, tree_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF into KDL tree.");
            return;
        }

		// Get kinematic chain
		if (!tree_.getChain("base_link", "end_effector", chain_)) {
             RCLCPP_ERROR(this->get_logger(), "Failed to get kinematic chain from 'base_link' to 'end_effector'.");
            return;
        }
		RCLCPP_INFO(this->get_logger(), "KDL Chain built with %d joints.", chain_.getNrOfJoints());

		// Create IK solver
		solver_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(chain_);

		// Start the step-by-step trajectory timer
		startSquareSequence();
        
        // Unsubscribe after receiving the URDF once
        subscription_.reset();
	}

    void startSquareSequence() {
        // Run at 0.5Hz (2.0 second interval) for a clear, discrete step movement
        double frequency = 0.5;
        auto period = std::chrono::duration<double>(1.0 / frequency);
        
        RCLCPP_INFO(this->get_logger(), "Starting 8-step square pattern at %.2f Hz (%.1f seconds per step).", frequency, 1.0 / frequency);
        RCLCPP_INFO(this->get_logger(), "Center point: (%.4f, %.4f, %.4f). Step size: %.4f m.", center_x_, center_y_, center_z_, step_size_);

        // Create the timer
        timer_ = this->create_wall_timer(
            period,
            std::bind(&MyRobotIK::trajectoryTimerCallback, this)
        );
    }
    
    void trajectoryTimerCallback() {
        if (!solver_) {
            return;
        }

        double x_pos = center_x_;
        double y_pos = center_y_;
        double z_pos = center_z_;
        
        // The 8-step sequence: 
        // 0: +X, 1: Center, 2: +Y, 3: Center, 
        // 4: -X, 5: Center, 6: -Y, 7: Center
        switch (step_index_) {
            case 0: // Move to +X
                x_pos = center_x_ + step_size_;
                break;
            case 1: // Return to Center (x_pos=center_x, y_pos=center_y)
                break;
            case 2: // Move to +Y
                y_pos = center_y_ + step_size_;
                break;
            case 3: // Return to Center
                break;
            case 4: // Move to -X
                x_pos = center_x_ - step_size_;
                break;
            case 5: // Return to Center
                break;
            case 6: // Move to -Y
                y_pos = center_y_ - step_size_;
                break;
            case 7: // Return to Center
                break;
        }

        // 2. Run IK for the new position
		KDL::JntArray q_out(chain_.getNrOfJoints());
        RCLCPP_INFO(this->get_logger(), "Step %d: Target position: x=%.4f, y=%.4f, z=%.4f", step_index_, x_pos, y_pos, z_pos);

		int ret = getJointAngles(x_pos, y_pos, z_pos, q_out);
        
        // 3. Publish the command and update index
		if (ret >= 0) {
			RCLCPP_DEBUG(this->get_logger(), "IK Success. Publishing command.");
			publishJointCommand(q_out);
		} else {
			// IK failed. Log warning but continue to the next step, hoping it resolves the issue.
			RCLCPP_WARN(this->get_logger(), "IK failed at step %d (x=%.4f, y=%.4f, z=%.4f) (error code: %d). Not sending command for this step.", step_index_, x_pos, y_pos, z_pos, ret);
		}

        // Increment and wrap the step index (0, 1, 2, ..., 7, 0, 1, ...)
        step_index_ = (step_index_ + 1) % 8;
    }

	//! Cartesian x, y, z => joint angles via IK
	int getJointAngles(const double x, const double y, const double z, KDL::JntArray& q_out){
		// Prepare IK solver input variables
		KDL::JntArray q_init(chain_.getNrOfJoints());
        // Initialize with a known safe position (or the previous q_out if available)
		for (unsigned int i = 0; i < chain_.getNrOfJoints(); ++i) {
            // Using 0.0 as a safe starting point for initialization
            q_init(i) = 0.0; 
        }
		
		// KDL::Frame represents the desired end-effector pose (position and orientation)
		// We use an identity rotation (default) and set only the position vector
		const KDL::Frame p_in(KDL::Vector(x, y, z));
		
		// Run IK solver
		return solver_->CartToJnt(q_init, p_in, q_out);
	}

	void publishJointCommand(const KDL::JntArray& joint_angles) {
		auto msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
		
		msg->data.clear();
		
		const double upper_safety = 1.57; // ~90 degrees
		const double lower_safety = -M_PI; // ~-180 degrees
		bool limit_exceeded = false;
		
		// Checking limits before adding angle to current command
		for (unsigned int i = 0; i < joint_angles.data.size(); ++i) {
			double corrected_angle = joint_angles(i) - joint_offset_[i];
			if (i == 3) {
				corrected_angle -= M_PI; 
			}
			if (corrected_angle > upper_safety || corrected_angle < lower_safety) {
				RCLCPP_ERROR(this->get_logger(), 
					"🚨 SAFETY TRIP: Joint %u angle %.4f radians exceeds limits (Min: %.2f, Max: %.2f)!", 
					i, corrected_angle, lower_safety, upper_safety);
				limit_exceeded = true;
				break; 
			}
			msg->data.push_back(corrected_angle);
		}
    
		// Kill program if exceed joint limits
		if (limit_exceeded) {
			if (timer_) {
				timer_->cancel();
				RCLCPP_FATAL(this->get_logger(), "!!! MOTION TERMINATED: Joint angle safety limit was breached. !!!");
			}
			return;
		}

		std::stringstream ss_corrected;
		ss_corrected << "Publishing command: [";
		for (size_t i = 0; i < msg->data.size(); ++i) {
			ss_corrected << std::fixed << std::setprecision(4) << msg->data[i] << (i < msg->data.size() - 1 ? ", " : "");
		}
		ss_corrected << "]";
		RCLCPP_DEBUG(this->get_logger(), "%s", ss_corrected.str().c_str());

		publisher_->publish(std::move(msg));
	}

	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_; // Timer for the discrete trajectory
	KDL::Tree tree_;
	KDL::Chain chain_;
	std::unique_ptr<KDL::ChainIkSolverPos_LMA> solver_;
    std::vector<double> joint_offset_; 
    
    double center_x_, center_y_, center_z_; // Center point (0.1102, 0, 0.135)
    double step_size_; // 0.05m
    int step_index_; // Current position in the 8-step cycle (0-7)
};

class FKSolverNode : public rclcpp::Node{
public:
	FKSolverNode() : Node("fk_solver_standalone"){
		// 1. Subscription for URDF (to build the KDL model)
		robot_description_sub_ = this->create_subscription<std_msgs::msg::String>(
			"robot_description",
			rclcpp::QoS(rclcpp::KeepLast(1)).durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL),
			std::bind(&FKSolverNode::robotDescriptionCallback, this, _1)
		);

		// 2. Subscription for Joint States (used for FK)
		joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
			"/joint_states",
			10, // QoS history depth
			std::bind(&FKSolverNode::jointStateCallback, this, _1)
		);

		RCLCPP_INFO(this->get_logger(), "FKSolverNode Initialized. Waiting for /robot_description...");
	}

private:
	void robotDescriptionCallback(const std_msgs::msg::String& msg){
		RCLCPP_INFO(this->get_logger(), "Received robot_description. Building KDL model...");

		// Construct KDL tree from URDF
		const std::string urdf = msg.data;
		if (!kdl_parser::treeFromString(urdf, tree_)) {
			RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF into KDL tree.");
			return;
		}

		// Get kinematic chain from the base to the end_effector
		if (!tree_.getChain("base_link", "end_effector", chain_)) {
				RCLCPP_ERROR(this->get_logger(), "Failed to get kinematic chain from 'base_link' to 'end_effector'.");
			return;
		}
		RCLCPP_INFO(this->get_logger(), "KDL Chain built with %d joints.", chain_.getNrOfJoints());

		// Create FK solver
		fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
		
		// Determine the order of joints KDL expects
		getKDLJointNames();

		// Unsubscribe from robot_description after setup is complete
		robot_description_sub_.reset(); 
	}

	// Helper function to cache the joint names in the order KDL expects them
	void getKDLJointNames() {
		for(unsigned int i = 0; i < chain_.getNrOfSegments(); ++i) {
			const KDL::Joint& joint = chain_.getSegment(i).getJoint();
			if (joint.getType() != KDL::Joint::None) {
				kdl_joint_names_.push_back(joint.getName());
			}
		}
		RCLCPP_INFO(this->get_logger(), "KDL Chain Joint Order established (Total: %zu joints).", kdl_joint_names_.size());
	}

	void jointStateCallback(const sensor_msgs::msg::JointState& msg) {
		if (!fk_solver_) {
			RCLCPP_WARN_ONCE(this->get_logger(), "FK Solver not ready. Waiting for robot_description...");
			return;
		}
		
		// Skip if the message doesn't contain all expected joints
		if (msg.position.size() < kdl_joint_names_.size()) {
				RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
				"JointState message incomplete (%zu positions found, %zu required). Skipping FK.", 
				msg.position.size(), kdl_joint_names_.size());
				return;
		}

		// Map joint names to their positions for fast lookup
		std::map<std::string, double> joint_map;
		for (size_t i = 0; i < msg.name.size(); ++i) {
			if (i < msg.position.size()) {
				joint_map[msg.name[i]] = msg.position[i];
			}
		}

		// 1. Create KDL Joint Array (q_in)
		KDL::JntArray q_in(chain_.getNrOfJoints());
		bool all_joints_found = true;

		for (size_t i = 0; i < kdl_joint_names_.size(); ++i) {
			const std::string& joint_name = kdl_joint_names_[i];
			
			if (joint_map.count(joint_name)) {
				q_in(i) = joint_map[joint_name];
			} else {
				RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
					"Joint position for '%s' not found in /joint_states message. Cannot compute FK.", joint_name.c_str());
				all_joints_found = false;
				break;
			}
		}

		if (all_joints_found) {
			// 2. Run FK Solver
			KDL::Frame end_effector_pose;
			int ret = fk_solver_->JntToCart(q_in, end_effector_pose);

			if (ret >= 0) {
				// 3. Extract and Log Result
				double x = end_effector_pose.p.x();
				double y = end_effector_pose.p.y();
				double z = end_effector_pose.p.z();

				// Extract orientation (e.g., as roll, pitch, yaw)
				double roll, pitch, yaw;
				end_effector_pose.M.GetRPY(roll, pitch, yaw);

				RCLCPP_INFO(this->get_logger(), 
					"--- FK Result (from /joint_states) ---");
				RCLCPP_INFO(this->get_logger(), 
					"Position: (X=%.4f, Y=%.4f, Z=%.4f) m", x, y, z);
				RCLCPP_INFO(this->get_logger(), 
					"Orientation (RPY): (R=%.4f, P=%.4f, Y=%.4f) rad", roll, pitch, yaw);
				RCLCPP_INFO(this->get_logger(), 
					"---------------------------------------");
			} else {
				RCLCPP_ERROR(this->get_logger(), "FK computation failed (error code: %d).", ret);
			}
		}
	}

	// Class members
	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_description_sub_;
	rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
	std::vector<std::string> kdl_joint_names_; 
	std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_; 
	KDL::Tree tree_;
	KDL::Chain chain_;

};

int main(int argc, char* argv[]){
	rclcpp::init(argc, argv);
    
    // Create smart pointers for both nodes
	auto ik_node = std::make_shared<MyRobotIK>();
	auto fk_node = std::make_shared<FKSolverNode>();

    // Create an executor to manage both nodes concurrently
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(ik_node);
    executor.add_node(fk_node);

    // Spin the executor, which blocks and processes callbacks for both nodes
	executor.spin();

	rclcpp::shutdown();
	return 0;
}