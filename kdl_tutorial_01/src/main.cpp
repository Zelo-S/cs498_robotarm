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
#include "kdl/chainfksolverpos_recursive.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using std::placeholders::_1;
using namespace std::chrono_literals;

class MyRobotIK : public rclcpp::Node{
public:
	MyRobotIK() : Node("leg_inverse_kinematics_example"){
        joint_offset_ = {0, 0, 0, 0, 0};
        
        center_x_ = 0.160;
        center_y_ = 0.000;
        center_z_ = 0.125;
        step_size_ = 0.02; // 20mm steps 
		step_size_diag_ = 0.0141421356;
        step_index_ = 0;
		
		curr_x_ = center_x_;
		curr_y_ = center_y_;
		curr_z_ = center_z_;
        
		sub_step_index_ = 0;
		
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

		const std::string urdf = msg.data;
		if (!kdl_parser::treeFromString(urdf, tree_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF into KDL tree.");
            return;
        }

		if (!tree_.getChain("base_link", "end_effector", chain_)) {
             RCLCPP_ERROR(this->get_logger(), "Failed to get kinematic chain from 'base_link' to 'end_effector'.");
            return;
        }
		RCLCPP_INFO(this->get_logger(), "KDL Chain built with %d joints.", chain_.getNrOfJoints());

		solver_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(chain_);

		runActionSpaceSequence();
        
        subscription_.reset();
	}

    void runActionSpaceSequence() {
		double total_move_time = 2.0; 
		double frequency = (double)total_sub_steps_ / total_move_time;
        auto period = std::chrono::duration<double>(1.0 / frequency);
        
        RCLCPP_INFO(this->get_logger(), "16-step pattern at %.2f Hz (%.1f seconds per step).", frequency, 1.0 / frequency);
        RCLCPP_INFO(this->get_logger(), "Center point loc: (%.4f, %.4f, %.4f). Step size: %.4f m.", center_x_, center_y_, center_z_, step_size_);

        timer_ = this->create_wall_timer(
            period,
            std::bind(&MyRobotIK::trajectoryTimerCallback, this)
        );
    }
    
	void trajectoryTimerCallback() {
		if (!solver_) {
			return;
		}

		if (sub_step_index_ == 0) {
			double x_pos = center_x_;
			double y_pos = center_y_;
			double z_pos = center_z_;
			
			switch (step_index_) {
				case 0: x_pos = center_x_ + step_size_; break;
				case 1: break;
				case 2: x_pos = center_x_ + step_size_diag_; y_pos = center_y_ + step_size_diag_; break;
				case 3: break;
				case 4: y_pos = center_y_ + step_size_; break;
				case 5: break;
				case 6: x_pos = center_x_ - step_size_diag_; y_pos = center_y_ + step_size_diag_; break;
				case 7: break;
				case 8: x_pos = center_x_ - step_size_; break;
				case 9: break;
				case 10: x_pos = center_x_ - step_size_diag_; y_pos = center_y_ - step_size_diag_; break;
				case 11: break;
				case 12: y_pos = center_y_ - step_size_; break;
				case 13: break;
				case 14: x_pos = center_x_ + step_size_diag_; y_pos = center_y_ - step_size_diag_; break;
				case 15: break;
			}
			
			KDL::JntArray q_start(chain_.getNrOfJoints());
			KDL::JntArray q_target(chain_.getNrOfJoints());

			int ret_start = getJointAngles(curr_x_, curr_y_, curr_z_, q_start);
			int ret_target = getJointAngles(x_pos, y_pos, z_pos, q_target);

			if (ret_start >= 0 && ret_target >= 0) {
				RCLCPP_INFO(this->get_logger(), "--- MACRO STEP %d ---", step_index_);
				RCLCPP_INFO(this->get_logger(), "Starting new trajectory from (%.4f, %.4f, %.4f) to (%.4f, %.4f, %.4f)",
							curr_x_, curr_y_, curr_z_, x_pos, y_pos, z_pos);

				interpolateJoints(q_start, q_target, total_sub_steps_, current_trajectory_);

				curr_x_ = x_pos;
				curr_y_ = y_pos;
				curr_z_ = z_pos;
			} else {
				RCLCPP_WARN(this->get_logger(), "IK failed for start (%d) or target (%d). Skipping macro step.", ret_start, ret_target);
				step_index_ = (step_index_ + 1) % 16;
				return;
			}
		}

		if (!current_trajectory_.empty()) {
			const KDL::JntArray& q_step = current_trajectory_[sub_step_index_];
			publishJointCommand(q_step);
		}

		sub_step_index_++;
		
		if (sub_step_index_ > total_sub_steps_) {
			sub_step_index_ = 0; 
			step_index_ = (step_index_ + 1) % 16; 
		}
	}
	
	void interpolateJoints(const KDL::JntArray& q_start, const KDL::JntArray& q_end, int total_steps, std::vector<KDL::JntArray>& trajectory) {

		trajectory.clear();

		unsigned int n_joints = q_start.data.size();

		KDL::JntArray delta_q = q_end;
		KDL::Subtract(q_end, q_start, delta_q);

		for (int i = 0; i <= total_steps; ++i) {
			double t_normalized = (double)i / (double)total_steps;
			KDL::JntArray q_step(n_joints);

			for (unsigned int j = 0; j < n_joints; ++j) {
				q_step(j) = q_start(j) + t_normalized * delta_q(j);
			}
			trajectory.push_back(q_step);
		}
		
	}

	int getJointAngles(const double x, const double y, const double z, KDL::JntArray& q_out){
		KDL::JntArray q_init(chain_.getNrOfJoints());
		for (unsigned int i = 0; i < chain_.getNrOfJoints(); ++i) {
            q_init(i) = 0.0; 
        }
		
		const KDL::Frame p_in(KDL::Vector(x, y, z));
		return solver_->CartToJnt(q_init, p_in, q_out);
	}

	void publishJointCommand(const KDL::JntArray& joint_angles) {
		auto msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
		
		msg->data.clear();
		
		const double upper_safety = 1.57; // ~90 degrees
		const double lower_safety = -M_PI; // ~-180 degrees
		bool limit_exceeded = false;
		
		for (unsigned int i = 0; i < joint_angles.data.size(); ++i) {
			double corrected_angle = joint_angles(i) - joint_offset_[i];
			if (i == 3) {
				corrected_angle -= M_PI; 
			}
			if (corrected_angle > upper_safety || corrected_angle < lower_safety) {
				RCLCPP_ERROR(this->get_logger(), 
					"!!! Joint %u angle %.4f radians exceeds limits (Min: %.2f, Max: %.2f) !!!", 
					i, corrected_angle, lower_safety, upper_safety);
				limit_exceeded = true;
				break; 
			}
			msg->data.push_back(corrected_angle);
		}
    
		if (limit_exceeded) {
			if (timer_) {
				timer_->cancel();
				RCLCPP_FATAL(this->get_logger(), "!!! Joint angle safety limit was breached. !!!");
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
    rclcpp::TimerBase::SharedPtr timer_; // Timer for the 16-step sequence
	KDL::Tree tree_;
	KDL::Chain chain_;
	std::unique_ptr<KDL::ChainIkSolverPos_LMA> solver_;
    std::vector<double> joint_offset_; 
    
    double center_x_, center_y_, center_z_; // Center point (0.1102, 0, 0.135)
    double step_size_; // 0.02
    double step_size_diag_; // 0.014
    int step_index_; // Current position in the 16-step cycle (0-15)
	
	double curr_x_, curr_y_, curr_z_;
	std::vector<KDL::JntArray> current_trajectory_; // Stores the 100 joint states
	int sub_step_index_;                            // Current step index within the 100 steps
	const int total_sub_steps_ = 100;               // Total steps for interpolation
};

/*class FKSolverNode : public rclcpp::Node{
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

				/*RCLCPP_INFO(this->get_logger(), 
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
};*/

int main(int argc, char* argv[]){
	rclcpp::init(argc, argv);
    
	auto ik_node = std::make_shared<MyRobotIK>();
	// auto fk_node = std::make_shared<FKSolverNode>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(ik_node);
    // executor.add_node(fk_node);
	executor.spin();

	rclcpp::shutdown();
	return 0;
}