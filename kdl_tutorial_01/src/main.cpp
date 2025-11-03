#include <stdio.h>
#include <string>
#include <vector>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "kdl/tree.hpp"
#include "kdl/chain.hpp"
#include "kdl/frames.hpp"
#include "kdl/jntarray.hpp"
#include "kdl/chainiksolverpos_lma.hpp"
#include "kdl_parser/kdl_parser.hpp"

using std::placeholders::_1;


class MyRobotIK : public rclcpp::Node{
public:
	MyRobotIK() : Node("leg_inverse_kinematics_example"){
        // Initialize joint offset (this offset will be subtracted from IK solutions)
        // joint_offset_ = {0.00454547, 0.87705, 0.4008, 2.04701, -0.00454538};
        joint_offset_ = {0, 0, 0, 0, 0};
        
        // 1. Subscription for URDF (to build the KDL model)
		subscription_ = this->create_subscription<std_msgs::msg::String>(
		"robot_description",
		rclcpp::QoS(rclcpp::KeepLast(1))
			.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL),
		std::bind(&MyRobotIK::robotDescriptionCallback, this, _1));

        // 2. Publisher for Joint Position Commands
        publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/forward_position_controller/commands", 
            10
        );
        RCLCPP_INFO(this->get_logger(), "MyRobotIK Node Initialized with Joint Command Publisher.");
        RCLCPP_INFO(this->get_logger(), "Joint offset configured: [%.5f, %.5f, %.5f, %.5f, %.5f]",
                    joint_offset_[0], joint_offset_[1], joint_offset_[2], 
                    joint_offset_[3], joint_offset_[4]);
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

		// Print basic information about the tree
		std::cout << "nb joints:     " << tree_.getNrOfJoints() << std::endl;
		std::cout << "nb segments:   " << tree_.getNrOfSegments() << std::endl;
		std::cout << "root segment:  " << tree_.getRootSegment()->first << std::endl;

		// Get kinematic chain
		if (!tree_.getChain("base_link", "end_effector", chain_)) {
             RCLCPP_ERROR(this->get_logger(), "Failed to get kinematic chain from 'base_link' to 'end_effector'.");
            return;
        }
		std::cout << "chain nb joints: " << chain_.getNrOfJoints() << std::endl;

		// Create IK solver
		solver_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(chain_);

		// Run usage example
		usageExample();
        
        // Unsubscribe after receiving the URDF once
        subscription_.reset();
	}

	//! Cartesian x, y, z => joint angles via IK
	int getJointAngles(const double x, const double y, const double z, KDL::JntArray& q_out){
		// Prepare IK solver input variables
		KDL::JntArray q_init(chain_.getNrOfJoints());
        // Initialize with zeros or a known safe position
		for (unsigned int i = 0; i < chain_.getNrOfJoints(); ++i) {
            q_init(i) = 0.0;
        }
		
		const KDL::Frame p_in(KDL::Vector(x, y, z));
        RCLCPP_INFO(this->get_logger(), "Running IK solver...");
		// Run IK solver
		return solver_->CartToJnt(q_init, p_in, q_out);
	}

	void publishJointCommand(const KDL::JntArray& joint_angles) {
		auto msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
		
		msg->data.clear();
		
		// Apply offset correction: subtract the offset from each joint angle
		for (unsigned int i = 0; i < joint_angles.data.size(); ++i) {
			double corrected_angle = joint_angles(i) - joint_offset_[i];
			if (i == 3) corrected_angle *= -1;
			msg->data.push_back(corrected_angle);
		}

		// Log the original and corrected commands
		std::stringstream ss_original, ss_corrected;
		ss_original << "IK solution (before offset): [";
		ss_corrected << "Publishing command (after offset): [";
		
		for (size_t i = 0; i < joint_angles.data.size(); ++i) {
			ss_original << joint_angles(i) << (i < joint_angles.data.size() - 1 ? ", " : "");
			ss_corrected << msg->data[i] << (i < msg->data.size() - 1 ? ", " : "");
		}
		ss_original << "]";
		ss_corrected << "]";
		
		RCLCPP_INFO(this->get_logger(), "%s", ss_original.str().c_str());
		RCLCPP_INFO(this->get_logger(), "%s", ss_corrected.str().c_str());

		// Publish the corrected command
		publisher_->publish(std::move(msg));
	}

	void usageExample(){
		KDL::JntArray q_out(chain_.getNrOfJoints());
		int ret;
		
		// Oscillation parameters
		const double x_min = 0.11;
		const double x_max = 0.21;
		const double x_step = 0.01;  // Step size for each iteration
		const double y_pos = 0.00;
		const double z_pos = 0.16;
		
		double x_pos = x_min;
		int direction = 1;  // 1 for increasing, -1 for decreasing
		
		RCLCPP_INFO(this->get_logger(), "Starting oscillation between x=%.2f and x=%.2f", x_min, x_max);
		
		while(true){
			// Calculate and send command for current position
			RCLCPP_INFO(this->get_logger(), "Target position: x=%.3f, y=%.3f, z=%.3f", x_pos, y_pos, z_pos);
			ret = getJointAngles(x_pos, y_pos, z_pos, q_out);
			
			if (ret >= 0) {
				printf("IK Success. Joint angles: %.3f, %.3f, %.3f, %.3f, %.3f\n", 
					   q_out(0), q_out(1), q_out(2), q_out(3), q_out(4));
				publishJointCommand(q_out);
			} else {
				RCLCPP_ERROR(this->get_logger(), "IK failed at x=%.3f (error code: %d)", x_pos, ret);
			}
			
			// Wait before next command
			RCLCPP_INFO(this->get_logger(), "Waiting 2 seconds...");
			rclcpp::sleep_for(std::chrono::seconds(2));
			
			// Update x position
			x_pos += direction * x_step;
			
			// Reverse direction if we've reached the limits
			if (x_pos >= x_max) {
				x_pos = x_max;
				direction = -1;
				RCLCPP_INFO(this->get_logger(), "Reached maximum, reversing direction");
			} else if (x_pos <= x_min) {
				x_pos = x_min;
				direction = 1;
				RCLCPP_INFO(this->get_logger(), "Reached minimum, reversing direction");
			}
		}
	}

	// Class members
	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
	KDL::Tree tree_;
	KDL::Chain chain_;
	std::unique_ptr<KDL::ChainIkSolverPos_LMA> solver_;
    std::vector<double> joint_offset_; // Joint offset to subtract from IK solutions
};

int main(int argc, char* argv[]){
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<MyRobotIK>());
	rclcpp::shutdown();
	return 0;
}