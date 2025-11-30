#include <stdio.h>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <map>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "kdl/tree.hpp"
#include "kdl/chain.hpp"
#include "kdl/frames.hpp"
#include "kdl/jntarray.hpp"
#include "kdl/chainfksolverpos_recursive.hpp" // For Forward Kinematics
#include "kdl_parser/kdl_parser.hpp"

using std::placeholders::_1;

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
	rclcpp::spin(std::make_shared<FKSolverNode>());
	rclcpp::shutdown();
	return 0;
}