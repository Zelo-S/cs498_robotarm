#include <stdio.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath> // Include for sin() and M_PI

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

// Define M_PI if it's not defined (common in some standard libraries)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using std::placeholders::_1;
using namespace std::chrono_literals;

class MyRobotIK : public rclcpp::Node{
public:
	MyRobotIK() : Node("leg_inverse_kinematics_example"){
        // Initialize joint offset
        joint_offset_ = {0, 0, 0, 0, 0};
        // Initialize start time for trajectory calculation
        start_time_ = this->now();
       
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
		std::cout << "Chain nb joints: " << chain_.getNrOfJoints() << std::endl;

		// Create IK solver
		// LMA is a good choice for fast, robust IK
		solver_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(chain_);

		// Start the smooth trajectory timer
		startSmoothTrajectory();
       
        // Unsubscribe after receiving the URDF once
        subscription_.reset();
	}

    void startSmoothTrajectory() {
        // Run at 50Hz (20ms interval) for a smoother trajectory
        double frequency = 50.0;
        auto period = std::chrono::duration<double>(1.0 / frequency);
        
        // Initialize trajectory parameters
        x_min_ = 0.11;
        x_max_ = 0.21;
        y_pos_ = 0.00;
        z_pos_ = 0.155;
        amplitude_ = (x_max_ - x_min_) / 2.0; // Amplitude of the oscillation
        offset_ = x_min_ + amplitude_;      // Center point
        period_sec_ = 4.0;                  // Time for one full back-and-forth cycle (4 seconds)
        
        RCLCPP_INFO(this->get_logger(), "Starting smooth sine wave oscillation at %.2f Hz.", frequency);
        RCLCPP_INFO(this->get_logger(), "Oscillating x between %.3f and %.3f over %.1f seconds.", x_min_, x_max_, period_sec_);

        // Create the timer
        timer_ = this->create_wall_timer(
            period,
            std::bind(&MyRobotIK::trajectoryTimerCallback, this)
        );
    }
    
    void trajectoryTimerCallback() {
        if (!solver_) {
            // Wait until the KDL model is initialized
            return;
        }

        // 1. Calculate the current time and position
        rclcpp::Duration elapsed_time = this->now() - start_time_;
        double time_s = elapsed_time.seconds();
        
        // Use a sine wave to create a smooth, repeating oscillation:
        // x(t) = Offset + Amplitude * sin(2*PI*t / Period)
        double x_pos = offset_ + amplitude_ * std::cos((2.0 * M_PI * time_s) / period_sec_);

        // 2. Run IK for the new position
		KDL::JntArray q_out(chain_.getNrOfJoints());
        RCLCPP_INFO(this->get_logger(), "Target position: x=%.4f, y=%.4f, z=%.4f", x_pos, y_pos_, z_pos_);

		int ret = getJointAngles(x_pos, y_pos_, z_pos_, q_out);
        
        // 3. Publish the command
		if (ret >= 0) {
			RCLCPP_DEBUG(this->get_logger(), "IK Success. Publishing command.");
			publishJointCommand(q_out);
		} else {
			// IK failed, likely due to a singular position or being out of reach
			RCLCPP_WARN(this->get_logger(), "IK failed at x=%.4f (error code: %d). Not sending command.", x_pos, ret);
		}
    }

	//! Cartesian x, y, z => joint angles via IK
	int getJointAngles(const double x, const double y, const double z, KDL::JntArray& q_out){
		// Prepare IK solver input variables
		KDL::JntArray q_init(chain_.getNrOfJoints());
        // Initialize with current or a known safe position (a better choice here might be the previous q_out)
		for (unsigned int i = 0; i < chain_.getNrOfJoints(); ++i) {
            q_init(i) = 0.0;
        }
		
		const KDL::Frame p_in(KDL::Vector(x, y, z));
		// Run IK solver
		// Using the LMA solver's overload with no explicit orientation (identity rotation)
		return solver_->CartToJnt(q_init, p_in, q_out);
	}

	void publishJointCommand(const KDL::JntArray& joint_angles) {
		auto msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
		
		msg->data.clear();
		
		const double upper_safety = 1.57; 
		const double lower_safety = -3.15; 
		bool limit_exceeded = false;
		
		// 1. Calculate and Check Limits
		for (unsigned int i = 0; i < joint_angles.data.size(); ++i) {
			double corrected_angle = joint_angles(i) - joint_offset_[i];
			
			if (i == 3) {
				corrected_angle -= 3.1415; 
			}
			
			// Check the corrected angle against the safety limit (in magnitude)
			if (corrected_angle > upper_safety) {
				RCLCPP_ERROR(this->get_logger(), 
					"🚨 SAFETY TRIP: Joint %u angle %.4f radians exceeds limit of %.2f radians!", 
					i, corrected_angle, upper_safety);
				limit_exceeded = true;
				break; // Stop processing angles
			}else if (corrected_angle < lower_safety) {
				RCLCPP_ERROR(this->get_logger(), 
					"🚨 SAFETY TRIP: Joint %u angle %.4f radians exceeds limit of %.2f radians!", 
					i, corrected_angle, lower_safety);
				limit_exceeded = true;
				break; // Stop processing angles
			}
			
			msg->data.push_back(corrected_angle);
		}
    
		// 2. Termination Logic
		if (limit_exceeded) {
			// Stop the smooth trajectory by resetting the timer
			if (timer_) {
				timer_->cancel();
				RCLCPP_FATAL(this->get_logger(), "!!! MOTION TERMINATED: Joint angle safety limit was breached. !!!");
			}
			// Do NOT publish the command
			return;
		}

		// 3. Publish the corrected command (only if limits were not exceeded)
		
		// Log the command (Optional, kept for debugging context)
		std::stringstream ss_corrected;
		ss_corrected << "Publishing command: [";
		for (size_t i = 0; i < msg->data.size(); ++i) {
			ss_corrected << std::fixed << std::setprecision(4) << msg->data[i] << (i < msg->data.size() - 1 ? ", " : "");
		}
		ss_corrected << "]";
		RCLCPP_DEBUG(this->get_logger(), "%s", ss_corrected.str().c_str());

		publisher_->publish(std::move(msg));
	}
    // Removed usageExample() as its logic is now in startSmoothTrajectory/trajectoryTimerCallback

	// Class members
	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_; // Timer for the smooth trajectory
	KDL::Tree tree_;
	KDL::Chain chain_;
	std::unique_ptr<KDL::ChainIkSolverPos_LMA> solver_;
    std::vector<double> joint_offset_; 
    
    // Trajectory state variables
    rclcpp::Time start_time_;
    double x_min_, x_max_, y_pos_, z_pos_;
    double amplitude_;
    double offset_;
    double period_sec_; // Full cycle period

};

int main(int argc, char* argv[]){
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<MyRobotIK>());
	rclcpp::shutdown();
	return 0;
}