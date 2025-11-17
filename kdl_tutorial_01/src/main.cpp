#include <stdio.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <random>

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
        
		curr_x_ = 0.160;
		curr_y_ = 0.000;
		curr_z_ = 0.130;
        step_size_ = 0.02; // 20mm steps 
		step_size_diag_ = 0.0141421356;
        step_index_ = 0;
		
        
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
        
        RCLCPP_INFO(this->get_logger(), "8-step pattern at %.2f Hz (%.1f seconds per step).", frequency, 1.0 / frequency);
        RCLCPP_INFO(this->get_logger(), "Starting loc: (%.4f, %.4f, %.4f). Step size: %.4f m.", curr_x_, curr_y_, curr_z_, step_size_);

        timer_ = this->create_wall_timer(
            period,
            std::bind(&MyRobotIK::trajectoryTimerCallback, this)
        );

        // timer2_ = this->create_wall_timer(
            // period,
            // std::bind(&MyRobotIK::trajectoryTimerCallback, this)
        // );
    }
    
	void trajectoryTimerCallback() {
		if (!solver_) {
			return;
		}

		if (sub_step_index_ == 0) {
			double x_pos = curr_x_;
			double y_pos = curr_y_;
			double z_pos = curr_z_;
			
			switch (step_index_) {
				case 0: x_pos = curr_x_ + step_size_; break;
				case 1: x_pos = curr_x_ + step_size_diag_; y_pos = curr_y_ + step_size_diag_; break;
				case 2: y_pos = curr_y_ + step_size_; break;
				case 3: x_pos = curr_x_ - step_size_diag_; y_pos = curr_y_ + step_size_diag_; break;
				case 4: x_pos = curr_x_ - step_size_; break;
				case 5: x_pos = curr_x_ - step_size_diag_; y_pos = curr_y_ - step_size_diag_; break;
				case 6: y_pos = curr_y_ - step_size_; break;
				case 7: x_pos = curr_x_ + step_size_diag_; y_pos = curr_y_ - step_size_diag_; break;
			}
			
			traj_target_x_ = x_pos;
			traj_target_y_ = y_pos;
			traj_target_z_ = z_pos;
			
			traj_start_x_ = curr_x_;
			traj_start_y_ = curr_y_;
			traj_start_z_ = curr_z_;

		}
		

		double t_normalized = (double)sub_step_index_ / (double)total_sub_steps_;

		double interp_x = traj_start_x_ + t_normalized * (traj_target_x_ - traj_start_x_);
		double interp_y = traj_start_y_ + t_normalized * (traj_target_y_ - traj_start_y_);
		double interp_z = traj_start_z_ + t_normalized * (traj_target_z_ - traj_start_z_);
		
		KDL::JntArray q_current(chain_.getNrOfJoints());
		int ret = getJointAngles(interp_x, interp_y, interp_z, q_current);

		if (ret >= 0) {
			RCLCPP_INFO(this->get_logger(), "On trajectory step_index_: %d", step_index_);
			publishJointCommand(q_current);
		} else {
			RCLCPP_WARN(this->get_logger(), "IK failed for step index: %d", step_index_);
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> distrib(0, 7);
			step_index_ = distrib(gen);
			return;
		}

		sub_step_index_++;
		
		if (sub_step_index_ > total_sub_steps_) {
			sub_step_index_ = 0; 
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> distrib(0, 7);
			step_index_ = distrib(gen);
			RCLCPP_INFO(this->get_logger(), "Stepping in NEW direction %d", step_index_);
			curr_x_ = traj_target_x_;
			curr_y_ = traj_target_y_;
			curr_z_ = traj_target_z_;
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
    rclcpp::TimerBase::SharedPtr timer_; // Timer for the 8-step sequence
	KDL::Tree tree_;
	KDL::Chain chain_;
	std::unique_ptr<KDL::ChainIkSolverPos_LMA> solver_;
    std::vector<double> joint_offset_; 
    
    double step_size_; // 0.02
    double step_size_diag_; // 0.014
    int step_index_; // Current position in the 8-step cycle (0-15)
	
	double curr_x_, curr_y_, curr_z_;
	double traj_target_x_, traj_target_y_, traj_target_z_;
	double traj_start_x_, traj_start_y_, traj_start_z_;
	int sub_step_index_;                            // Current step index within the 100 steps
	const int total_sub_steps_ = 100;               // Total steps for interpolation
};

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