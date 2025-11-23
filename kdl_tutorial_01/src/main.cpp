#include <stdio.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <random>
#include <thread> // Added for sleep in blocking move

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

// Headers for image viewing
#include "sensor_msgs/msg/image.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using std::placeholders::_1;
using namespace std::chrono_literals;

struct EEPos {
	double x;
	double y;
	double z;
};

class MyRobotIK : public rclcpp::Node{
public:
	MyRobotIK() : Node("leg_inverse_kinematics_example"){
        joint_offset_ = {0, 0, 0, 0, 0};
        
		ZERO_X_ = 0.100;
		ZERO_Y_ = 0.000;
		ZERO_Z_ = 0.130;

		curr_x_ = ZERO_X_;
		curr_y_ = ZERO_Y_;
		curr_z_ = ZERO_Z_;

        step_size_ = 0.02; 
		step_size_diag_ = 0.0141421356;
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

		camera_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
			"/image_raw", 
			1, 
			std::bind(&MyRobotIK::imageCallback, this, _1)
		);
		RCLCPP_INFO(this->get_logger(), "Subscribing to camera topic: /image_raw");
		
		dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
		parameters_ = cv::aruco::DetectorParameters::create();
		
		// TODO: REPLACE LATER
		camera_matrix_ = (cv::Mat_<double>(3, 3) << 
			449.55619677,   0.0,         347.40316357, 
			0.0,         452.21172792, 223.13192478, 
			0.0, 0.0, 1.0);
			
		dist_coeffs_ = (cv::Mat_<double>(5, 1) << 
			0.67764151, -2.68262838,  0.01394802, -0.00579161,  3.23228471); 
        RCLCPP_INFO(this->get_logger(), "MyRobotIK Node Initialized with Joint Command Publisher.");
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

		// TODO: THIS IS MAIN RUNNING SEQUENCE
		timer_ = this->create_wall_timer(
            1s, // long enough timeout to block
            std::bind(&MyRobotIK::runSequence, this)
        );
        
        subscription_.reset();
	}

	// is blocking
	void runSequence() {
		if (!solver_ || !rclcpp::ok()) {
			return;
		}

		while (rclcpp::ok()) {
            RCLCPP_INFO(this->get_logger(), "--- STARTING NEW SEQUENCE ITERATION ---");

            // 1. Choose one of the 8 different positions from current position 
            chooseNextTarget();

            // 2. IK calculate, move smooth to the chosen target position(pushing block move)
            EEPos currEEPos = {curr_x_, curr_y_, curr_z_};
            EEPos targetEEPos = {traj_target_x_, traj_target_y_, traj_target_z_};
            EEPos zeroEEPos = {ZERO_X_, ZERO_Y_, ZERO_Z_};
            RCLCPP_INFO(this->get_logger(), "2) Moving to target...");
            moveEESmooth(currEEPos, targetEEPos, 2.0);

            // 2.a Update current position after reaching target
            curr_x_ = traj_target_x_;
            curr_y_ = traj_target_y_;
            curr_z_ = traj_target_z_;

            // 3. Go back to start position to get better camera top down view 
            RCLCPP_INFO(this->get_logger(), "3) Moving to zero/reset position...");
            moveEESmooth(targetEEPos, zeroEEPos, 1.5);

            // 4. Camera has clear view of scene, now take picture and state est 
            RCLCPP_INFO(this->get_logger(), "4) *** PSEUDO CODE: Taking Picture/Reading State at Zero Position ***");
            std::this_thread::sleep_for(500ms);

            // 5. Restore current position(the one before going back to start position) 
            RCLCPP_INFO(this->get_logger(), "5) Restoring to target position...");
            moveEESmooth(zeroEEPos, targetEEPos, 1.0); // 1.0 seconds duration

            // 6. Update random state index for next iteration. In MPC, this will be chosen more "wisely"
            std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> distrib(0, 7);
			step_index_ = distrib(gen);
            RCLCPP_INFO(this->get_logger(), "--- SEQUENCE COMPLETE. Next direction: %d ---", step_index_);
            
            // 6. Just wait a bit
            std::this_thread::sleep_for(500ms);
		}
	}
	
    void chooseNextTarget() {
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
        
        RCLCPP_INFO(this->get_logger(), "New Target: (%.4f, %.4f, %.4f)", traj_target_x_, traj_target_y_, traj_target_z_);
    }

	void moveEESmooth(const struct EEPos& currEEPos, const struct EEPos& targetEEPos, const double total_duration_s) {
        if (total_duration_s <= 0.0) return;

        // Calculate time per step
        auto start_time = this->now();
        double time_per_step_s = total_duration_s / (double)total_sub_steps_;
        auto time_per_step_chrono = std::chrono::duration<double>(time_per_step_s);

        for (int i = 0; i < total_sub_steps_; ++i) {
            auto loop_start_time = std::chrono::steady_clock::now();
            
            double t_normalized = (double)(i + 1) / (double)total_sub_steps_; 

            double interp_x = currEEPos.x + t_normalized * (targetEEPos.x - currEEPos.x);
            double interp_y = currEEPos.y + t_normalized * (targetEEPos.y - currEEPos.y);
            double interp_z = currEEPos.z + t_normalized * (targetEEPos.z - currEEPos.z);

            KDL::JntArray q_current(chain_.getNrOfJoints());
            int ret = getJointAngles(interp_x, interp_y, interp_z, q_current);
            
            if (ret >= 0) {
                publishJointCommand(q_current);
            } else {
                RCLCPP_WARN(this->get_logger(), "IK failed for smooth move step %d!", i);
                return; 
            }

            auto loop_end_time = std::chrono::steady_clock::now();
            auto elapsed_loop_time = loop_end_time - loop_start_time;
            
            if (elapsed_loop_time < time_per_step_chrono) {
                std::this_thread::sleep_for(time_per_step_chrono - elapsed_loop_time);
            }
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
		
		const double upper_safety = 1.57; // 90 degrees
		const double lower_safety = -M_PI; // -180 degrees
		bool limit_exceeded = false;
		
		for (unsigned int i = 0; i < joint_angles.data.size(); ++i) {
			double corrected_angle = joint_angles(i) - joint_offset_[i];
			if (i == 3) {
				corrected_angle -= M_PI; 
			}
			if (i == 3 && (corrected_angle > upper_safety || corrected_angle < lower_safety)) {
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
	
	void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
		RCLCPP_DEBUG(this->get_logger(), "Calling imageCallback...");
        try {
			cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
			cv::Mat frame = cv_ptr->image.clone();
			
			// --- Display and Save Logic ---
			std::lock_guard<std::mutex> lock(image_mutex_);
			current_frame_ = frame.clone();
			
			cv::Mat frame_to_display = current_frame_.clone();
			std::string step_info = "Step: " + std::to_string(step_index_);
			cv::putText(frame_to_display, step_info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
			
			cv::imshow("Camera View (IK Step)", frame_to_display);
			// cv::imwrite("/home/steve/ros2_ws/src/kdl-tutorials/kdl_tutorial_01/test_camdata/img_s"+std::to_string(step_index_)+".jpg", frame_to_display);
			cv::waitKey(1); 
	
		} catch (cv_bridge::Exception& e) {
			RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
		}
    }
	
	double ZERO_X_;
	double ZERO_Y_;
	double ZERO_Z_;

	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_; // Timer for the sequence
	KDL::Tree tree_;
	KDL::Chain chain_;
	std::unique_ptr<KDL::ChainIkSolverPos_LMA> solver_;
    std::vector<double> joint_offset_; 
    
    double step_size_; 
    double step_size_diag_; 
    int step_index_; 
	
	double curr_x_, curr_y_, curr_z_;
	double traj_target_x_, traj_target_y_, traj_target_z_;
	
	const int total_sub_steps_ = 100; // for traj-interpolation
	
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_subscription_;
    cv::Mat current_frame_;
    std::mutex image_mutex_;
	
	// Aruco Detection Members
	cv::Ptr<cv::aruco::Dictionary> dictionary_;
	cv::Ptr<cv::aruco::DetectorParameters> parameters_;
	float marker_length_ = 0.015; 
	
	// Camera Calibration Data
	cv::Mat camera_matrix_;
	cv::Mat dist_coeffs_;
};

int main(int argc, char* argv[]){
	rclcpp::init(argc, argv);
    
	auto ik_node = std::make_shared<MyRobotIK>();
	// Start the blocking robot sequence in a separate thread
    std::thread sequence_thread([&]() {
        // You may need to call a function on the node object to start the sequence
        // This assumes MyRobotIK::runSequence is modified slightly to run without a timer
        ik_node->runSequenceInThread(); 
    });

    // The main thread runs the single-threaded executor, which handles:
    // - The camera image callback (imageCallback)
    // - The joint command publishing (publishJointCommand)
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(ik_node);
    executor.spin();
    
	// clean remaining up
    if (sequence_thread.joinable()) {
        sequence_thread.join();
    }
	cv::destroyAllWindows();
	rclcpp::shutdown();
	return 0;
}