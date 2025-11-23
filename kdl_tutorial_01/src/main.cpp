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

// Headers for image viewing
#include "sensor_msgs/msg/image.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// Headers for aruco detection
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
        
		ZERO_X_ = 0.160;
		ZERO_Y_ = 0.000;
		ZERO_Z_ = 0.130;

		curr_x_ = ZERO_X_;
		curr_y_ = ZERO_Y_;
		curr_z_ = ZERO_Z_;
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

		camera_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
			"/image_raw", // The topic you listed
			1, // QoS history depth
			std::bind(&MyRobotIK::imageCallback, this, _1)
		);
		RCLCPP_INFO(this->get_logger(), "Subscribing to camera topic: /image_raw");
		
		// --- Aruco Initialization ---
		dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
		parameters_ = cv::aruco::DetectorParameters::create();
		
		// !!! REPLACE WITH YOUR ACTUAL CAMERA CALIBRATION DATA !!!
		camera_matrix_ = (cv::Mat_<double>(3, 3) << 
			449.55619677,   0.0,         347.40316357, // fx, 0, cx
			0.0,         452.21172792, 223.13192478, // 0, fy, cy
			0.0, 0.0, 1.0);
			
		dist_coeffs_ = (cv::Mat_<double>(5, 1) << 
			0.67764151, -2.68262838,  0.01394802, -0.00579161,  3.23228471); // k1, k2, p1, p2, k3 (assuming no distortion)
		// -------------------------------------------------------------

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
		
		EEPos currEEPos = {traj_start_x_, traj_start_y_, traj_start_z_};
		EEPos targetEEPos = {traj_target_x_, traj_target_y_, traj_target_z_};
		moveEESmooth(currEEPos, targetEEPos);
				
		resetPosition();
		// TODO: take picture and get state
		restorePosition();

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
	
	void resetPosition(){
		struct EEPos curr_pos = {curr_x_, curr_y_, curr_z_};
		struct EEPos zero_pos = {ZERO_X_, ZERO_Y_, ZERO_Z_};
		moveEESmooth(curr_pos, zero_pos);
	}

	void restorePosition(){
		struct EEPos zero_pos = {ZERO_X_, ZERO_Y_, ZERO_Z_};
		struct EEPos curr_pos = {curr_x_, curr_y_, curr_z_};
		moveEESmooth(zero_pos, curr_pos);
	}
	
	void moveEESmooth(const struct EEPos& currEEPos, const struct EEPos& targetEEPos) {
		double t_normalized = (double)sub_step_index_ / (double)total_sub_steps_;

		double interp_x = currEEPos.x + t_normalized * (targetEEPos.x - currEEPos.x);
		double interp_y = currEEPos.y + t_normalized * (targetEEPos.y - currEEPos.y);
		double interp_z = currEEPos.z + t_normalized * (targetEEPos.z - currEEPos.z);
		
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
		return;
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
        try {
			cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
			cv::Mat frame = cv_ptr->image.clone();
			
			// --- Aruco Detection and Pose Estimation ---
			std::vector<int> marker_ids;
			std::vector<std::vector<cv::Point2f>> marker_corners, rejected_candidates;
			cv::Mat rvecs, tvecs;
	
			// 1. Detect Markers
			cv::aruco::detectMarkers(frame, dictionary_, marker_corners, marker_ids, parameters_, rejected_candidates);
	
			if (!marker_ids.empty()) {
				// 2. Estimate Pose (Rvec and Tvec)
				cv::aruco::estimatePoseSingleMarkers(
					marker_corners, 
					marker_length_, 
					camera_matrix_, 
					dist_coeffs_, 
					rvecs, 
					tvecs
				);
	
				// 3. Draw and Process Results (only for the first detected marker)
				for (size_t i = 0; i < marker_ids.size(); ++i) {
					// Draw detected markers and axes
					cv::aruco::drawDetectedMarkers(frame, marker_corners, marker_ids);
					cv::aruco::drawAxis(frame, camera_matrix_, dist_coeffs_, rvecs.row(i), tvecs.row(i), marker_length_ * 0.5f);
	
					// Log the pose of the first marker (optional, but useful)
					if (i == 0) {
						double tx = tvecs.at<double>(i, 0);
						double ty = tvecs.at<double>(i, 1);
						double tz = tvecs.at<double>(i, 2);
						
						RCLCPP_INFO(this->get_logger(), 
							"Marker %d Pose (X, Y, Z): (%.4f, %.4f, %.4f) m", 
							marker_ids[i], tx, ty, tz);
							
						// !!! HERE you can integrate the Tvec into your IK target !!!
						// For example, update curr_x_, curr_y_, curr_z_ based on Tvec
						// This is how you would close the visual servoing loop.
						// Note: You must transform this Tvec into your 'base_link' frame.
					}
				}
			}
			
			// --- Display and Save Logic ---
			std::lock_guard<std::mutex> lock(image_mutex_);
			current_frame_ = frame.clone();
			
			// The display logic remains the same
			cv::Mat frame_to_display = current_frame_.clone();
			std::string step_info = "Step: " + std::to_string(step_index_) + " Sub: " + std::to_string(sub_step_index_);
			cv::putText(frame_to_display, step_info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
			
			// cv::imshow("Camera View (IK Step)", frame_to_display);
			// cv::imwrite("/home/steve/ros2_ws/src/kdl-tutorials/kdl_tutorial_01/test_camdata/img_s"+std::to_string(step_index_)+"_ss"+std::to_string(sub_step_index_)+".jpg", frame_to_display);
			// cv::waitKey(1); // Required to update the window
	
		} catch (cv_bridge::Exception& e) {
			RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
		}
    }
	
	double ZERO_X_;
	double ZERO_Y_;
	double ZERO_Z_;

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
	
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_subscription_;
    cv::Mat current_frame_;
    std::mutex image_mutex_;
	
	// Aruco Detection Members
	cv::Ptr<cv::aruco::Dictionary> dictionary_;
	cv::Ptr<cv::aruco::DetectorParameters> parameters_;
	float marker_length_ = 0.015; // 50 mm, **Set this to your actual marker size in meters**
	
	// Camera Calibration Data (You MUST replace these with your actual calibrated values)
	cv::Mat camera_matrix_;
	cv::Mat dist_coeffs_;
};

int main(int argc, char* argv[]){
	rclcpp::init(argc, argv);
    
	auto ik_node = std::make_shared<MyRobotIK>();
	// auto fk_node = std::make_shared<FKSolverNode>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(ik_node);
    // executor.add_node(fk_node);
	executor.spin();
	
	cv::destroyAllWindows();

	rclcpp::shutdown();
	return 0;
}