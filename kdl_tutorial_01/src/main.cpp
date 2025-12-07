#include <stdio.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <random>
#include <thread>

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

struct Point {
	double x;
	double y;
	double z;
};

struct ObjectPose {
	double x;
	double y;
	double z;
	double r;
	double p;
	double yw;	
};

class MyRobotIK : public rclcpp::Node{
public:
	MyRobotIK() : Node("leg_inverse_kinematics_example"){
        joint_offset_ = {0, 0, 0, 0, 0};
		
		GOAL_X_ = 0.2335;
		GOAL_Y_ = 0.0775;
		GOAL_Z_ = 0.1700;
        
		ZERO_X_ = 0.120;
		ZERO_Y_ = 0.000;
		ZERO_Z_ = 0.130;

		curr_x_ = ZERO_X_;
		curr_y_ = ZERO_Y_;
		curr_z_ = ZERO_Z_;

        step_size_ = 0.02; 
		step_size_diag_ = 0.0141421356;
        step_index_ = 0;
		
		/*camera_matrix = (cv::Mat_<double>(3, 3) <<
			3027.295717240753, 0, 502.5763413358262,
			0, 2940.135565245645, 221.9959609898277,
			0, 0, 1
		);

		// --- Distortion Coefficients (D) - 1x5 Double-Precision ---
		// This is a single row, 5 column matrix.
		dist_coeffs = (cv::Mat_<double>(1, 5) <<
			11.39361170222715, -168.4910558986219, -0.07359531333450298, 1.435873612908785, 1564.41683752833
		);*/
		
		// Currently the best intrinsics
		camera_matrix = (cv::Mat_<double>(3, 3) <<
			449.55619677,   0.,         347.40316357,
			0.,         452.21172792, 223.13192478,
			0.,           0.,           1.       
		);

		// --- Distortion Coefficients (D) - 1x5 Double-Precision ---
		// This is a single row, 5 column matrix.
		dist_coeffs = (cv::Mat_<double>(1, 5) <<
			0.67764151, -2.68262838,  0.01394802, -0.00579161,  3.23228471
		);

		// Optional: Verification output
		std::cout << "Camera Matrix (K):\n" << camera_matrix << std::endl;
		std::cout << "\nDistortion Coefficients (D):\n" << dist_coeffs << std::endl;
		
		// Camera device index - adjust if needed (usually 0 or /dev/video0)
		camera_device_index_ = 0;
		
		subscription_ = this->create_subscription<std_msgs::msg::String>(
		"robot_description",
		rclcpp::QoS(rclcpp::KeepLast(1))
			.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL),
		std::bind(&MyRobotIK::robotDescriptionCallback, this, _1));

        publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>( 
            "/forward_position_controller/commands", 
            10
        );
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

		// THIS IS MAIN RUNNING SEQUENCE
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
		// TODO: definitely cleaner way to do this later
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> distrib(0, 7);
		int iter = 0;
		while (rclcpp::ok()) {
            RCLCPP_INFO(this->get_logger(), "--- STARTING NEW SEQUENCE ITERATION ---");

            // 1. Choose one of the 8 different positions from current position 
            chooseNextTarget();
			int retry_choose_attempt = 0;
			while(traj_target_x_ < ZERO_X_-0.001){
				if(retry_choose_attempt > 100){
					break;
				}
				RCLCPP_INFO(this->get_logger(), "End-effector to collide base to go from %f to %f  -- retrying chooseNextTarget #%d", curr_x_, traj_target_x_, retry_choose_attempt);
				step_index_ = distrib(gen);
				chooseNextTarget();
				retry_choose_attempt++;
			}

            // 2. IK calculate, move smooth to the chosen target position(pushing block move)
            Point currEEPos = {curr_x_, curr_y_, curr_z_};
            Point targetEEPos = {traj_target_x_, traj_target_y_, traj_target_z_};
            Point zeroEEPos = {ZERO_X_, ZERO_Y_, ZERO_Z_};
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
            RCLCPP_INFO(this->get_logger(), "4) *** Capturing frame directly from camera ***");
            cv::Mat captured_frame = captureSingleFrame();
			const ObjectPose objectPose = updateObjectPoseArUco(captured_frame); // TODO: updates position of ID0, 1, and 2
			RCLCPP_INFO(this->get_logger(), "AFTER Getting you the target object pose...");
			
            RCLCPP_INFO(this->get_logger(), "Goal position is: (%f %f %f) with rot (%f %f %f)", objectPose.x, objectPose.y, objectPose.z, objectPose.r, objectPose.p, objectPose.yw);

            // 5. Restore current position(the one before going back to start position) 
            RCLCPP_INFO(this->get_logger(), "5) Restoring to target position...");
            moveEESmooth(zeroEEPos, targetEEPos, 1.5);

            // 6. Update random state index for next iteration, in MPC, this will be chosen more "wisely"
			step_index_ = distrib(gen);
            RCLCPP_INFO(this->get_logger(), "--- SEQUENCE COMPLETE. Next direction: %d ---", step_index_);
            
            // 6. Just wait a bit
            std::this_thread::sleep_for(500ms);
			iter++;
		}
	}
	
	const ObjectPose updateObjectPoseArUco(cv::Mat frame) {
		const float MARKER_LENGTH_M = 0.025f;

		ObjectPose target_object_pose;
		target_object_pose = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		ObjectPose top_left_m_pose;
		top_left_m_pose = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

		cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
		cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

		std::vector<std::vector<cv::Point2f>> marker_corners;
		std::vector<int> marker_ids;
		
		cv::Mat rvecs, tvecs;

		cv::aruco::detectMarkers(frame, dictionary, marker_corners, marker_ids, params);
		
		cv::Mat debug_frame = frame.clone();

		if (!marker_ids.empty()) {
			cv::aruco::estimatePoseSingleMarkers(marker_corners, MARKER_LENGTH_M, camera_matrix, dist_coeffs, rvecs, tvecs);
			cv::aruco::drawDetectedMarkers(debug_frame, marker_corners, marker_ids);

			for (size_t i = 0; i < marker_ids.size(); ++i) {
				int current_id = marker_ids[i];
				cv::Mat rvec_single = rvecs.row(i);
				cv::Mat tvec_single = tvecs.row(i);
				std::cout << "ID " << i << " has tvec: " << tvec_single << " and rvec: " << rvec_single << "\n";
				cv::aruco::drawAxis(debug_frame, camera_matrix, dist_coeffs, rvec_single, tvec_single, MARKER_LENGTH_M);

				if (current_id == 0) { // OBJECT
					target_object_pose.r = rvec_single.at<double>(0, 0);
					target_object_pose.p = rvec_single.at<double>(0, 1);
					target_object_pose.yw = rvec_single.at<double>(0, 2);
					target_object_pose.x = tvec_single.at<double>(0, 0);
					target_object_pose.y = tvec_single.at<double>(0, 1);
					target_object_pose.z = tvec_single.at<double>(0, 2);
				} else if (current_id == 1) { // TOP LEFT CORNER
					top_left_m_pose.r = rvec_single.at<double>(0, 0);
					top_left_m_pose.p = rvec_single.at<double>(0, 1);
					top_left_m_pose.yw = rvec_single.at<double>(0, 2);
					top_left_m_pose.x = tvec_single.at<double>(0, 0);
					top_left_m_pose.y = tvec_single.at<double>(0, 1);
					top_left_m_pose.z = tvec_single.at<double>(0, 2);
				}
			}
			double curr_x = target_object_pose.x;
			double curr_y = target_object_pose.y;
			
			double top_left_corner_x = top_left_m_pose.x;
			double top_left_corner_y = top_left_m_pose.y; 
				
			double distance = std::sqrt((curr_x - top_left_corner_x) * (curr_x - top_left_corner_x) + (curr_y - top_left_corner_y) * (curr_y - top_left_corner_y));
			std::stringstream ss;
			ss << std::fixed << std::setprecision(4) << "Dist: " << distance;
			std::string dist_text = ss.str();
			cv::putText(debug_frame, // target image
                dist_text, // text
                cv::Point(100, 100), // top-left position
                cv::FONT_HERSHEY_DUPLEX,
                1.0, // font scale
                cv::Scalar(118, 185, 0), // font color (BGR)
                2);
			cv::imshow("Detected ArUco Markers with Axes", debug_frame);
			cv::waitKey(500);
		} else {
			std::cerr << "No ArUco marker found." << std::endl;
		}
		
		// TODO: Ideally, the following poses should be true:
		/*
		[main-1] ID 1 has tvec: [-0.02261585825783664, -0.04563866880499076, 0.2964818466842044] and rvec: [0.5512701218469557, -2.997373104571034, -0.08813415073767798]
		[main-1] ID 2 has tvec: [0.07742773227251662, -0.06253287122362568, 0.304633047630919] and rvec: [2.09123639890954, -2.056450963305633, -0.9875874028890603]
		*/
		
		return target_object_pose;
	}
	
	cv::Mat captureSingleFrame() {
		RCLCPP_INFO(this->get_logger(), "Opening camera device %d...", camera_device_index_);
		cv::VideoCapture cap(camera_device_index_);
		
		if (!cap.isOpened()) {
			RCLCPP_ERROR(this->get_logger(), "Failed to open camera device %d", camera_device_index_);
			return cv::Mat();
		}
		
		cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
		
		RCLCPP_INFO(this->get_logger(), "Warming up camera...");
		for (int i = 0; i < 5; i++) {
			cv::Mat temp;
			cap >> temp;
			std::this_thread::sleep_for(50ms);
		}
		
		cv::Mat frame;
		cap >> frame;
		
		cap.release();
		RCLCPP_INFO(this->get_logger(), "Camera closed.");
		
		if (frame.empty()) {
			RCLCPP_ERROR(this->get_logger(), "Captured frame is empty!");
		}
		
		return frame;
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

	void moveEESmooth(const struct Point& currEEPos, const struct Point& targetEEPos, const double total_duration_s) {
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

	double GOAL_X_;
	double GOAL_Y_;
	double GOAL_Z_;
	
	double ZERO_X_;
	double ZERO_Y_;
	double ZERO_Z_;

	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
	KDL::Tree tree_;
	KDL::Chain chain_;
	std::unique_ptr<KDL::ChainIkSolverPos_LMA> solver_;
    std::vector<double> joint_offset_; 
    
    double step_size_; 
    double step_size_diag_; 
    int step_index_; 
	
	double curr_x_, curr_y_, curr_z_;
	double traj_target_x_, traj_target_y_, traj_target_z_;
	
	const int total_sub_steps_ = 400; // for traj-interpolation
	
	int camera_device_index_; // Camera device (e.g., 0 for /dev/video0)

	// These members must be defined in your class/context
	cv::Mat camera_matrix; // 3x3 Intrinsic matrix (K)
	cv::Mat dist_coeffs;   // 1x5 or 1x8 Distortion coefficients (D)
	const float MARKER_LENGTH_M = 0.025f; // 25mm in meters
};

int main(int argc, char* argv[]){
	rclcpp::init(argc, argv);
    
	auto ik_node = std::make_shared<MyRobotIK>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(ik_node);
	executor.spin();
	
	cv::destroyAllWindows();

	rclcpp::shutdown();
	return 0;
}