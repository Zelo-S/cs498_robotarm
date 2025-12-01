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
	Point tlc;
	Point trc;
	Point blc;
	Point brc;
};

class MyRobotIK : public rclcpp::Node{
public:
	MyRobotIK() : Node("leg_inverse_kinematics_example"){
        joint_offset_ = {0, 0, 0, 0, 0};
		
        
		ZERO_X_ = 0.120;
		ZERO_Y_ = 0.000;
		ZERO_Z_ = 0.130;

		curr_x_ = ZERO_X_;
		curr_y_ = ZERO_Y_;
		curr_z_ = ZERO_Z_;

        step_size_ = 0.02; 
		step_size_diag_ = 0.0141421356;
        step_index_ = 0;
		
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
			
			RCLCPP_INFO(this->get_logger(), "Debugging joint angles...");
			KDL::JntArray temp_q_out(chain_.getNrOfJoints());
			Point currEEPos = {0.100, 0, 0.175};             // 2. IK calculate, move smooth to the chosen target position(pushing block move)
			Point targetEEPos = {0.110, 0, 0.175};
			moveEESmooth(currEEPos, targetEEPos, 2.0);
			
			/*// 2. Do a move sequence(in world), ignore chooseNextTarget() for now
            RCLCPP_INFO(this->get_logger(), "--- RESET ---");
			moveToWorldCoord(0, -0.160, ZERO_Z_);
			std::this_thread::sleep_for(500ms);

            RCLCPP_INFO(this->get_logger(), "--- COORD 3 ---");
			moveToWorldCoord(0.01, -0.160, ZERO_Z_);
			std::this_thread::sleep_for(500ms);

            RCLCPP_INFO(this->get_logger(), "--- COORD 3 ---");
			moveToWorldCoord(0.02, -0.160, ZERO_Z_);
			std::this_thread::sleep_for(500ms);


			// 3. Move back to zero pos to capture camera
            RCLCPP_INFO(this->get_logger(), "--- RESET ---");
			moveToWorldCoord(0, -0.160, ZERO_Z_);
			std::this_thread::sleep_for(500ms);*/

            // 4. Camera has clear view of scene, now take picture and state est 
            // RCLCPP_INFO(this->get_logger(), "4) *** Capturing frame directly from camera ***");
            cv::Mat captured_frame = captureSingleFrame();
			const ObjectPose& objectPose = getObjectPose(captured_frame);
			
            RCLCPP_INFO(this->get_logger(), "Goal position is: (%f %f %f)", objectPose.x, objectPose.y, objectPose.z);

            // 6. Just wait a bit
            std::this_thread::sleep_for(500ms);
			iter++;
		}
	}
	
	void moveToWorldCoord(double x, double y, double z){
		z = ZERO_Z_;
		RCLCPP_INFO(this->get_logger(), "Debugging joint angles from the world...");
		KDL::JntArray temp_q_out(chain_.getNrOfJoints());
		KDL::Frame T_robot_world = getRobotPosFromWorld(x, y, ZERO_Z_, temp_q_out);
		std::cout << "T_robot_world for (-0.08, -0.07) is: " << T_robot_world.p.x() << " " << T_robot_world.p.y() << " " << T_robot_world.p.z() << "\n\n";
		Point currEEPos = {curr_x_, curr_y_, curr_z_};             // 2. IK calculate, move smooth to the chosen target position(pushing block move)
		Point targetEEPos = {T_robot_world.p.x(), T_robot_world.p.y(), ZERO_Z_};
		moveEESmooth(currEEPos, targetEEPos, 2.0);
		curr_x_ = targetEEPos.x;
		curr_y_ = targetEEPos.y;
		curr_z_ = targetEEPos.z;
	}
	
	const ObjectPose& getObjectPose(cv::Mat frame){
		static ObjectPose result_pose; 
		result_pose = {-1.0, -1.0, -1.0, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}};
	
		if (frame.empty()) {
			RCLCPP_ERROR(this->get_logger(), "NO FRAME AVAILABLE!!!");
			return result_pose;
		}
	
		cv::Mat hsv_frame, yellow_mask;
		cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);
		cv::Scalar lower_yellow(20, 100, 100);
		cv::Scalar upper_yellow(40, 255, 255);
	
		cv::inRange(hsv_frame, lower_yellow, upper_yellow, yellow_mask);

		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::morphologyEx(yellow_mask, yellow_mask, cv::MORPH_CLOSE, kernel); // Fills gaps
		cv::morphologyEx(yellow_mask, yellow_mask, cv::MORPH_OPEN, kernel);  // Removes noise
	
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(yellow_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	
		if (contours.empty()) {
			RCLCPP_WARN(this->get_logger(), "No yellow object found.");
			return result_pose;
		}
	
		double max_area = 0;
		std::vector<cv::Point> target_contour;
		for (const auto& contour : contours) {
			double area = cv::contourArea(contour);
			if (area > max_area) {
				max_area = area;
				target_contour = contour;
			}
		}
	
		if (max_area < 500) {
			RCLCPP_WARN(this->get_logger(), "Largest yellow contour is too small (Area < 500).");
			return result_pose;
		}
	
		cv::RotatedRect rotated_rect = cv::minAreaRect(target_contour);
		
		cv::Point2f center_f = rotated_rect.center;
		result_pose.x = center_f.x;
		result_pose.y = center_f.y;
		result_pose.z = ZERO_Z_;
	
		cv::Point2f corners_f[4];
		rotated_rect.points(corners_f);
	
		// TODO: corner sorting
		std::vector<cv::Point2f> sorted_corners(corners_f, corners_f + 4);
		std::sort(sorted_corners.begin(), sorted_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
			return a.x < b.x;
		});
		if (sorted_corners[0].y > sorted_corners[1].y) std::swap(sorted_corners[0], sorted_corners[1]); // TL, BL
		if (sorted_corners[2].y > sorted_corners[3].y) std::swap(sorted_corners[2], sorted_corners[3]); // TR, BR
		result_pose.tlc = {sorted_corners[0].x, sorted_corners[0].y, ZERO_Z_}; // TLC (TL in image space)
		result_pose.blc = {sorted_corners[1].x, sorted_corners[1].y, ZERO_Z_}; // BLC (BL in image space)
		result_pose.trc = {sorted_corners[2].x, sorted_corners[2].y, ZERO_Z_}; // TRC (TR in image space)
		result_pose.brc = {sorted_corners[3].x, sorted_corners[3].y, ZERO_Z_}; // BRC (BR in image space)
	
	
		cv::Mat debug_frame = frame.clone();
		cv::Point corners_int[4];
		for(int i = 0; i < 4; ++i) {
			corners_int[i] = cv::Point(corners_f[i].x, corners_f[i].y);
		}
		std::vector<cv::Point> corners_vec(corners_int, corners_int + 4); 
		std::vector<std::vector<cv::Point>> corners_list = {corners_vec};
		cv::polylines(debug_frame, corners_list, true, cv::Scalar(0, 255, 255), 2);
		cv::circle(debug_frame, center_f, 5, cv::Scalar(0, 0, 255), -1);
	
		RCLCPP_INFO(this->get_logger(), "[GET OBJECT POSE] GOT YELLOW OBJECT ");
		cv::imshow("Detected Yellow Object", debug_frame);
		cv::waitKey(500);
	
		return result_pose;
	}
	
	cv::Mat captureSingleFrame() {
		RCLCPP_INFO(this->get_logger(), "Opening camera device %d...", camera_device_index_);
		cv::VideoCapture cap(camera_device_index_);
		
		if (!cap.isOpened()) {
			RCLCPP_ERROR(this->get_logger(), "Failed to open camera device %d", camera_device_index_);
			return cv::Mat();
		}
		
		// Optional: Set camera properties for better quality
		cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
		
		// Allow camera to warm up and adjust exposure
		RCLCPP_INFO(this->get_logger(), "Warming up camera...");
		for (int i = 0; i < 5; i++) {
			cv::Mat temp;
			cap >> temp;
			std::this_thread::sleep_for(50ms);
		}
		
		// Capture the actual frame
		cv::Mat frame;
		cap >> frame;
		
		// Release camera immediately
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
	
	KDL::Frame getRobotPosFromWorld(const double world_x, const double world_y, const double world_z, KDL::JntArray& q_out) {
		double x_w_b = -0.154;
		double y_w_b = -0.160;
		double z_w_b = ZERO_Z_;
		double z_angle_w_b = 0.0 * M_PI / 180.0; // 45 degrees in radians
		KDL::Rotation R_W_B = KDL::Rotation::RPY(0.0, 0.0, z_angle_w_b);
		KDL::Vector P_W_B(x_w_b, y_w_b, z_w_b);
		KDL::Frame T_W_B(R_W_B, P_W_B);
		KDL::Frame T_B_W = T_W_B.Inverse();
		std::cout << std::fixed << std::setprecision(4);

		KDL::Vector P_B_W = T_B_W.p;

		KDL::Rotation R_B_W = T_B_W.M;
		std::cout << "\nR_robotbase_world (Rotation Matrix):\n";
		std::cout << R_B_W(0, 0) << "  " << R_B_W(0, 1) << "  " << R_B_W(0, 2) << "\n";
		std::cout << R_B_W(1, 0) << "  " << R_B_W(1, 1) << "  " << R_B_W(1, 2) << "\n";
		std::cout << R_B_W(2, 0) << "  " << R_B_W(2, 1) << "  " << R_B_W(2, 2) << "\n";

		KDL::Vector P_world_kdl(world_x, world_y, world_z);

		// Transform the world coordinate to the robot base frame coordinate
		KDL::Rotation R_W_EE = KDL::Rotation::Identity(); 
		KDL::Vector P_W_EE(world_x, world_y, world_z);
		KDL::Frame T_W_EE(R_W_EE, P_W_EE);
		KDL::Frame T_B_EE = T_B_W * T_W_EE; // gives coords in robot base frame 
		// Now, call the existing KDL IK solver logic
		KDL::JntArray q_init(chain_.getNrOfJoints());
		for (unsigned int i = 0; i < chain_.getNrOfJoints(); ++i) {
			q_init(i) = 0.0; // Initial guess for IK
		}

		RCLCPP_INFO(this->get_logger(), 
        "Target World: (%.4f, %.4f, %.4f) -> Target Base: (%.4f, %.4f, %.4f)", 
        world_x, world_y, world_z, T_B_EE.p.x(), T_B_EE.p.y(), T_B_EE.p.z());
    
		return T_B_EE;
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