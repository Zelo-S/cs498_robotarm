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
		
		ZERO_X_ = 0.120;
		ZERO_Y_ = 0.000;
		ZERO_Z_ = 0.130;

		GOAL_X_ =   -0.0177;
		GOAL_Y_ =   -0.0345;
		GOAL_YAW_ = -0.4210;
		
		GOAL_DISTANCE = 0.025;

		curr_x_ = ZERO_X_;
		curr_y_ = ZERO_Y_;
		curr_z_ = ZERO_Z_;
		
        step_size_ = 0.005; 
		step_size_diag_ = step_size_ / std::sqrt(2);
        step_index_ = 0;
		
		camera_matrix = (cv::Mat_<double>(3, 3) <<
			449.55619677,   0.,         347.40316357,
			0.,         452.21172792, 223.13192478,
			0.,           0.,           1.       
		);

		dist_coeffs = (cv::Mat_<double>(1, 5) <<
			0.67764151, -2.68262838,  0.01394802, -0.00579161,  3.23228471
		);

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

	/*
	 * Main running loop for the system
	 */
	void runSequence() {
		if (!solver_ || !rclcpp::ok()) {
			return;
		}
		// TODO: definitely cleaner way to do this later
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> distrib(0, 7);
		int iter = 0;
		Point zeroEEPos = {ZERO_X_, ZERO_Y_, ZERO_Z_};
		Point targetEEPos = {ZERO_X_, ZERO_Y_, ZERO_Z_};
		while (rclcpp::ok()) {
            RCLCPP_INFO(this->get_logger(), "--- STARTING NEW SEQUENCE ITERATION ---");
            Point currEEPos = {curr_x_, curr_y_, curr_z_};

            // 3. Go back to start position to get better camera top down view 
            RCLCPP_INFO(this->get_logger(), "3) Moving to zero/reset position...");
            moveEESmooth(currEEPos, zeroEEPos, 1.5);

            // 4. Camera has clear view of scene, now take picture and state est 
            RCLCPP_INFO(this->get_logger(), "4) *** Capturing frame directly from camera ***");
            cv::Mat captured_frame = captureSingleFrame();
			const ObjectPose objectPose = updateObjectPoseArUco(captured_frame); // TODO: updates position of ID0, 1, and 2
			double current_distance = gradeDistanceMetric(objectPose);
			if(current_distance <= GOAL_DISTANCE){
				RCLCPP_INFO(this->get_logger(), "GOAL DISTANCE REACHED, DONE!");
				break;
			}

			RCLCPP_INFO(this->get_logger(), "5) Plannning next %d horizon...", PLAN_HORIZON);
			int next_action = planUsingHorizon(objectPose);
			
            RCLCPP_INFO(this->get_logger(), "Goal position is: (%f %f %f) with rot (%f %f %f)", objectPose.x, objectPose.y, objectPose.z, objectPose.r, objectPose.p, objectPose.yw);

            // 5. Restore current position(the one before going back to start position) 
            RCLCPP_INFO(this->get_logger(), "6) Restoring to target position...");
            moveEESmooth(zeroEEPos, currEEPos, 1.5);

            // 6. Update state index for current iteration to send to IK planner
			step_index_ = next_action;
            RCLCPP_INFO(this->get_logger(), "--- SEQUENCE COMPLETE. Next direction: %d ---", step_index_);

            getEETargetPosition(); // info for robot to decide what traj_target is 

            // 2. IK calculate, move smooth to the chosen target position(pushing block move)
			targetEEPos = {traj_target_x_, traj_target_y_, ZERO_Z_};
            RCLCPP_INFO(this->get_logger(), "2) Moving to target...");
            moveEESmooth(currEEPos, targetEEPos, 2.0);

            // 2.a Update current position after reaching target
            curr_x_ = traj_target_x_;
            curr_y_ = traj_target_y_;
            curr_z_ = traj_target_z_;
			         
            
            // 6. Just wait a bit
            std::this_thread::sleep_for(500ms);
			iter++;
		}
	}
	
	/*
	 * Runs predictNextState for PLAN_HORIZON # of steps, selects next best direction to push in using MPC
	 */
	int planUsingHorizon(const ObjectPose start_object_pose) {
		RCLCPP_INFO(this->get_logger(), "\n\n\t====== Start Planning ======");
		const std::vector<int> ACTIONS = {0, 1, 7};

		ObjectPose curr_object_pose;
		curr_object_pose= {start_object_pose.x, start_object_pose.y, start_object_pose.z, start_object_pose.r, start_object_pose.p, start_object_pose.yw};

		ObjectPose next_step_object_pose;
		int next_step_action = -1;

		ObjectPose planning_object_pose = {start_object_pose.x, start_object_pose.y, start_object_pose.z, start_object_pose.r, start_object_pose.p, start_object_pose.yw};
		for(int i=0; i<PLAN_HORIZON; ++i){

			double curr_distance = gradeDistanceMetric(planning_object_pose);
			int planning_best_action = -1;

			ObjectPose planning_next_best_object_pose;

			for(int action : ACTIONS){
				// Object pose for getting result pose of Action "action"
				ObjectPose candidate_planning_next_best_object_pose;
				candidate_planning_next_best_object_pose = {planning_object_pose.x, planning_object_pose.y, planning_object_pose.z, planning_object_pose.r, planning_object_pose.p, planning_object_pose.yw};
				candidate_planning_next_best_object_pose = predictNextState(candidate_planning_next_best_object_pose, action); // TAKE SINGLE STEP in current horizon step
				double candidate_next_distance = gradeDistanceMetric(candidate_planning_next_best_object_pose);

				if(candidate_next_distance < curr_distance){
					curr_distance = candidate_next_distance;
					planning_next_best_object_pose = candidate_planning_next_best_object_pose;
					planning_best_action = action;
				}
			}
			
			planning_object_pose = {planning_next_best_object_pose.x, planning_next_best_object_pose.y, planning_next_best_object_pose.z, planning_next_best_object_pose.r, planning_next_best_object_pose.p, planning_next_best_object_pose.yw};;	
			RCLCPP_INFO(this->get_logger(), "\tOn horizon %d: chose best action %d = min distance of %f", i, planning_best_action, planning_object_pose);
			
			if (i == 0){
				next_step_object_pose = {planning_object_pose.x, planning_object_pose.y, planning_object_pose.z, planning_object_pose.r, planning_object_pose.p, planning_object_pose.yw};;	
				next_step_action = planning_best_action;
			}
		}
		
		double final_min_distance = gradeDistanceMetric(next_step_object_pose);
		RCLCPP_INFO(this->get_logger(), "\tAfter HORIZON steps, next best action is %d = distance of %f", next_step_action, final_min_distance);
		RCLCPP_INFO(this->get_logger(), "\n\t====== End Planning ======\n");
		return next_step_action;
	}
	
	/*
	 * Gets s_k+1 from (s_k, a_k)
	 */
	const ObjectPose predictNextState(const ObjectPose target_object_pose, int direction){
		ObjectPose next_object_pose;
		next_object_pose= {target_object_pose.x, target_object_pose.y, target_object_pose.z, target_object_pose.r, target_object_pose.p, target_object_pose.yw};
		if (direction == 0){ // Straight +x push
			next_object_pose.x += std::cos(-135 * M_PI / 180.0) * step_size_ / std::sqrt(2);
			next_object_pose.y += std::sin(-135 * M_PI / 180.0) * step_size_ / std::sqrt(2);
		}else if (direction == 1){ // Diagonal left-hand push
			next_object_pose.x += std::cos(-180 * M_PI / 180.0) * step_size_ / std::sqrt(2);
		}else if (direction == 7){ // Diagonal right-hand push
			next_object_pose.y += std::sin(-90 * M_PI / 180.0) * step_size_ / std::sqrt(2);
		}
		return next_object_pose;
	}	
	
	/*
	 * Grades distance and (rotation)(soon) between current object pose and goal object pose
	 */
	double gradeDistanceMetric(const ObjectPose target_object_pose){
		double distance = std::sqrt((target_object_pose.x - GOAL_X_) * (target_object_pose.x - GOAL_X_) + (target_object_pose.y - GOAL_Y_) * (target_object_pose.y - GOAL_Y_));
		return distance;
	}
	
	/*
	 * Handles getting aruco poses of goal + object markers, gets distance and shows frame
	 */
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
					target_object_pose.x = tvec_single.at<double>(0, 0);
					target_object_pose.y = tvec_single.at<double>(0, 1);
					target_object_pose.z = tvec_single.at<double>(0, 2);
					cv::Mat R;
					cv::Rodrigues(rvec_single, R);
					double euler_R;
					double euler_P;
					double euler_Y;
					rotMatrixToEulerAngles(R, euler_R, euler_P, euler_Y);
					target_object_pose.r = euler_R;
					target_object_pose.p = euler_P;
					target_object_pose.yw = euler_Y;
				} else if (current_id == 1) { // TOP LEFT CORNER
					top_left_m_pose.x = tvec_single.at<double>(0, 0);
					top_left_m_pose.y = tvec_single.at<double>(0, 1);
					top_left_m_pose.z = tvec_single.at<double>(0, 2);
					cv::Mat R;
					cv::Rodrigues(rvec_single, R);
					double euler_R;
					double euler_P;
					double euler_Y;
					rotMatrixToEulerAngles(R, euler_R, euler_P, euler_Y);
					top_left_m_pose.r = euler_R;
					top_left_m_pose.p = euler_P;
					top_left_m_pose.yw = euler_Y;
				}
			}
				
			double distance = gradeDistanceMetric(target_object_pose);
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
		POSE OF ID0(object) at END_EFFECTOR BEGIN: 
		[camdbg-1] ID 1 has tvec: [0.07730741100712113, 0.06207665734340477, 0.2991301244755749] and rvec: [-2.149722751548426, 2.16264255794435, 0.09551008417326658]

		POSE OF ID0(object) at :
		Goal position is: (0.057321 0.040886 0.301466)

		POSE OF TOP-LEFT marker(goal position anchor):
		[main-1] ID 1 has tvec: [-0.04657767203672131, -0.07397812788536889, 0.3073267657800779] and rvec: [-2.304558565816952, 2.223394781272074, 0.1611281108275131]

		*/
		return target_object_pose;
	}

	/*
	 * Opens camera for exactly one frame for capture
	 */	
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
	
	void rotMatrixToEulerAngles(const cv::Mat& R, double& roll, double& pitch, double& yaw) {
		CV_Assert(R.rows == 3 && R.cols == 3 && R.type() == CV_64F);
		double R31 = R.at<double>(2, 0);
		pitch = std::asin(-R31); 
		roll = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		yaw = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
		roll = roll * 180.0 / M_PI;
		pitch = pitch * 180.0 / M_PI;
		yaw = yaw * 180.0 / M_PI;
	}

	// TODO: DEPRECATE OR CHANGE LOGIC AFTER MPC OPTIM IS DONE	
    void getEETargetPosition() {
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

	/*
	 * Creates smooth trajectory between two Points for end-effector
	 */
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
	
	const int PLAN_HORIZON = 4;

	double ZERO_X_;
	double ZERO_Y_;
	double ZERO_Z_;

	double GOAL_X_;
	double GOAL_Y_;
	double GOAL_YAW_;
	
	double GOAL_DISTANCE;

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