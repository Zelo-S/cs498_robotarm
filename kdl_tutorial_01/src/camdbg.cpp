#include <opencv2/opencv.hpp>
#include <iostream>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

struct ObjectPose {
	double x;
	double y;
	double z;
	double r;
	double p;
	double yw;	
};
	
void updateObjectPoseArUco(cv::Mat frame) {
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
        449.55619677,   0.,         347.40316357,
        0.,         452.21172792, 223.13192478,
        0.,           0.,           1.       
    );

    // --- Distortion Coefficients (D) - 1x5 Double-Precision ---
    // This is a single row, 5 column matrix.
    cv::Mat dist_coeffs = (cv::Mat_<double>(1, 5) <<
        0.67764151, -2.68262838,  0.01394802, -0.00579161,  3.23228471
    );
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
            cv::aruco::drawAxis(debug_frame, camera_matrix, dist_coeffs, rvec_single, tvec_single, MARKER_LENGTH_M);

            if (current_id == 0) { // OBJECT
                target_object_pose.r = rvec_single.at<double>(0, 0);
                target_object_pose.p = rvec_single.at<double>(0, 1);
                target_object_pose.yw = rvec_single.at<double>(0, 2);
                target_object_pose.x = tvec_single.at<double>(0, 0);
                target_object_pose.y = tvec_single.at<double>(0, 1);
                target_object_pose.z = tvec_single.at<double>(0, 2);
                cv::Mat R; // R will be a 3x3 rotation matrix
                cv::Rodrigues(rvec_single, R);
                printf("ID %d has tvec: (%.4f, %.4f, %.4f) and rvec: (%.4f, %.4f, %.4f)\n", i, target_object_pose.x, target_object_pose.y, target_object_pose.z, R.at<double>());
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
}

int main(int argc, char** argv) {
    // 1. Open the default camera (index 0)
    // Use the camera index that corresponds to your device, typically 0.
    cv::VideoCapture cap(0); 

    // Check if the camera opened successfully
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera." << std::endl;
        return -1;
    }

    // 2. Set the desired resolution (Width and Height)
    // Note: OpenCV uses (Width, Height). We set 640x480.
    // The camera may not support these exact values and will choose the nearest supported resolution.
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    // Log the actual resolution the camera is using
    std::cout << "Attempted resolution: 640x480" << std::endl;
    std::cout << "Actual resolution: " 
              << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" 
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    // 3. Create a window to display the video
    cv::namedWindow("Live Camera Feed", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    
    bool CENTER_DEBUG = false;

    // 4. Start the continuous video loop
    while (true) {
        // Capture a new frame from the camera
        cap.read(frame); 

        // Check if the frame was read correctly (e.g., camera unplugged)
        if (frame.empty()) {
            std::cerr << "ERROR! Blank frame grabbed." << std::endl;
            break;
        }
        
        if (CENTER_DEBUG){
            cv::Scalar color(0, 255, 0);

            cv::Point top(320, 0);
            cv::Point bottom(320, 480);

            cv::Point left(0, 240);
            cv::Point right(640, 240);
            
            cv::line(frame, top, bottom, color, 1);
            cv::line(frame, left, right, color, 1);
            
            cv::Point point_to_draw(320, 240);
            cv::circle(frame, point_to_draw, 3, color, 1);
        }

        // Display the frame in the window
        updateObjectPoseArUco(frame);
        // cv::imshow("Live Camera Feed", frame);
        

        // Wait for 1 millisecond. If 'q' is pressed, exit the loop.
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // 5. Release the camera and close the display window
    cap.release();
    cv::destroyAllWindows();

    return 0;
}