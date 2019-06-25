
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <opencv2/core/eigen.hpp>



#include "feature.h"
#include "utils.h"
#include "evaluate_odometry.h"
#include "visualOdometry.h"
#include "Frame.h"

using namespace std;


int main(int argc, char **argv)
{

	// 载入图片和标定数据
    bool display_ground_truth = false;
    std::vector<Matrix> pose_matrix_gt;
    if(argc == 4)
    {   display_ground_truth = true;
        cerr << "Display ground truth trajectory" << endl;
        // load ground truth pose
        string filename_pose = string(argv[3]);
        pose_matrix_gt = loadPoses(filename_pose);
    }
    if(argc < 3)
    {
        cerr << "Usage: ./run path_to_sequence path_to_calibration [optional]path_to_ground_truth_pose" << endl;
        return 1;
    }

    // 数据集路径，目前只测试了kitti00
    string filepath = string(argv[1]);
    cout << "Filepath: " << filepath << endl;

    // 相机参数
    string strSettingPath = string(argv[2]);
    cout << "Calibration Filepath: " << strSettingPath << endl;

	std::vector<cv::Mat> pose_results;


    MVSO::MultiViewStereoOdometry mvso(strSettingPath);
    
	// 相机矩阵
    cv::Mat projMatrl = mvso.camera_.getLeftProjectionMatrix();
    cv::Mat projMatrr = mvso.camera_.getRightProjectionMatrix();

    // -----------------------------------------
    // Initialize variables
    // -----------------------------------------
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation_stereo = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);
    
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat frame_pose32 = cv::Mat::eye(4, 4, CV_32F);

    std::cout << "frame_pose " << frame_pose << std::endl;
    cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);
    FeatureSet currentVOFeatures;
    cv::Mat points4D, points3D;
    int init_frame_id = 0;

    // ------------------------
    // 读入第一帧图像
    // ------------------------
    cv::Mat imageLeft_t0_color,  imageLeft_t0;
    loadImageLeft(imageLeft_t0_color,  imageLeft_t0, init_frame_id, filepath);
    
    cv::Mat imageRight_t0_color, imageRight_t0;  
    loadImageRight(imageRight_t0_color, imageRight_t0, init_frame_id, filepath);


    float fps;

	pose_results.push_back(mvso.grabImage(imageLeft_t0_color, imageRight_t0_color));
	
    // -----------------------------------------
    // 运行视觉里程计
    // -----------------------------------------
    clock_t tic = clock();

    for (int frame_id = init_frame_id+1; frame_id <= 4540; frame_id++)
    {
        std::cout << std::endl << "frame_id " << frame_id << std::endl;
        // ------------
        // 读图
        // ------------
        cv::Mat imageLeft_t1_color,  imageLeft_t1;
        loadImageLeft(imageLeft_t1_color,  imageLeft_t1, frame_id, filepath);        
        cv::Mat imageRight_t1_color, imageRight_t1;  
        loadImageRight(imageRight_t1_color, imageRight_t1, frame_id, filepath);

		cv::Mat pose_mvso;
		pose_mvso = mvso.grabImage(imageLeft_t1, imageRight_t1);
		cv::Mat rotation_mvso, translation_mvso;
		rotation_mvso = pose_mvso.colRange(0, 3);
		translation_mvso = pose_mvso.col(3);
		pose_results.push_back(pose_mvso);

		//cout << pose_mvso << endl;


		rotation = rotation_mvso.clone();
		translation_stereo = translation_mvso.clone();


        cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
        // std::cout << "rotation: " << rotation_euler << std::endl;
        // std::cout << "translation: " << translation_stereo.t() << std::endl;

        cv::Mat rigid_body_transformation;
		//integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation_stereo);
        
		if(abs(rotation_euler[1])<0.2 && abs(rotation_euler[0])<0.2 && abs(rotation_euler[2])<0.2)
        {
			integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation_stereo);
        } else {
            std::cout << "Too large rotation"  << std::endl;
        }

        // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;

        // std::cout << "frame_pose" << frame_pose << std::endl;


        Rpose =  frame_pose(cv::Range(0, 3), cv::Range(0, 3));
        cv::Vec3f Rpose_euler = rotationMatrixToEulerAngles(Rpose);
        // std::cout << "Rpose_euler" << Rpose_euler << std::endl;

        cv::Mat pose = frame_pose.col(3).clone();

        clock_t toc = clock();
        fps = float(frame_id-init_frame_id)/(toc-tic)*CLOCKS_PER_SEC;

        // std::cout << "Pose" << pose.t() << std::endl;
        std::cout << "FPS: " << fps << std::endl;

        display(frame_id, trajectory, pose, pose_matrix_gt, fps, display_ground_truth);

    }
	auto eval_time = chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	string time = std::ctime(&eval_time);
	time.erase(time.length() - 1);
	for (char& c : time)
		if (c == ':')
			c = '~';
	string filename = cv::format("trajectory%s.png", time.c_str());
	cout << "filename: " << filename << endl;
	cv::imwrite(filename, trajectory);
    return 0;
}

