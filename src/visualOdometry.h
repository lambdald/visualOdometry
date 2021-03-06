#ifndef VISUAL_ODOM_H
#define VISUAL_ODOM_H

#include <opencv2/opencv.hpp>

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <opencv2/core/eigen.hpp>

#include "feature.h"
#include "bucket.h"
#include "utils.h"
#include "Frame.h"
#include "cameramodel.h"
#include "Map.h"

void visualOdometry(int current_frame_id, std::string filepath,
                    cv::Mat& projMatrl, cv::Mat& projMatrr,
                    cv::Mat& rotation, cv::Mat& translation_mono, cv::Mat& translation_stereo, 
                    cv::Mat& image_left_t0,
                    cv::Mat& image_right_t0,
                    FeatureSet& current_features,
                    cv::Mat& points4D);

void visualOdometryIMU(int current_frame_id, std::string filepath,
                    cv::Mat& projMatrl, cv::Mat& projMatrr,
                    cv::Mat& rotation, cv::Mat& translation_mono, cv::Mat& translation_stereo, 
                    cv::Mat& image_left_t0,
                    cv::Mat& image_right_t0,
                    FeatureSet& current_features,
                    std::vector<std::vector<double>> time_gyros);


void matchingFeatures(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      FeatureSet& currentVOFeatures,
                      std::vector<cv::Point2f>&  pointsLeft_t0, 
                      std::vector<cv::Point2f>&  pointsRight_t0, 
                      std::vector<cv::Point2f>&  pointsLeft_t1, 
                      std::vector<cv::Point2f>&  pointsRight_t1);


void trackingFrame2Frame(cv::Mat& projMatrl, cv::Mat& projMatrr,
                         std::vector<cv::Point2f>&  pointsLeft_t0,
                         std::vector<cv::Point2f>&  pointsLeft_t1, 
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation);

void displayTracking(cv::Mat& imageLeft_t1, 
                     std::vector<cv::Point2f>&  pointsLeft_t0,
                     std::vector<cv::Point2f>&  pointsLeft_t1);

void displayTracking(cv::Mat& imageLeft_t1,
	std::vector<cv::Point2f>&  pointsLeft_t0,
	std::vector<cv::Point2f>&  pointsLeft_t1,
	cv::Point2f epipoint);

namespace MVSO
{
    class MultiViewStereoOdometry
    {
    public:
        enum class State { INVALID, OK, LOST};
        MultiViewStereoOdometry(const std::string& settingPath);

        cv::Mat grabImage(cv::Mat imgLeft, cv::Mat imgRight);
       

		void matchingFeatures2(Frame* lastFrame, Frame* currentFrame, std::vector<cv::Point2f>& lastFrameKpts);



		void circularMatching(std::vector<cv::Point2f> &pointsLeft_t0,
			std::vector<cv::Point2f> &pointsRight_t0,
			std::vector<cv::Point2f> &pointsLeft_t1,
			std::vector<cv::Point2f> &pointsRight_t1,
			std::vector<bool>& matchStatus);
		

		void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
			std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
			std::vector<cv::Point2f>& points0_return,
			std::vector<uchar>& status0, std::vector<uchar>& status1,
			std::vector<uchar>& status2, std::vector<uchar>& status3);

		void deleteUnmatchFeaturesCircle2(
			std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
			std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
			std::vector<cv::Point2f>& points0_return,
			std::vector<uchar>& status0, std::vector<uchar>& status1,
			std::vector<uchar>& status2, std::vector<uchar>& status3,
			std::vector<bool>& matchResult);


		void trackingFrame2Frame(
			std::vector<cv::Point2f>&  pointsLeft_t0,
			std::vector<cv::Point2f>&  pointsLeft_t1,
			cv::Mat& points3D_t0);

		//void trackingFrame2FrameICP();



        CameraModel camera_;
        std::shared_ptr<Frame> lastFrame_, currentFrame_;
		cv::Mat pose_;

		cv::Mat tracking();
		std::shared_ptr<Map> map_;

		std::queue<std::shared_ptr<Frame>> frames_;
    };
}


#endif
