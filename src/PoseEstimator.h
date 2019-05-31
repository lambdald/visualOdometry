#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "cameramodel.h"

namespace MVSO
{

	class PoseEstimator
	{
	public:
		PoseEstimator(CameraModel& camera);

		cv::Mat estimatePose(
			std::vector<cv::Point2f>&  pointsLeft_t0,
			std::vector<cv::Point2f>&  pointsLeft_t1,
			std::vector<cv::Point3f>& points3D_t0);

		cv::Mat estimatePose(
			std::vector<cv::Point2f>&  pointsLeft_t0,
			std::vector<cv::Point2f>&  pointsLeft_t1,
			std::vector<cv::Point3f>& points3D_t0,
			std::vector<cv::Point3f>& points3D_t1);


		~PoseEstimator();

		CameraModel camera_;
	};

}

