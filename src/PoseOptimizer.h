#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "cameramodel.h"


namespace MVSO {

class PoseOptimizer
{
public:
	PoseOptimizer(CameraModel& camera);

	void optimizePose(
		const std::vector< cv::Point3f > points_3d,
		const std::vector< cv::Point2f > points_2d,
		cv::Mat& R, cv::Mat& t);

	void optimizePose(
		const std::vector< cv::Point3f > points_3d,
		const std::vector< cv::Point2f > points_2d,
		const std::vector< double >& weights,
		cv::Mat& R, cv::Mat& t);

	void optimizePose(
		const std::vector< cv::Point3f > points3d_t0,
		const std::vector< cv::Point3f > points3d_t1,
		cv::Mat& R, cv::Mat& t);


	CameraModel camera_;

	~PoseOptimizer();
};

}


