#include "cameramodel.h"
namespace MVSO {

	CameraModel::CameraModel(const CameraModel & cam)
	{
		fx_ = cam.fx_;
		fy_ = cam.fy_;
		cx_ = cam.cx_;
		cy_ = cam.cy_;
		bf_ = cam.bf_;

		projMatLeft_ = cam.projMatLeft_.clone();
		projMatRight_ = cam.projMatRight_.clone();
		intrinsicMat_ = cam.intrinsicMat_.clone();
	}

	CameraModel::CameraModel(float fx, float fy, float cx, float cy, float bf) :
		fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(bf)
	{
		projMatLeft_ = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0, 0., 1., 0.);
		projMatRight_ = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0, 0., 1., 0.);
		intrinsicMat_ = (cv::Mat_<float>(3, 3) << fx, 0., cx, 0., fy, cy, 0., 0., 1.);
	}

	cv::Mat CameraModel::getLeftProjectionMatrix() const
	{
		return projMatLeft_;
	}

	cv::Mat CameraModel::getRightProjectionMatrix() const
	{
		return projMatRight_;
	}

}
