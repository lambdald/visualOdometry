#include "cameramodel.h"
namespace MVSO {

CameraModel::CameraModel(float fx, float fy, float cx, float cy, float bf):
    fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(bf)
{
    projMatLeft_ = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    projMatRight_ = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);
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
