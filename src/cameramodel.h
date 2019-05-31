#ifndef CAMERAMODEL_H
#define CAMERAMODEL_H

#include <opencv2/core.hpp>

namespace MVSO {

class CameraModel
{
public:
    CameraModel() = default;
	CameraModel(const CameraModel& cam);
    CameraModel(float fx, float fy, float cx, float cy, float bf);
    cv::Mat getLeftProjectionMatrix() const;
    cv::Mat getRightProjectionMatrix() const;

    double fx_, fy_, cx_, cy_, bf_;
    cv::Mat projMatLeft_;
    cv::Mat projMatRight_;
	cv::Mat intrinsicMat_;
};

}

#endif // CAMERAMODEL_H
