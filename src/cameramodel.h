#ifndef CAMERAMODEL_H
#define CAMERAMODEL_H

#include <opencv2/core.hpp>

namespace MVSO {

class CameraModel
{
public:
    CameraModel() = default;
    CameraModel(float fx, float fy, float cx, float cy, float bf);
    cv::Mat getLeftProjectionMatrix() const;
    cv::Mat getRightProjectionMatrix() const;
private:
    double fx_, fy_, cx_, cy_, bf_;
    cv::Mat projMatLeft_;
    cv::Mat projMatRight_;
};

}

#endif // CAMERAMODEL_H
