#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

namespace MVSO {


class Frame
{
  public:

    static int FRAME_COUNT;
    const int bucketSize = 5;

    Frame() = default;
    Frame(cv::Mat imgLeft, cv::Mat imgRight);
    void setFeature(const std::vector<cv::Point2f> &keypoints);
    cv::Mat getLeftImg();
    cv::Mat getRightImg();
    void prepareFeature();
    void featureDetection(std::vector<cv::Point2f> &points);
private:

    void bucketingFeature(int bucket_size);


    const int frameId_;
    cv::Mat grayImgLeft_;
    cv::Mat grayImgRight_;
    cv::Mat imgLeft_;
    std::vector<cv::Point2f> keyPoints_;
    std::vector<int> pointAges_;
    std::vector<int> baseKeyPointId_;
};

}




#endif
