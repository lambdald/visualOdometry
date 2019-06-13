#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

namespace MVSO {


class Frame
{
	friend class MultiViewStereoOdometry;

  public:

    static int FRAME_COUNT;
    const int bucketSize = 15;

    Frame() = default;
    Frame(cv::Mat imgLeft, cv::Mat imgRight);
    void setFeature(const std::vector<cv::Point2f> &keypoints);
    cv::Mat getLeftImg();
    cv::Mat getRightImg();
    void prepareFeature();
    void featureDetection(std::vector<cv::Point2f> &points);
	std::vector<cv::Point2f> getKeypoints();
	std::vector<cv::Point3f> getKeypoints3D();
    void bucketingFeature(int bucket_size);
	void removeInvalidNewFeature(std::vector<bool>& status);
	void addStereoMatch(std::vector<cv::Point2f>& keypoints, std::vector<cv::Point3f>& keypoints3D);
	void addStereoMatch(std::vector<cv::Point2f>& keypoints, cv::Mat& keypoints3D);
	void setInterframeMatching(std::vector<int>& matchId);


	void updateFeatures();

private:
    const int frameId_;
    cv::Mat grayImgLeft_;
    cv::Mat grayImgRight_;
    cv::Mat imgLeft_;
    std::vector<cv::Point2f> keyPoints_;
    std::vector<int> pointAges_;
    std::vector<int> baseKeyPointIndex_;
	std::vector<cv::Point3f> keypoints3D_;
};

}




#endif
