#include "Frame.h"


namespace MVSO {

int Frame::FRAME_COUNT = 0;

Frame::Frame(cv::Mat imgLeft, cv::Mat imgRight): frameId_(FRAME_COUNT++)
{
    if(imgLeft.channels() == 1 && imgRight.channels() == 1)
    {
        grayImgLeft_ = imgLeft.clone();
        grayImgRight_ = imgRight.clone();
    }
    else
    {
        cv::cvtColor(imgLeft, grayImgLeft_, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgRight, grayImgRight_, cv::COLOR_BGR2GRAY);
    }
}

void Frame::setFeature(const std::vector<cv::Point2f>& keypoints)
{
    keyPoints_ = keypoints;
    pointAges_ = std::vector<int>(keypoints.size(), 0);
}

cv::Mat Frame::getLeftImg()
{
    return grayImgLeft_;
}

cv::Mat Frame::getRightImg()
{
    return grayImgRight_;
}

void Frame::prepareFeature()
{
    if(keyPoints_.size() < 2000)
    {
        std::vector<cv::Point2f>  points_new;
        featureDetection(points_new);
        keyPoints_.insert(keyPoints_.end(), points_new.begin(), points_new.end());
        std::vector<int>  ages_new(points_new.size(), 0);
        pointAges_.insert(pointAges_.end(), ages_new.begin(), ages_new.end());
    }
}

void Frame::featureDetection(std::vector<cv::Point2f>& points)
{
    std::vector<cv::KeyPoint> keypoints;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    cv::FAST(grayImgLeft_, keypoints, fast_threshold, nonmaxSuppression);
    cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}

void Frame::bucketingFeature(int bucket_size)
{

    // TODO
    int bucketWidth = grayImgLeft_.cols/bucketSize + 1;
    int bucketHeight = grayImgLeft_.rows/bucketSize + 1;
    std::vector<std::vector< std::vector<std::tuple<cv::Point2f, int>> >> buckets(bucketHeight, std::vector< std::vector<std::tuple<cv::Point2f, int>> >());
    for(int i = 0; i < keyPoints_.size(); i++)
    {
        cv::Point2f& pt = keyPoints_[i];
        int age = pointAges_[i];
        int c = std::floor(pt.x/bucketWidth);
        int r = std::floor(pt.y/bucketHeight);
        if(buckets[r][c].size() < bucket_size)
            buckets[r][c].push_back(std::make_tuple(pt, age));
        else
        {
            for(int x = 0; x < buckets[r][c].size(); x++)
            {
                if(std::get<1>(buckets[r][c][x]) > 10)
            }
        }
    }

    keyPoints_.clear();
    pointAges_.clear();
    for(int r = 0; r < bucketHeight; r++)
    {
        for(int c = 0; c < bucketHeight; c++)
        {
            auto& bucket = buckets[r][c];
            cv::Point2f bestPt1, bestPt2;
            int bestAge1 = -1, bestAge2 = -1;
            for(auto& kpt: bucket)
            {
                auto pt = std::get<0>(kpt);
                auto age = std::get<1>(kpt);
                if(age < 10)
                {
                    if(age > )
                }

            }
        }
    }

}

}

