#ifndef FEATURE_H
#define FEATURE_H

#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>


struct FeaturePoint{
  cv::Point2f  point;
  int id;
  int age;
};

struct FeatureSet {
    std::vector<cv::Point2f>  points;
    std::vector<int>  ages;
    int size(){
        return points.size();
    }
    void clear(){
        points.clear();
        ages.clear();
    }
 };


void deleteUnmatchFeatures(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1, std::vector<uchar>& status);

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f>& points);

void featureDetectionGoodFeaturesToTrack(cv::Mat image, std::vector<cv::Point2f>& points);

void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status);

void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
                          std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
                          std::vector<cv::Point2f>& points0_return,
                          std::vector<uchar>& status0, std::vector<uchar>& status1,
                          std::vector<uchar>& status2, std::vector<uchar>& status3,
                          std::vector<int>& ages);

void circularMatching(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                      std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                      std::vector<cv::Point2f>& points_l_0_return,
                      FeatureSet& current_features);

void bucketingFeatures(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket);

void appendNewFeatures(cv::Mat& image, FeatureSet& current_features);

void appendNewFeatures(std::vector<cv::Point2f> points_new, FeatureSet& current_features);

#endif
