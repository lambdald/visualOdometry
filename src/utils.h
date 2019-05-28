#ifndef UTILS_H
#define UTILS_H
#define _CRT_SECURE_NO_WARNINGS

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>

#include "feature.h"
#include "evaluate/matrix.h"


// º∆ ±£¨∑µªÿ∫¡√Î
class TicTok
{
public:
	TicTok()
	{
		start = std::chrono::steady_clock::now();
	}
	long long tok()
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
	}
private:
	std::chrono::steady_clock::time_point start;
};



// --------------------------------
// Visualization
// --------------------------------
void drawFeaturePoints(cv::Mat image, std::vector<cv::Point2f>& points);

void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, std::vector<Matrix>& pose_matrix_gt, float fps, bool showgt);



// --------------------------------
// Transformation
// --------------------------------
void integrateOdometryStereo(int frame_id, cv::Mat& rigid_body_transformation, cv::Mat& frame_pose, const cv::Mat& rotation, 
                            const cv::Mat& translation_stereo);

bool isRotationMatrix(cv::Mat &R);

cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);

// --------------------------------
// I/O
// --------------------------------

void loadImageLeft(cv::Mat& image_color, cv::Mat& image_gary, int frame_id, const std::string& filepath);

void loadImageRight(cv::Mat& image_color, cv::Mat& image_gary, int frame_id, const std::string& filepath);

void loadGyro(std::string filename, std::vector<std::vector<double>>& time_gyros);
// read time gyro txt file with format of timestamp, gx, gy, gz

#endif
