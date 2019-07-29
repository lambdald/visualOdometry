#include "PoseEstimator.h"
#include "PoseOptimizer.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <opencv2/core/eigen.hpp>

namespace MVSO
{
	void solveICP(std::vector<cv::Point3f>& points3D_t0, std::vector<cv::Point3f>& points3D_t1, cv::Mat& R, cv::Mat& t)
	{
		cv::Point3f p1, p2;     // center of mass
		int N = points3D_t0.size();
		for (int i = 0; i < N; i++)
		{
			p1 += points3D_t0[i];
			p2 += points3D_t1[i];
		}
		p1 = cv::Point3f(cv::Vec3f(p1) / N);
		p2 = cv::Point3f(cv::Vec3f(p2) / N);
		std::vector<cv::Point3f>     q1(N), q2(N); // remove the center
		for (int i = 0; i < N; i++)
		{
			q1[i] = points3D_t0[i] - p1;
			q2[i] = points3D_t1[i] - p2;
		}

		// compute q1*q2^T
		Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
		for (int i = 0; i < N; i++)
		{
			W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
		}

		// SVD on W
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d V = svd.matrixV();

		if (U.determinant() * V.determinant() < 0)
		{
			for (int x = 0; x < 3; ++x)
			{
				U(x, 2) *= -1;
			}
		}

		Eigen::Matrix3d R_ = U * (V.transpose());
		Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

		cv::eigen2cv(R_, R);
		cv::eigen2cv(t_, t);
	}


	void solveICPRansac(std::vector<cv::Point3f>& pts1, std::vector<cv::Point3f>& pts2,
		cv::Mat& R, cv::Mat& t,std::vector<int>& inliers,
		int iterationCount, float reprojectionError)
	{
		cv::Mat bestR = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
		cv::Mat bestT = (cv::Mat_<double>(3, 1) << 0, 0, 0);
		std::vector<int> bestInliers;
		cv::Mat ptsMat1(pts1);
		cv::Mat ptsMat2(pts2);
		for (int it = 0; it < iterationCount; it++)
		{
			std::vector<cv::Point3f> subset1;
			std::vector<cv::Point3f> subset2;
			for (int i = 0; i < 5; i++)
			{
				int id = rand() % pts1.size();

				subset1.push_back(pts1[id]);
				subset2.push_back(pts2[id]);
			}

			cv::Mat tmpR, tmpT;
			solveICP(subset1, subset2, tmpR, tmpT);

			std::vector<int> inliers;
			for (int i = 0; i < pts1.size(); i++)
			{
				cv::Point3f& p1 = pts1[i];
				cv::Point3f& p2 = pts2[i];
				cv::Point3f ep2;
	
				ep2.x = tmpR.at<double>(0, 0) * p1.x + tmpR.at<double>(0, 1) * p1.y + tmpR.at<double>(0, 2) * p1.z + tmpT.at<double>(0);
				ep2.y = tmpR.at<double>(1, 0) * p1.x + tmpR.at<double>(1, 1) * p1.y + tmpR.at<double>(1, 2) * p1.z + tmpT.at<double>(1);
				ep2.z = tmpR.at<double>(2, 0) * p1.x + tmpR.at<double>(2, 1) * p1.y + tmpR.at<double>(2, 2) * p1.z + tmpT.at<double>(2);

				float err = sqrt((p2.x - ep2.x)*(p2.x - ep2.x) + (p2.y - ep2.y)*(p2.y - ep2.y) + (p2.z - ep2.z));
				if (err < reprojectionError)
				{
					inliers.push_back(i);
				}
			}
			if (inliers.size() > bestInliers.size())
			{
				bestInliers = inliers;
				bestR = tmpR;
				bestT = tmpT;
			}

		}
		R = bestR;
		t = bestT;
		inliers = bestInliers;
	}


	MVSO::PoseEstimator::PoseEstimator(CameraModel & camera): camera_(camera)
	{
	}

	cv::Mat PoseEstimator::estimatePose(std::vector<cv::Point2f>& pointsLeft_t0, std::vector<cv::Point2f>& pointsLeft_t1, std::vector<cv::Point3f>& points3D_t0)
	{
		// Calculate frame to frame transformation
		cv::Mat pose;
		cv::Mat rotation, translation;

		// -----------------------------------------------------------
		// Rotation(R) estimation using Nister's Five Points Algorithm
		// -----------------------------------------------------------
		double focal = camera_.fx_;
		cv::Point2d principle_point(camera_.cx_, camera_.cy_);

		//recovering the pose and the essential cv::matrix
		cv::Mat E, mask;
		cv::Mat translation_mono = cv::Mat::zeros(3, 1, CV_64F);
		E = cv::findEssentialMat(pointsLeft_t1, pointsLeft_t0, focal, principle_point, cv::RANSAC, 0.999, 1.0, mask);
		cv::recoverPose(E, pointsLeft_t1, pointsLeft_t0, rotation, translation_mono, focal, principle_point, mask);
		// std::cout << "recoverPose rotation: " << rotation << std::endl;

		// ------------------------------------------------
		// Translation (t) estimation by use solvePnPRansac
		// ------------------------------------------------
		cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
		cv::Mat inliers;
		cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);

		int iterationsCount = 100;        // number of Ransac iterations.
		float reprojectionError = 1.0;    // maximum allowed distance to consider it an inlier.
		float confidence = 0.98;          // RANSAC successful confidence.
		bool useExtrinsicGuess = false;
		int flags = cv::SOLVEPNP_EPNP;

		//cv::Rodrigues(rotation, rvec);
		cv::solvePnPRansac(points3D_t0, pointsLeft_t1, camera_.intrinsicMat_, distCoeffs, rvec, translation,
			useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
			inliers, flags);


		std::vector<cv::Point3f> points3d;
		std::vector<cv::Point2f> points2d;
		std::vector<double> weights;

		for (int i = 0; i < inliers.rows; i++)
		{
			int id = inliers.at<int>(i, 0);

			cv::Point3f p3d = points3D_t0[id];
			cv::Point2f p0 = pointsLeft_t0[id], p1 = pointsLeft_t1[id];
			points3d.push_back(p3d);
			points2d.push_back(p1);

			double w = sqrt((p0.x - p1.x)*(p0.x - p1.x) + (p0.y - p1.y)*(p0.y - p1.y));
			//w *= 1/(1+exp(abs(p3d.z - 5)));
			if (w > 7)
				w = 7;
			else if (w < 1.0)
				w = 1.0;
			weights.push_back(exp(-w));
		}

		PoseOptimizer optimizer(camera_);
		//cv::Rodrigues(rvec, rotation);
		//optimizer.optimizePose(points3d, points2d, rotation, translation);
		optimizer.optimizePose(points3d, points2d, weights,rotation, translation);


		//cv::Rodrigues(rvec, rotation);
		rotation = rotation.t();
		translation = -translation;
		// std::cout << "inliers size: " << inliers.size() << std::endl;
		cv::hconcat(rotation, translation, pose);
		return pose;
	}

	cv::Mat PoseEstimator::estimatePose(std::vector<cv::Point2f>& points_t0, std::vector<cv::Point2f>& points_t1, std::vector<cv::Point3f>& points3D_t0, std::vector<cv::Point3f>& points3D_t1)
	{
		
		cv::Mat R, t;
		std::vector<int> inliers;
		solveICPRansac(points3D_t0, points3D_t1, R, t, inliers, 100, 1.0);

		std::vector<cv::Point3f> pts1, pts2;
		for (int n : inliers)
		{
			pts1.push_back(points3D_t0[n]);
			pts2.push_back(points3D_t1[n]);
		}


		PoseOptimizer optimizer(camera_);
		optimizer.optimizePose(pts1, pts2, R, t);
		cv::Mat pose;
		cv::hconcat(R, t, pose);
		return pose;

	}

	PoseEstimator::~PoseEstimator()
	{
	}

}

