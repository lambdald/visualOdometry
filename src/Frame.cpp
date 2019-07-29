#include "Frame.h"
#include "utils.h"
#include <exception>
namespace MVSO {

	int Frame::FRAME_COUNT = 0;

	Frame::Frame(cv::Mat imgLeft, cv::Mat imgRight) : frameId_(FRAME_COUNT++)
	{
		if (imgLeft.channels() == 1 && imgRight.channels() == 1)
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
		if (keyPoints_.size() < 2000)
		{
			std::vector<cv::Point2f>  points_new;
			featureDetection(points_new);
			keyPoints_.insert(keyPoints_.end(), points_new.begin(), points_new.end());

			pointAges_.resize(keyPoints_.size(), -1);
			baseKeyPointIndex_.resize(keyPoints_.size(), -1);
		}
	}

	void Frame::featureDetection(std::vector<cv::Point2f>& points)
	{
		std::vector<cv::KeyPoint> keypoints;
		int fast_threshold = 23;
		bool nonmaxSuppression = true;
		cv::FAST(grayImgLeft_, keypoints, fast_threshold, nonmaxSuppression);
		cv::KeyPoint::convert(keypoints, points, std::vector<int>());
	}

	std::vector<cv::Point2f> Frame::getKeypoints()
	{
		return keyPoints_;
	}

	std::vector<cv::Point3f> Frame::getKeypoints3D()
	{
		return keypoints3D_;
	}

	void Frame::bucketingFeature(int bucket_size)
	{
		int ageThresh = 10;
		int bucketWidth = grayImgLeft_.cols / bucketSize + 1;
		int bucketHeight = grayImgLeft_.rows / bucketSize + 1;

		std::vector<std::vector<int> > buckets(bucketHeight*bucketWidth);
		std::cout << buckets.size() << std::endl;
		for (int i = 0; i < keyPoints_.size(); i++)
		{
			cv::Point2f pt = keyPoints_[i];
			if (pt.x > grayImgLeft_.cols || pt.y > grayImgLeft_.rows)
				continue;
			int age = pointAges_[i];
			int c = std::floor(pt.x / bucketSize);
			int r = std::floor(pt.y / bucketSize);
			//if (buckets[r][c].size() < bucket_size)
			buckets[r*bucketWidth+c].push_back(i);
			//count++;

		}

		std::vector<cv::Point2f> newKeypoints;
		std::vector<int> newAges;
		std::vector<int> newKeypointIndex;
		//newKeypoints.reserve(count);
		//newAges.reserve(count);
		//newKeypointIndex.reserve(count);
		for (int r = 0; r < bucketHeight; r++)
		{
			for (int c = 0; c < bucketWidth; c++)
			{

				auto& ids = buckets[r*bucketWidth + c];
				if (ids.size() <= bucket_size)
				{
					for (int i : ids)
					{
						newKeypoints.push_back(keyPoints_[i]);
						newAges.push_back(pointAges_[i]);
						newKeypointIndex.push_back(baseKeyPointIndex_[i]);
					}
				}
				else {
					int best = ids[0];
					int bestScore = abs(pointAges_[best] - ageThresh);
					for (int i : ids)
					{
						int age = pointAges_[i];
						if (abs(age - ageThresh) < bestScore)
						{
							best = i;
							bestScore = abs(age - ageThresh);
						}
					}

					newKeypoints.push_back(keyPoints_[best]);
					newAges.push_back(pointAges_[best]);
					newKeypointIndex.push_back(baseKeyPointIndex_[best]);
				}
				


			}
		}

		keyPoints_ = newKeypoints;
		pointAges_ = newAges;
		baseKeyPointIndex_ = newKeypointIndex;
	}

	void Frame::removeInvalidNewFeature(std::vector<bool>& status)
	{
		for (int i = 0; i < status.size(); i++)
		{
			if (pointAges_[i] != -1)
			{
				status[i] = true;
			}
		}

		removeInvalidElement(keyPoints_, status);
		removeInvalidElement(pointAges_, status);
		removeInvalidElement(baseKeyPointIndex_, status);
	}

	void Frame::addStereoMatch(std::vector<cv::Point2f>& keypoints, std::vector<cv::Point3f>& keypoints3D)
	{
		keyPoints_ = keypoints;
		keypoints3D_ = keypoints3D;
	}

	void Frame::addStereoMatch(std::vector<cv::Point2f>& keypoints, cv::Mat & keypoints3D)
	{
		keyPoints_ = keypoints;
		keypoints3D_ = std::vector<cv::Point3f>(keypoints3D);
	}

	void Frame::setInterframeMatching(std::vector<int>& matchId, Frame* refFrame)
	{
		baseKeyPointIndex_ = matchId;
		pointAges_ = std::vector<int>(matchId.size(), -1);
		for (int i = 0; i < matchId.size(); i++)
		{
			if (matchId[i] != -1)
				pointAges_[i] = refFrame->pointAges_[matchId[i]] + 1;
		}
	}

}

