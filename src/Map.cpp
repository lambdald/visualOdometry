#include "Map.h"

namespace MVSO
{
	void MVSO::Map::addNewFrame(std::shared_ptr<Frame> frame)
	{
		this->frames_.push_back(frame);
	}

	std::vector<std::shared_ptr<Frame>> MVSO::Map::getNewestFrames(int num)
	{
		if (frames_.size() < num)
			return frames_;
		else
		{
			decltype(frames_) fs;
			for (int i = 1; i <= num; i++)
			{
				fs.push_back(frames_[frames_.size() - i]);
			}
			return fs;
		}
	}
}