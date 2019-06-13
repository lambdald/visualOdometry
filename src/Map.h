#ifndef MAP_H
#define MAP_H

#include <vector>
#include <memory>

#include "Frame.h"

namespace MVSO
{
	class Map
	{
	public:
		Map() = default;
		void addNewFrame(std::shared_ptr<Frame> frame);
		std::vector<std::shared_ptr<Frame>> getNewestFrames(int num=1);
	private:
		std::vector<std::shared_ptr<Frame>> frames_;
	};

}

#endif