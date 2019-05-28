
find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS} )

include_directories(evaluate)


add_library( Odometry
 "feature.cpp"
 "bucket.cpp"
 "utils.cpp"
 "visualOdometry.cpp"
 "Frame.cpp"
 "evaluate/matrix.cpp"
 "evaluate/evaluate_odometry.cpp"
 "evaluate/pointClouds.cpp"
 )


add_executable( kitti_demo main.cpp )

target_link_libraries( Odometry ${OpenCV_LIBS} )
target_link_libraries( kitti_demo ${OpenCV_LIBS} Odometry )
