
find_package( Eigen3 NO_MODULE )

find_package( OpenCV REQUIRED )

#find_package( g2o REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(${EIGNE3_INCLUDE_DIRS})

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
 "cameramodel.cpp"
 "PoseEstimator.cpp"
 "PoseOptimizer.cpp"
 )


add_executable( kitti_demo main.cpp )

target_link_libraries( Odometry ${OpenCV_LIBS} )
target_link_libraries( kitti_demo ${OpenCV_LIBS} Odometry g2o_core g2o_stuff g2o_types_sba g2o_solver_eigen g2o_types_slam3d)
