cmake_minimum_required(VERSION 3.0)

project(VisualOdometry)


find_package(Eigen3 NO_MODULE)
include_directories ( ${EIGEN3_INCLUDE_DIRS} )

message(STATUS "EIGENPATH: " ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
