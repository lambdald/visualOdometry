cmake_minimum_required(VERSION 3.0)

project(VisualOdometry)


add_definitions     ( ${EIGEN3_DEFINITIONS} )
include_directories ( ${EIGEN3_INCLUDE_DIRS} )

message(STATUS "EIGENPATH: " ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
