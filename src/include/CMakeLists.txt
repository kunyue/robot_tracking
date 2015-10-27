cmake_minimum_required(VERSION 2.8)
project( robotTracking )
find_package( OpenCV REQUIRED )
find_package(Eigen3  REQUIRED)


set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)


message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    lib path: ${OpenCV_LIB_DIR}")
add_executable( robotTracking demo.cpp robotTracking.cpp util.cpp robotFeature.cpp lap.cpp)
target_link_libraries( robotTracking ${OpenCV_LIBS} )
