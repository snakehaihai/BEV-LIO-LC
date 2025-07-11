cmake_minimum_required(VERSION 2.8.3)
project(bev_lio_lc)

SET(CMAKE_BUILD_TYPE "Debug")

ADD_COMPILE_OPTIONS(-std=c++17 )
ADD_COMPILE_OPTIONS(-std=c++17 )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3" )

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -std=c++0x -std=c++17 -fexceptions")



message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

# libtorch change to your path here
set(Torch_DIR "home/chx/Downloads/libtorch")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/chx/Downloads/libtorch")
find_package(Torch REQUIRED)

# CUDA
find_package(CUDA REQUIRED)

# gtsam
find_package( GTSAMCMakeTools )
find_package(GTSAM REQUIRED QUIET)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
  cv_bridge
  genmsg
)

find_package(OpenCV REQUIRED
    COMPONENTS core imgproc features2d xfeatures2d
)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer thread serialization chrono)
find_package(Boost REQUIRED COMPONENTS system filesystem date_time)
find_package(FLANN REQUIRED)
message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${FLANN_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_LIBS}
  # change to your path here
  /home/chx/Downloads/libtorch/include
  /home/chx/Downloads/libtorch/include/torch/csrc/api/include
  include)

add_library(${PROJECT_NAME}
  include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp
  src/image_processing.cpp
  # src/imu_processing.cpp
  src/bev_feature.cpp
  # src/timing.cpp
  # src/use_ikfom.cpp
  src/BEVprojector.cpp
)

add_message_files(
  FILES
  Pose6D.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  DEPENDS EIGEN3 PCL GTSAM
  INCLUDE_DIRS
)
add_executable(bevlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/preprocess.cpp)
target_link_libraries(bevlio_mapping ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES} ${FLANN_LIBRARIES} ${OpenCV_LIBS} gtsam)
target_include_directories(bevlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
add_dependencies(bevlio_mapping fast_lio_generate_messages_cpp)