cmake_minimum_required(VERSION 3.16)
project(yolo_trt_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(yolo_trt_worker yolo_trt_detector.cpp)

target_include_directories(
  yolo_trt_worker
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu
    /usr/local/cuda/targets/aarch64-linux/include
)

target_link_directories(
  yolo_trt_worker
  PRIVATE
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/targets/aarch64-linux/lib
)

target_link_libraries(
  yolo_trt_worker
  PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    cudart
)
