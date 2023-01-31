cmake_minimum_required(VERSION 2.6)

project(repvgg)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75;)

# tensorrt
include_directories(/home/zhaoming/workspace-cpp/TensorRT/include)
link_directories(/home/zhaoming/workspace-cpp/TensorRT/lib)
set(TRT_LIBS nvinfer nvinfer_plugin nvparsers)

#opencv
set(OPENCV_VERSION 348)
set(OPENCV_INLCUDE_DIR /home/zhaoming/workspace-cpp/opencv/install/include)
set(OPENCV_LIB_DIR /home/zhaoming/workspace-cpp/opencv/install/lib)
set(OPENCV_LIBS opencv_imgcodecs opencv_core opencv_highgui opencv_imgproc opencv_features2d )
include_directories(${OPENCV_INLCUDE_DIR})
link_directories(${OPENCV_LIB_DIR})

add_library(repvgg  SHARED repvgg.cpp)
target_link_libraries(repvgg nvinfer)
target_link_libraries(repvgg cudart)

add_executable(gentrt gen_trt.cpp)
target_link_libraries(gentrt nvinfer)
target_link_libraries(gentrt cudart)
target_link_libraries(gentrt repvgg)

add_executable(testTime test_time.cpp)
target_link_libraries(testTime nvinfer)
target_link_libraries(testTime cudart)
target_link_libraries(testTime repvgg)

add_definitions(-O2 -pthread)

