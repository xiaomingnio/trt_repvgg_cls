cmake_minimum_required(VERSION 2.6)

project(AlgEventCLS)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

# add_definitions(-DCVT_TO_GIE_MODEL) # convert tensorrt model

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

include_directories(./include)
link_directories(./src)
SET(SRCLIST ./test.cpp
        ./src/eventCls.cpp)

#include
include_directories(./repvgg)
link_directories(./repvgg/cmake-build-debug)
find_library(ENGINE_LIB librepvgg.so ./repvgg/cmake-build-debug)


add_executable(main ${SRCLIST})

set(LIBRARIES nvinfer  cudart cudart cublas curand opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)

target_link_libraries(main ${LIBRARIES} ${ENGINE_LIB})

add_definitions(-O2 -pthread)

