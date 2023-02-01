//
// Created by zhaoming on 2023/1/18.
//
#ifndef ALGEVENTCLS_H
#define ALGEVENTCLS_H
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
//#include "plugin_factory.h"
#include "repvgg.h"
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
using namespace nvinfer1;

typedef struct tag_predRes
{
    int idx;
    std::string cls;
    float score;
} PredRes;

class AlgEventCLS
{
public:
    AlgEventCLS();
    ~AlgEventCLS();
public:
    void getEvents(cv::Mat& input, PredRes *results);

private:

    static const int INPUT_H = 224;
    static const int INPUT_W = 224;
    static const int OUTPUT_SIZE = 14;
    static const int BS = 1;
    const char* INPUT_BLOB_NAME = "roi";
    const char* OUTPUT_BLOB_NAME = "prob";
    std::vector<std::string> labelmap = {"000-one", "001-five", "002-fist", "003-ok", "004-heartSingle", "005-yearh",
                                         "006-three", "007-four","008-six", "009-Iloveyou", "010-gun", "011-thumbUp", "012-nine", "013-pink"};
    Logger gLogger_;
    IRuntime* runtime_;
    IExecutionContext *context_;
    ICudaEngine* engine_;
    cudaStream_t stream_;

    RepVGG model = RepVGG();

    std::string netName = "RepVGG-A0";

    void loadModel(std::string netName);

    std::vector<float> prepareImage(cv::Mat vec_img);
};

#endif // ALGEVENTCLS_H

