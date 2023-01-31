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
    static const int OUTPUT_SIZE = 35;
    static const int BS = 1;
    const char* INPUT_BLOB_NAME = "roi";
    const char* OUTPUT_BLOB_NAME = "prob";
    std::vector<std::string> labelmap = {"乱堆物料", "人骑车_行驶状态", "出店经营", "卖衣服游商", "合规广告类", "合规的早餐车", "垃圾桶未满溢", "垃圾桶满溢", "平摊", "广告误检",
                                         "店内(橱窗内)晾晒", "店内经营", "废弃摊位_不在经营的摊位", "打包垃圾", "撑伞", "无照经营游商", "晒玉米粮食等", "暴露垃圾", "气模拱门",
                                         "沿街晾晒", "渣土堆积", "盆栽花卉", "矩摊", "石头_假山_雕塑_墙体等", "砖块堆积", "篮筐", "经营性物资", "误检", "车上堆积", "车辆无遮挡",
                                         "车辆有遮挡", "运动中游商", "违规不上报广告类", "违规广告类", "遮盖布"};
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

