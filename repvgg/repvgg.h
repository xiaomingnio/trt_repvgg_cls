//
// Created by zhaoming on 2023/1/19.
//

#ifndef ALGEVENTCLS_REPVGG_H
#define ALGEVENTCLS_REPVGG_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
//#include <opencv2/opencv.hpp>

#define USE_FP16

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

// stuff we know about the network and the input/output blobs
#define MAX_BATCH_SIZE 1

using namespace nvinfer1;

static Logger gLogger;

class RepVGG{
private:
    const std::vector<int> groupwise_layers{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26};
    const std::map<std::string, int> groupwise_counts = {
            {"RepVGG-A0", 1},
            {"RepVGG-A1", 1},
            {"RepVGG-A2", 1},
            {"RepVGG-B0", 1},
            {"RepVGG-B1", 1},
            {"RepVGG-B1g2", 2},
            {"RepVGG-B1g4", 4},
            {"RepVGG-B2", 1},
            {"RepVGG-B2g2", 2},
            {"RepVGG-B2g4", 4},
            {"RepVGG-B3", 1},
            {"RepVGG-B3g2", 2},
            {"RepVGG-B3g4", 4}};
    const std::map<std::string, std::vector<int>> num_blocks = {
            {"RepVGG-A0", {2, 4, 14, 1}},
            {"RepVGG-A1", {2, 4, 14, 1}},
            {"RepVGG-A2", {2, 4, 14, 1}},
            {"RepVGG-B0", {4, 6, 16, 1}},
            {"RepVGG-B1", {4, 6, 16, 1}},
            {"RepVGG-B1g2", {4, 6, 16, 1}},
            {"RepVGG-B1g4", {4, 6, 16, 1}},
            {"RepVGG-B2", {4, 6, 16, 1}},
            {"RepVGG-B2g2", {4, 6, 16, 1}},
            {"RepVGG-B2g4", {4, 6, 16, 1}},
            {"RepVGG-B3", {4, 6, 16, 1}},
            {"RepVGG-B3g2", {4, 6, 16, 1}},
            {"RepVGG-B3g4", {4, 6, 16, 1}}};
    const std::map<std::string, std::vector<float>> width_multiplier = {
            {"RepVGG-A0", {0.75, 0.75, 0.75, 2.5}},
            {"RepVGG-A1", {1, 1, 1, 2.5}},
            {"RepVGG-A2", {1.5, 1.5, 1.5, 2.75}},
            {"RepVGG-B0", {1, 1, 1, 2.5}},
            {"RepVGG-B1", {2, 2, 2, 4}},
            {"RepVGG-B1g2", {2, 2, 2, 4}},
            {"RepVGG-B1g4", {2, 2, 2, 4}},
            {"RepVGG-B2", {2.5, 2.5, 2.5, 5}},
            {"RepVGG-B2g2", {2.5, 2.5, 2.5, 5}},
            {"RepVGG-B2g4", {2.5, 2.5, 2.5, 5}},
            {"RepVGG-B3", {3, 3, 3, 5}},
            {"RepVGG-B3g2", {3, 3, 3, 5}},
            {"RepVGG-B3g4", {3, 3, 3, 5}}};



//    std::string wts_path = "../RepVGG-A0.wts";

    std::map<std::string, Weights> loadWeights(const std::string file);
    IActivationLayer *RepVGGBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch, int stride, int groups, std::string lname);
    IActivationLayer *makeStage(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, int &layer_idx, const int group_count, ITensor &input, int inch, int outch, int stride, int blocks, std::string lname);
    ICudaEngine *createEngine(std::string netName, unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt);

public:

    static const int INPUT_H = 224;
    static const int INPUT_W = 224;
    static const int OUTPUT_SIZE = 35;

    const char *INPUT_BLOB_NAME = "data";
    const char *OUTPUT_BLOB_NAME = "prob";
    void APIToModel(std::string netName, unsigned int maxBatchSize, IHostMemory **modelStream);
    void doInference(IExecutionContext &context, float *input, float *output, int batchSize);
};


#endif //ALGEVENTCLS_REPVGG_H
