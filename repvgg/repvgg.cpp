#include "repvgg.h"

std::map<std::string, Weights> RepVGG::loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }
    std::cout << "Finished Load weights: " << file << std::endl;
    return weightMap;
}

IActivationLayer * RepVGG::RepVGGBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch, int stride, int groups, std::string lname)
{
    IConvolutionLayer *conv = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "rbr_reparam.weight"], weightMap[lname + "rbr_reparam.bias"]);
    conv->setStrideNd(DimsHW{stride, stride});
    conv->setPaddingNd(DimsHW{1, 1});
    conv->setNbGroups(groups);
    assert(conv);
    IActivationLayer *relu = network->addActivation(*conv->getOutput(0), ActivationType::kRELU);
    assert(relu);
    return relu;
}

IActivationLayer * RepVGG::makeStage(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, int &layer_idx, const int group_count, ITensor &input, int inch, int outch, int stride, int blocks, std::string lname)
{
    IActivationLayer *layer;
    for (int i = 0; i < blocks; ++i)
    {
        int group = 1;
        if (std::find(groupwise_layers.begin(), groupwise_layers.end(), layer_idx) != groupwise_layers.end())
            group = group_count;
        if (i == 0)
            layer = RepVGGBlock(network, weightMap, input, inch, outch, 2, group, lname + std::to_string(i) + ".");
        else
            layer = RepVGGBlock(network, weightMap, *layer->getOutput(0), inch, outch, 1, group, lname + std::to_string(i) + ".");
        layer_idx += 1;
    }
    return layer;
}
// Creat the engine using only the API and not any parser.
ICudaEngine * RepVGG::createEngine(std::string netName, unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
{
    const std::vector<int> blocks = num_blocks.at(netName);
    const std::vector<float> widths = width_multiplier.at(netName);
    const int group_count = groupwise_counts.at(netName);
    int layer_idx = 1;

    std::map<std::string, Weights> weightMap = loadWeights("../" + netName + ".wts");

    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    int in_planes = std::min(64, int(64 * widths[0]));
    auto stage0 = RepVGGBlock(network, weightMap, *data, 3, in_planes, 2, 1, "stage0.");
    assert(stage0);

    auto stage1 = makeStage(network, weightMap, layer_idx, group_count, *stage0->getOutput(0), in_planes, int(64 * widths[0]), 2, blocks[0], "stage1.");
    assert(stage1);
    auto stage2 = makeStage(network, weightMap, layer_idx, group_count, *stage1->getOutput(0), int(64 * widths[0]), int(128 * widths[1]), 2, blocks[1], "stage2.");
    assert(stage2);
    auto stage3 = makeStage(network, weightMap, layer_idx, group_count, *stage2->getOutput(0), int(128 * widths[1]), int(256 * widths[2]), 2, blocks[2], "stage3.");
    assert(stage3);
    auto stage4 = makeStage(network, weightMap, layer_idx, group_count, *stage3->getOutput(0), int(256 * widths[2]), int(512 * widths[3]), 2, blocks[3], "stage4.");
    assert(stage4);

    IPoolingLayer *pool = network->addPoolingNd(*stage4->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    pool->setStrideNd(DimsHW{7, 7});
    pool->setPaddingNd(DimsHW{0, 0});
    assert(pool);

    IFullyConnectedLayer *linear = network->addFullyConnected(*pool->getOutput(0), OUTPUT_SIZE, weightMap["linear.weight"], weightMap["linear.bias"]);
    assert(linear);

    auto logist = network->addSoftMax(*linear->getOutput(0));

    (logist)->getOutput(0)->setName(OUTPUT_BLOB_NAME);

    logist->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*logist->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);

#ifdef USE_FP16
    std::cout<<" Using FP16 mode "<<std::endl;
    config->setFlag(BuilderFlag::kFP16);
//    builder->setFp16Mode(true);
#endif

    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }
    return engine;
}

void RepVGG::APIToModel(std::string netName, unsigned int maxBatchSize, IHostMemory **modelStream)
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = createEngine(netName, maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void RepVGG::doInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}




