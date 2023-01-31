//
// Created by zhaoming on 2023/1/19.
//
#include "repvgg.h"
int main(int argc, char **argv)
{
    RepVGG model = RepVGG();
    if (argc != 2)
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./testTime RepVGG-A0 // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    std::string netName = std::string(argv[1]);
    std::ifstream file(netName + ".engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    static float data[3 * model.INPUT_H * model.INPUT_W];
    for (int i = 0; i < 3 * model.INPUT_H * model.INPUT_W; i++)
        data[i] = 1.0;

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[model.OUTPUT_SIZE];
    // warm up
    model.doInference(*context, data, prob, 1);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1000; i++) {
        model.doInference(*context, data, prob, 1);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000 << " us" << std::endl;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < model.OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
    }
    std::cout << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
        std::cout << prob[model.OUTPUT_SIZE - 10 + i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}

