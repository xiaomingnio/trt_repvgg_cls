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
        std::cerr << "./gentrt RepVGG-A0 // serialize model to plan file" << std::endl;
        return -1;
    }

    size_t size{0};


    std::string netName = std::string(argv[1]);
    IHostMemory *modelStream{nullptr};
    model.APIToModel(netName, MAX_BATCH_SIZE, &modelStream);
    assert(modelStream != nullptr);

    std::ofstream p(netName + ".engine", std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;


}

