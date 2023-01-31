//
// Created by zhaoming on 2023/1/18.
//
#include "eventCls.h"
std::vector<float> AlgEventCLS::prepareImage(cv::Mat img) {
    std::vector<float> result(BS * 3 * INPUT_H * INPUT_W);
    float* data = result.data();
    // 数据预处理
    cv::Mat rsz_img;
    cv::resize(img, rsz_img, cv::Size(224, 224));
    cv::Mat flt_img(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));

    // 手撕三通道，填入data
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = flt_img.data + row * flt_img.step;
//        std::cout << (float)uc_pixel[0] << std::endl;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = ((float)uc_pixel[2]/255. - 0.5)/0.5;
            data[ i + INPUT_H * INPUT_W] = ((float)uc_pixel[1]/255.-0.5)/0.5;
            data[ i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[0]/255.-0.5)/0.5;
            uc_pixel += 3;
            ++i;
        }
    }
    return result;
}




AlgEventCLS::AlgEventCLS(){
    loadModel(netName);
};
AlgEventCLS::~AlgEventCLS(){
    // Destroy the engine
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
};

void AlgEventCLS::getEvents(cv::Mat& input, PredRes *results){
    std::vector<float> res = prepareImage(input);


    float out[OUTPUT_SIZE];
//    float data[3 * model.INPUT_H * model.INPUT_W];

    float *data = res.data();

//    for (int i = 0; i < 3 * model.INPUT_H * model.INPUT_W; i++)
//        std::cout << data[i] << std::endl;

    model.doInference(*context_, data, out, 1);

    int maxidx = 0;
    float maxconf = 0.0;

    for(int k =0; k<OUTPUT_SIZE;k++)
    {
        std::cout << out[k] << std::endl;
        if( out[k] > maxconf)
        {
            maxidx = k;
            maxconf = out[k];
        }
    }
    results->idx = maxidx;
    results->cls = labelmap[maxidx];
    results->score = maxconf;

}

void AlgEventCLS::loadModel(std::string netName){
    char *trtModelStream{nullptr};
    size_t size{0};

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

    runtime_ = createInferRuntime(gLogger);
    assert(runtime_ != nullptr);
    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);
    delete[] trtModelStream;
}
