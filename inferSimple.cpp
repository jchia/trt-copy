#include <cstring>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include "addPlugin.hpp"

using namespace std;
namespace nvi = nvinfer1;

class Logger : public nvi::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity != Severity::kINFO)
            cout << msg << '\n';
    }
} gLogger;

struct Row {
    float x[37];
    int32_t trigger;
};

uint32_t const kWarmupNumIters = 20;
uint32_t const kInferNumIters = 20000;
uint32_t const kBenchNumIters = 5000;
uint32_t const kProfileNumIters = 5000;

int main(int argc, char** argv) {
    bool wantHelp = false;
    if (argc != 2)
        wantHelp = true;
    else {
        if (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))
            wantHelp = true;
    }
    if (wantHelp) {
        cout << "Usage: " << argv[0] << " engine-filename\n";
        cout << "\tengine-filename:\t\tPath to the TensorRT engine file.\n";
        return 1;
    }

    rgr::AddPluginCreator pluginCreator;

    ifstream predictorFile(argv[1], ios::binary);
    if (!predictorFile) {
        cerr << "Failed to open predictor file: " << argv[1] << '\n';
        return 2;
    }
    predictorFile.seekg(0, predictorFile.end);
    int64_t predictorSize = predictorFile.tellg();
    if (predictorSize < 0) {
        cerr << "Error reading predictor file.\n";
        return 3;
    }
    predictorFile.seekg(0, predictorFile.beg);

    vector<char> predictorBytes(predictorSize);
    predictorFile.read(predictorBytes.data(), predictorSize);
    predictorFile.close();

    unique_ptr<nvi::IRuntime> runtime(nvi::createInferRuntime(gLogger));
    auto& registry = runtime->getPluginRegistry();
    if (!registry.registerCreator(pluginCreator, pluginCreator.getPluginNamespace()))
        throw runtime_error("Error registering plugin creator.");
    unique_ptr<nvi::ICudaEngine> engine(runtime->deserializeCudaEngine(predictorBytes.data(), predictorSize));
    if (!engine) {
        cerr << "Error deserializing engine file.\n";
        return 4;
    }

    int32_t const kNumTensors = 2;
    if (engine->getNbIOTensors() != 2)
        return 6;
    array<void*, 2> buffers;
    array<size_t, kNumTensors> sizes;
    uint32_t numEl;
    for (uint32_t i = 0; i < kNumTensors; ++i) {
        char const* tensorName = engine->getIOTensorName(i);
        nvi::TensorLocation location = engine->getTensorLocation(tensorName);
        nvi::DataType type = engine->getTensorDataType(tensorName);
        nvi::TensorFormat format = engine->getTensorFormat(tensorName);
        nvi::TensorIOMode ioMode = engine->getTensorIOMode(tensorName);
        nvi::Dims dims = engine->getTensorShape(tensorName);

        if (strcmp(i == 0 ? "input" : "output", tensorName))
            return 7;
        if (location != nvi::TensorLocation::kDEVICE)
            return 8;
        if (nvi::DataType::kFLOAT != type)
            return 9;
        if (ioMode != (i ? nvi::TensorIOMode::kOUTPUT : nvi::TensorIOMode::kINPUT))
            return 10;
        if (format != nvi::TensorFormat::kLINEAR)
            return 11;

        uint32_t numElLocal = 1;
        for (int32_t j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] < 0)
                return 12;
            numElLocal *= static_cast<uint32_t>(dims.d[j]);
        }
        if (i) {
            if (numElLocal != numEl)
                return 12;
        } else {
            numEl = numElLocal;
        }
    }
    for (uint32_t i = 0; i < 2; ++i) {
        size_t const size = numEl * sizeof(float);
        cudaMalloc(&buffers[i], size);
        cudaMemset(buffers[i], 0, size);
    }
    float* input;
    float* output;
    cudaMallocHost(reinterpret_cast<void**>(&input), numEl * sizeof(float));
    cudaMallocHost(reinterpret_cast<void**>(&output), numEl * sizeof(float));

    unique_ptr<nvi::IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        cerr << "Failed to create execution context." << '\n';
        return 14;
    }

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    context->setTensorAddress("input", buffers[0]);
    context->setOutputTensorAddress("output", buffers[1]);

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal))
        return -1;
    if (!context->enqueueV3(stream))
        return -2;
    if (cudaStreamEndCapture(stream, &graph))
        return -4;
    if (cudaGraphInstantiate(&instance, graph, 0))
        return -5;

    auto runInference = [&]() -> bool {
        if (cudaMemcpyAsync(buffers[0], input, numEl * sizeof(float), cudaMemcpyHostToDevice, stream))
            return false;
        if (cudaGraphLaunch(instance, stream))
            return false;
        if (cudaMemcpyAsync(output, buffers[1], numEl * sizeof(float), cudaMemcpyDeviceToHost, stream))
            return false;
        if (cudaStreamSynchronize(stream))
            return false;
        return true;
    };


    for (uint32_t i = 0; i < 10; ++i) {
        fill(input, input + numEl, i * i);
        if (!runInference())
            return 15;
        cout << "ITER " << i;
        for (uint32_t j = 0; j < numEl; ++j)
            cout << ' ' << output[j];
        cout << '\n';
    }
    return 0;
}
