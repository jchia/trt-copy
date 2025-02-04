#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include <boost/lexical_cast.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "addPlugin.hpp"

using namespace std;
namespace nvi = nvinfer1;

class Logger : public nvi::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            cout << msg << '\n';
        }
    }
} gLogger;

bool buildEngine(char const* onnxFilePath, nvi::IHostMemory** engineStream) {
    auto builder = nvi::createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(
        1U << static_cast<int>(nvi::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
    rgr::AddPluginCreator pluginCreator;

    nvi::IPluginRegistry& registry = builder->getPluginRegistry();
    if (!registry.registerCreator(pluginCreator, pluginCreator.getPluginNamespace()))
        throw runtime_error("Error registering plugin creator.");
    nvi::IBuilderConfig* config = builder->createBuilderConfig();

    ifstream modelFile(onnxFilePath);
    modelFile.seekg(0, modelFile.end);
    int64_t numBytes = modelFile.tellg();
    modelFile.seekg(0, modelFile.beg);
    unique_ptr<char[]> modelBytes = make_unique<char[]>(numBytes);
    modelFile.read(modelBytes.get(), numBytes);
    if (modelFile.gcount() < numBytes) {
        cerr << "ERROR: Insufficient input ONNX file size, expected " << numBytes << ", got "
             << modelFile.gcount() << ".\n";
        return false;
    }

    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parse(modelBytes.get(), numBytes)) {
        cerr << "ERROR: Could not parse ONNX file to TensorRT network.\n";
        return false;
    }
    config->setProfilingVerbosity(nvi::ProfilingVerbosity::kDETAILED);
    config->setBuilderOptimizationLevel(3);
    config->setMaxAuxStreams(8);
    *engineStream = builder->buildSerializedNetwork(*network, *config);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Please specify input ONNX model filename and output TensorRT engine filename.\n";
        return 0;
    }

    char const* const onnxFilePath = argv[1];
    char const* const engineFilePath = argv[2];
    nvinfer1::IHostMemory* engineStream = nullptr;

    if (!buildEngine(onnxFilePath, &engineStream)) {
        cerr << "Failed to build the engine.\n";
        return 1;
    }
    if (!engineStream) {
        cerr << "engineStream unexpectedly null.\n";
        return 2;
    }
    ofstream outFile(engineFilePath, ios::binary | ios::out);
    outFile.write(static_cast<char*>(engineStream->data()), engineStream->size());
    if (outFile.fail()) {
        cerr << "Failed to write engine file.\n";
        return 3;
    }
    return 0;
}
