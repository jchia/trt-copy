#include "registerPlugin.hpp"
#include "addPlugin.hpp"

namespace {
    static rgr::AddPluginCreator gAddPluginCreator;
    static nvinfer1::IPluginCreatorV3One* gPluginCreators[] = { &gAddPluginCreator };
}

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder) {
}

extern "C" nvinfer1::IPluginCreatorV3One* const* getCreators(int32_t& nbCreators) {
    nbCreators = 1;  // We only have one plugin creator
    return gPluginCreators;
}
