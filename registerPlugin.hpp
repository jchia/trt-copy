#pragma once

#include <NvInfer.h>

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder* finder);
extern "C" nvinfer1::IPluginCreatorV3One* const* getCreators(int32_t& nbCreators);
