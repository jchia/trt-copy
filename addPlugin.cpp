#include <cstring>
#include <iostream>
#include "addPlugin.hpp"

using namespace std;

namespace nvi = nvinfer1;

namespace {
char const* kPluginNamespace = "";
char const* kPluginName = "CustomAdd";
char const* kPluginVersion = "1";
}

namespace rgr {

#define PLUGIN_ASSERT(x) do { if (!(x)) return 1; } while (false)

cudaError_t computeAdd(cudaStream_t stream, float const* input, float* output, uint32_t size, float const *scalar);

AddPlugin::AddPlugin() { }

nvi::IPluginCapability* AddPlugin::getCapabilityInterface(nvi::PluginCapabilityType type) noexcept {
    switch (type) {
        case nvi::PluginCapabilityType::kBUILD:
            return static_cast<nvi::IPluginV3OneBuild*>(this);
        case nvi::PluginCapabilityType::kRUNTIME:
            return static_cast<nvi::IPluginV3OneRuntime*>(this);
        case nvi::PluginCapabilityType::kCORE:
            return static_cast<nvi::IPluginV3OneCore*>(this);
        default:
            return nullptr;
    }
}

nvi::IPluginV3* AddPlugin::clone() noexcept {
    return static_cast<nvi::IPluginV3*>(new AddPlugin());
}

nvi::AsciiChar const* AddPlugin::getPluginNamespace() const noexcept {
    return kPluginNamespace;
}

nvi::AsciiChar const* AddPlugin::getPluginName() const noexcept {
    return kPluginName;
}

nvi::AsciiChar const* AddPlugin::getPluginVersion() const noexcept {
    return kPluginVersion;
}

int32_t AddPlugin::configurePlugin(
    nvi::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvi::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(nbInputs == 2 && nbOutputs == 1);
    PLUGIN_ASSERT(in[0].desc.type == nvi::DataType::kFLOAT && in[1].desc.type == nvi::DataType::kFLOAT);
    auto const& dims0 = in[0].desc.dims;
    auto const& dims1 = in[1].desc.dims;
    PLUGIN_ASSERT(dims0.nbDims <= 1 || dims0.nbDims <= 1);
    return 0;
}

int32_t AddPlugin::getOutputDataTypes(
    nvi::DataType* outputTypes, int32_t nbOutputs,
    const nvi::DataType* inputTypes, int32_t nbInputs) const noexcept {
    PLUGIN_ASSERT(nbInputs == 2 &&
                  inputTypes[0] == nvi::DataType::kFLOAT && inputTypes[1] == nvi::DataType::kFLOAT &&
                  nbOutputs == 1);
    outputTypes[0] = nvi::DataType::kFLOAT;
    return 0;
}

int32_t AddPlugin::getOutputShapes(
    nvi::DimsExprs const* inputs, int32_t nbInputs, nvi::DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, nvi::DimsExprs* outputs, int32_t nbOutputs,
    nvi::IExprBuilder& exprBuilder) noexcept {
    PLUGIN_ASSERT(nbInputs == 2 && nbShapeInputs == 0 && nbOutputs == 1);
    PLUGIN_ASSERT(inputs[0].nbDims <= 1 || inputs[1].nbDims <= 1);
    bool const lhsIsScalar = inputs[0].nbDims < inputs[1].nbDims;
    int32_t const nbDims = outputs[0].nbDims = inputs[lhsIsScalar].nbDims;
    for (int32_t i = 0; i < nbDims; ++i)
        outputs[0].d[i] = inputs[lhsIsScalar].d[i];
    return 0;
}

bool AddPlugin::supportsFormatCombination(
    int32_t pos, nvi::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept {
    if (nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos >= 3)
        return false;
    auto const& desc = inOut[pos].desc;
    return desc.type == nvi::DataType::kFLOAT && desc.format == nvi::TensorFormat::kLINEAR;
}

int32_t AddPlugin::onShapeChange(
    nvi::PluginTensorDesc const* in, int32_t nbInputs, nvi::PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(
        nbInputs == 2 && nbOutputs == 1 &&
        in[0].type == nvi::DataType::kFLOAT && in[1].type == nvi::DataType::kFLOAT &&
        out->type == nvi::DataType::kFLOAT);

    PLUGIN_ASSERT(in[0].dims.nbDims <= 1 && in[1].dims.nbDims <= 1);
    bool const lhsIsScalar = in[0].dims.nbDims < in[1].dims.nbDims;
    int32_t const nbDims = in[lhsIsScalar].dims.nbDims;
    PLUGIN_ASSERT(out[0].dims.nbDims == nbDims);
    for (int32_t i = 0; i < nbDims; ++i)
        PLUGIN_ASSERT(out[0].dims.d[i] == in[lhsIsScalar].dims.d[i]);
    mLhsIsScalar = lhsIsScalar;
    return 0;
}

int32_t AddPlugin::enqueue(
    nvi::PluginTensorDesc const* inputDesc, nvi::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept {
    auto const& inputShape = inputDesc[mLhsIsScalar].dims;
    auto const& outputShape = outputDesc->dims;
    PLUGIN_ASSERT(inputShape.nbDims == outputShape.nbDims);
    uint32_t numEl = 1;
    for (int32_t i = 0; i < inputShape.nbDims; ++i) {
        if (inputShape.d[i] != outputShape.d[i])
            return 1;
        if (inputShape.d[i] < 0)
            return 2;
        numEl *= static_cast<uint32_t>(inputShape.d[i]);
    }

    return computeAdd(
        stream, reinterpret_cast<float const*>(inputs[mLhsIsScalar]), reinterpret_cast<float*>(*outputs),
        numEl, reinterpret_cast<float const*>(inputs[!mLhsIsScalar]));
}

nvi::IPluginV3* AddPlugin::attachToContext(nvi::IPluginResourceContext* context) noexcept {
    return clone();
}

nvi::PluginFieldCollection const* AddPlugin::getFieldsToSerialize() noexcept {
    return nullptr;
}

AddPluginCreator::AddPluginCreator() { }

char const* AddPluginCreator::getPluginNamespace() const noexcept {
    return kPluginNamespace;
}

char const* AddPluginCreator::getPluginName() const noexcept {
    return kPluginName;
}

char const* AddPluginCreator::getPluginVersion() const noexcept {
    return kPluginVersion;
}

nvi::PluginFieldCollection const* AddPluginCreator::getFieldNames() noexcept {
    static const nvi::PluginFieldCollection kFc{0, nullptr};
    return &kFc;
}

nvi::IPluginV3* AddPluginCreator::createPlugin(
    nvi::AsciiChar const* name, nvi::PluginFieldCollection const* fc, nvi::TensorRTPhase phase) noexcept {
    return new AddPlugin();
}

}
