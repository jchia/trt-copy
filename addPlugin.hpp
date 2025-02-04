#pragma once
#include <cuda.h>
#if CUDA_VERSION < 12010
#error "CUDA version needs to be at least 12.1"
#endif

#include <NvInferPlugin.h>

namespace rgr {

class AddPlugin
    : public nvinfer1::IPluginV3, public nvinfer1::IPluginV3OneCore
    , public nvinfer1::IPluginV3OneBuild, public nvinfer1::IPluginV3OneRuntime {
public:
    AddPlugin();

    // IPluginV3.
    IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    IPluginV3* clone() noexcept override;


    // IPluginV3OneCore
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;
    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;

    // IPluginV3OneBuild
    int32_t configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(
        nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getNbOutputs() const noexcept override { return 1; }

    // IPluginV3OneRuntime
    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    bool mLhsIsScalar{};
};

class AddPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    AddPluginCreator();

    char const* getPluginNamespace() const noexcept override;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV3* createPlugin(
        nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;

};

} // namespace rgr
