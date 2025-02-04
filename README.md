Example to show TensorRT failure to eliminate concatenation layer when the concatenation input is the output of a custom layer.

# How to run:
```
$ make build-simple infer-simple
$ ./make-onnx-model.py --size 16 --output-filename sac16.onnx
$ ./make-onnx-model.py --custom --size 16 --output-filename sac16c.onnx
$ ./build-simple sac16.onnx sac16.plan
$ ./build-simple sac16c.onnx sac16c.plan
$ /opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all ./infer-simple sac16c.plan
$ /opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all ./infer-simple sac16.plan
```

## Alternative using trtexec:
```
$ make plugin.so
$ trtexec --verbose --onnx=sac16.onnx --saveEngine=sac16.plan
$ trtexec --verbose --onnx=sac16c.onnx --saveEngine=sac16c.plan --dynamicPlugins=./plugin.so
$ /opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all /usr/src/tensorrt/bin/trtexec --loadEngine=sac16.plan
$ /opt/nvidia/nsight-compute/2024.3.2/ncu --target-processes all /usr/src/tensorrt/bin/trtexec --loadEngine=sac16c.plan --dynamicPlugins=./plugin.so
```

(make-onnx-model.py can be skipped as the resulting .onnx files are already in the repo.)

# Explanation
The network is illustrated in network.png. sac16.onnx uses the regular Add operator. sac16c.onnx uses a CustomAdd operator, implemented as a plugin in addPlugin.cpp.

In the engine-building logs, for sac16.onnx, the following log lines suggest that concatenation was eliminated (see the lines after "Eliminating concatenation node_of_output"):
```
Original: 6 layers
After dead-layer removal: 6 layers
Graph construction completed in 9.6362e-05 seconds.
After adding DebugOutput nodes: 6 layers
After Myelin optimization: 6 layers
Applying ScaleNodes fusions.
After scale fusion: 6 layers
Running: ConstantSplit on onef
Running: EltWiseToPointwiseConversion on node_of_part0_plus1
Swap the layer type of node_of_part0_plus1 from ELEMENTWISE to POINTWISE
Running: EltWiseToPointwiseConversion on node_of_part1_plus1
Swap the layer type of node_of_part1_plus1 from ELEMENTWISE to POINTWISE
After dupe layer removal: 7 layers
After final dead-layer removal: 7 layers
After tensor merging: 7 layers
Running: PointWiseFusion on onef
PointWiseFusion: Fusing onef with PWN(node_of_part0_plus1)
Running: PointWiseFusion on onef_clone_1
PointWiseFusion: Fusing onef_clone_1 with PWN(node_of_part1_plus1)
After vertical fusions: 5 layers
After dupe layer removal: 5 layers
After final dead-layer removal: 5 layers
After tensor merging: 5 layers
Eliminating slice node_of_part0 by retargeting part0 from part0 to input
Eliminating slice node_of_part1 by retargeting part1 from part1 to input
After slice removal: 3 layers
Eliminating concatenation node_of_output
Retargeting part0_plus1 to output
Retargeting part1_plus1 to output
After concat removal: 2 layers
Trying to split Reshape and strided tensor
Graph optimization time: 0.000435699 seconds.
```

For sac16c.onnx, the following log lines suggest that concatenation was not eliminated:
```
Original: 6 layers
After dead-layer removal: 6 layers
Graph construction completed in 0.00010097 seconds.
After adding DebugOutput nodes: 6 layers
After Myelin optimization: 6 layers
Applying ScaleNodes fusions.
After scale fusion: 6 layers
Running: ConstantSplit on onef
After dupe layer removal: 7 layers
After final dead-layer removal: 7 layers
After tensor merging: 7 layers
After vertical fusions: 7 layers
After dupe layer removal: 7 layers
After final dead-layer removal: 7 layers
After tensor merging: 7 layers
Replacing slice node_of_part0 with copy from input to part0
Replacing slice node_of_part1 with copy from input to part1
After slice removal: 7 layers
Eliminating concatenation node_of_output
Generating copy for part0_plus1 to output because input does not support striding.
Generating copy for part1_plus1 to output because input does not support striding.
After concat removal: 8 layers
Trying to split Reshape and strided tensor
Graph optimization time: 0.000178867 seconds.
```

The ncu output shows the layers run and indicates the that each iteration involves 2 kernel runs for sac16 but 4 kernel runs for sac16c, including a 'copyVectorizedKernel'.

It is unclear what is means by "input does not support striding" or whether it is possible for a custom layer to support striding.
