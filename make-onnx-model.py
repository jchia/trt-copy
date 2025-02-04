#!/bin/env python

import argparse
import onnx
from onnx import helper, TensorProto

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-filename', required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--custom', action='store_true', help='Use CustomAdd operator instead of Add')
    args = parser.parse_args()

    size_val = args.size

    # Define input & output (both 1D float32 of length 'size_val')
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [size_val])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [size_val])

    # Define int32 constants
    zero = helper.make_tensor('zero', TensorProto.INT32, [1], [0])
    boundary = helper.make_tensor('boundary', TensorProto.INT32, [1], [size_val // 2])
    size_ = helper.make_tensor('size', TensorProto.INT32, [1], [size_val])
    one = helper.make_tensor('one', TensorProto.INT32, [1], [1])

    # Define float constant for addition
    onef = helper.make_tensor('onef', TensorProto.FLOAT, [1], [1.0])

    # Slice nodes
    slice_0 = helper.make_node(
        'Slice',
        inputs=['input', 'zero', 'boundary', 'zero', 'one'],
        outputs=['part0']
    )
    slice_1 = helper.make_node(
        'Slice',
        inputs=['input', 'boundary', 'size', 'zero', 'one'],
        outputs=['part1']
    )

    # Add +1.0 to each slice
    add_op = 'CustomAdd' if args.custom else 'Add'
    add_0 = helper.make_node(
        add_op,
        inputs=['part0', 'onef'],
        outputs=['part0_plus1']
    )
    add_1 = helper.make_node(
        add_op,
        inputs=['part1', 'onef'],
        outputs=['part1_plus1']
    )

    # Concat
    concat = helper.make_node(
        'Concat',
        inputs=['part0_plus1', 'part1_plus1'],
        outputs=['output'],
        axis=0
    )

    # Build the graph
    graph_def = helper.make_graph(
        [slice_0, slice_1, add_0, add_1, concat],
        'SplitAddConcatModel',
        [input_info],
        [output_info],
        initializer=[zero, boundary, size_, one, onef]
    )

    # Create and save the model
    model_def = helper.make_model(graph_def, opset_imports=[helper.make_opsetid('', 17)])
    onnx.save(model_def, args.output_filename)

if __name__ == '__main__':
    main()
