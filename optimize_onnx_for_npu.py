#!/usr/bin/env python3
"""
Optimize ONNX model for NPU deployment by replacing unsupported operators.

This script replaces:
- GELU -> Sigmoid approximation (x * sigmoid(1.702 * x))
- LayerNormalization -> Decomposed ops (ReduceMean, Sub, Div, Mul, Add)

Usage:
    python optimize_onnx_for_npu.py \
        --input exports/efficientsam3_repvit_m1_1.onnx \
        --output exports/efficientsam3_repvit_m1_1_npu.onnx

    # With specific replacements
    python optimize_onnx_for_npu.py \
        --input model.onnx \
        --output model_npu.onnx \
        --replace-gelu \
        --replace-layernorm
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def replace_gelu_with_sigmoid(graph):
    """
    Replace GELU activation with sigmoid approximation.

    GELU(x) ≈ x * sigmoid(1.702 * x)

    This is a close approximation that uses only Mul and Sigmoid ops.
    """
    nodes_to_remove = []
    nodes_to_add = []

    for node in graph.node:
        if node.op_type == "Gelu":
            input_name = node.input[0]
            output_name = node.output[0]
            node_name = node.name or f"gelu_{id(node)}"

            # Create constant for scale factor 1.702
            scale_name = f"{node_name}_scale"
            scale_tensor = numpy_helper.from_array(
                np.array([1.702], dtype=np.float32),
                name=scale_name
            )
            graph.initializer.append(scale_tensor)

            # Intermediate names
            scaled_name = f"{node_name}_scaled"
            sigmoid_name = f"{node_name}_sigmoid"

            # Create nodes: x * sigmoid(1.702 * x)
            # 1. scaled = x * 1.702
            mul_scale = helper.make_node(
                "Mul",
                inputs=[input_name, scale_name],
                outputs=[scaled_name],
                name=f"{node_name}_mul_scale"
            )

            # 2. sigmoid_out = sigmoid(scaled)
            sigmoid_node = helper.make_node(
                "Sigmoid",
                inputs=[scaled_name],
                outputs=[sigmoid_name],
                name=f"{node_name}_sigmoid"
            )

            # 3. output = x * sigmoid_out
            mul_out = helper.make_node(
                "Mul",
                inputs=[input_name, sigmoid_name],
                outputs=[output_name],
                name=f"{node_name}_mul_out"
            )

            nodes_to_add.extend([mul_scale, sigmoid_node, mul_out])
            nodes_to_remove.append(node)

    # Remove old nodes and add new ones
    for node in nodes_to_remove:
        graph.node.remove(node)
    graph.node.extend(nodes_to_add)

    return len(nodes_to_remove)


def replace_layernorm_decomposed(graph):
    """
    Replace LayerNormalization with decomposed basic ops.

    LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

    Decomposed into: ReduceMean, Sub, Pow, Add, Sqrt, Div, Mul, Add
    """
    nodes_to_remove = []
    nodes_to_add = []

    for node in graph.node:
        if node.op_type == "LayerNormalization":
            input_name = node.input[0]
            scale_name = node.input[1] if len(node.input) > 1 else None
            bias_name = node.input[2] if len(node.input) > 2 else None
            output_name = node.output[0]
            node_name = node.name or f"ln_{id(node)}"

            # Get epsilon attribute
            eps = 1e-5
            axis = -1
            for attr in node.attribute:
                if attr.name == "epsilon":
                    eps = attr.f
                if attr.name == "axis":
                    axis = attr.i

            # Create constants
            eps_name = f"{node_name}_eps"
            eps_tensor = numpy_helper.from_array(
                np.array([eps], dtype=np.float32),
                name=eps_name
            )
            graph.initializer.append(eps_tensor)

            pow_exp_name = f"{node_name}_pow_exp"
            pow_tensor = numpy_helper.from_array(
                np.array([2.0], dtype=np.float32),
                name=pow_exp_name
            )
            graph.initializer.append(pow_tensor)

            # Intermediate names
            mean_name = f"{node_name}_mean"
            sub_name = f"{node_name}_sub"
            pow_name = f"{node_name}_pow"
            var_name = f"{node_name}_var"
            var_eps_name = f"{node_name}_var_eps"
            std_name = f"{node_name}_std"
            norm_name = f"{node_name}_norm"
            scaled_name = f"{node_name}_scaled"

            new_nodes = []

            # 1. mean = ReduceMean(x, axis)
            mean_node = helper.make_node(
                "ReduceMean",
                inputs=[input_name],
                outputs=[mean_name],
                name=f"{node_name}_reduce_mean",
                axes=[axis],
                keepdims=1
            )
            new_nodes.append(mean_node)

            # 2. sub = x - mean
            sub_node = helper.make_node(
                "Sub",
                inputs=[input_name, mean_name],
                outputs=[sub_name],
                name=f"{node_name}_sub"
            )
            new_nodes.append(sub_node)

            # 3. pow = sub ^ 2
            pow_node = helper.make_node(
                "Pow",
                inputs=[sub_name, pow_exp_name],
                outputs=[pow_name],
                name=f"{node_name}_pow"
            )
            new_nodes.append(pow_node)

            # 4. var = ReduceMean(pow, axis)
            var_node = helper.make_node(
                "ReduceMean",
                inputs=[pow_name],
                outputs=[var_name],
                name=f"{node_name}_var",
                axes=[axis],
                keepdims=1
            )
            new_nodes.append(var_node)

            # 5. var_eps = var + eps
            add_eps_node = helper.make_node(
                "Add",
                inputs=[var_name, eps_name],
                outputs=[var_eps_name],
                name=f"{node_name}_add_eps"
            )
            new_nodes.append(add_eps_node)

            # 6. std = sqrt(var_eps)
            sqrt_node = helper.make_node(
                "Sqrt",
                inputs=[var_eps_name],
                outputs=[std_name],
                name=f"{node_name}_sqrt"
            )
            new_nodes.append(sqrt_node)

            # 7. norm = sub / std
            div_node = helper.make_node(
                "Div",
                inputs=[sub_name, std_name],
                outputs=[norm_name],
                name=f"{node_name}_div"
            )
            new_nodes.append(div_node)

            # 8. scaled = norm * gamma (if gamma exists)
            if scale_name:
                mul_scale_node = helper.make_node(
                    "Mul",
                    inputs=[norm_name, scale_name],
                    outputs=[scaled_name],
                    name=f"{node_name}_mul_scale"
                )
                new_nodes.append(mul_scale_node)
                final_input = scaled_name
            else:
                final_input = norm_name

            # 9. output = scaled + beta (if beta exists)
            if bias_name:
                add_bias_node = helper.make_node(
                    "Add",
                    inputs=[final_input, bias_name],
                    outputs=[output_name],
                    name=f"{node_name}_add_bias"
                )
                new_nodes.append(add_bias_node)
            else:
                # Rename final output
                new_nodes[-1].output[0] = output_name

            nodes_to_add.extend(new_nodes)
            nodes_to_remove.append(node)

    # Remove old nodes and add new ones
    for node in nodes_to_remove:
        graph.node.remove(node)
    graph.node.extend(nodes_to_add)

    return len(nodes_to_remove)


def analyze_model(model):
    """Analyze model and print operator statistics."""
    ops = {}
    for node in model.graph.node:
        op = node.op_type
        ops[op] = ops.get(op, 0) + 1

    print("\nOperator statistics:")
    for op, count in sorted(ops.items()):
        marker = ""
        if op in ["Gelu", "LayerNormalization", "Attention"]:
            marker = " ⚠️  (NPU unfriendly)"
        print(f"  {op}: {count}{marker}")

    return ops


def optimize_model(
    input_path: str,
    output_path: str,
    replace_gelu: bool = True,
    replace_layernorm: bool = True,
):
    """Optimize ONNX model for NPU deployment."""
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)

    # Analyze before optimization
    print("\n=== Before optimization ===")
    ops_before = analyze_model(model)

    # Apply optimizations
    gelu_count = 0
    ln_count = 0

    if replace_gelu and "Gelu" in ops_before:
        print("\nReplacing GELU with sigmoid approximation...")
        gelu_count = replace_gelu_with_sigmoid(model.graph)
        print(f"  Replaced {gelu_count} GELU nodes")

    if replace_layernorm and "LayerNormalization" in ops_before:
        print("\nReplacing LayerNormalization with decomposed ops...")
        ln_count = replace_layernorm_decomposed(model.graph)
        print(f"  Replaced {ln_count} LayerNormalization nodes")

    # Analyze after optimization
    print("\n=== After optimization ===")
    analyze_model(model)

    # Validate model
    print("\nValidating model...")
    try:
        onnx.checker.check_model(model)
        print("  Model is valid")
    except Exception as e:
        print(f"  Warning: Model validation failed: {e}")

    # Save model
    print(f"\nSaving optimized model: {output_path}")
    onnx.save(model, output_path)

    # Print file sizes
    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Input size:  {input_size:.2f} MB")
    print(f"  Output size: {output_size:.2f} MB")

    return {
        "gelu_replaced": gelu_count,
        "layernorm_replaced": ln_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimize ONNX model for NPU deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize with all replacements (default)
  python optimize_onnx_for_npu.py \\
      --input model.onnx \\
      --output model_npu.onnx

  # Only replace GELU
  python optimize_onnx_for_npu.py \\
      --input model.onnx \\
      --output model_npu.onnx \\
      --replace-gelu \\
      --no-replace-layernorm

Replacements:
  GELU -> x * sigmoid(1.702 * x)
  LayerNorm -> ReduceMean + Sub + Pow + Sqrt + Div + Mul + Add
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input ONNX model path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--replace-gelu",
        action="store_true",
        default=True,
        help="Replace GELU with sigmoid approximation (default: True)",
    )
    parser.add_argument(
        "--no-replace-gelu",
        action="store_false",
        dest="replace_gelu",
        help="Don't replace GELU",
    )
    parser.add_argument(
        "--replace-layernorm",
        action="store_true",
        default=True,
        help="Replace LayerNormalization with decomposed ops (default: True)",
    )
    parser.add_argument(
        "--no-replace-layernorm",
        action="store_false",
        dest="replace_layernorm",
        help="Don't replace LayerNormalization",
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run optimization
    optimize_model(
        input_path=args.input,
        output_path=args.output,
        replace_gelu=args.replace_gelu,
        replace_layernorm=args.replace_layernorm,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
