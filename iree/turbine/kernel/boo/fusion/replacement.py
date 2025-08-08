from collections.abc import Callable

import torch
from torch import fx
from torch.fx.node import Argument, Target


from iree.turbine.kernel.boo import ops as boo_ops

# 1 to 1 op replacements. The replacement function should return the new target
# and arguments, or 'None' if it doesn't apply.
ReplacementSchema = dict[
    Target,
    Callable[
        [tuple[Argument, ...], dict[str, object]],  # (node.args, node.meta)
        tuple[Callable, tuple[Argument, ...]] | None,  # (new_target, new_args) | None
    ],
]


def apply_replacements(graph: fx.Graph, replacements: ReplacementSchema):
    for node in graph.nodes:
        if node.op == "call_function":
            replacer_fn = replacements.get(str(node.target), lambda *_: None)
            replacement = replacer_fn(node.args, node.meta)
            if replacement is None:
                continue
            target, target_args = replacement
            with graph.inserting_after(node):
                call_boo = graph.call_function(target, target_args)
            node.replace_all_uses_with(call_boo, propagate_meta=True)
            graph.erase_node(node)


def replace_conv_relu(args: tuple[Argument, ...], meta: dict[str, object]):
    """Replace `torch.ops.conv_relu._fwd` with custom BOO implementation."""
    (
        x,
        weight,
        bias,
        padding,
        stride,
        dilation,
        groups,
    ) = args
    example_out = meta["val"]
    assert isinstance(example_out, torch.Tensor)
    output_is_channels_last = example_out.is_contiguous(
        memory_format=torch.channels_last
    )
    num_spatial_dims = len(example_out.shape) - 2

    def replacement_fn(
        input, weight, bias, padding, stride, dilation, groups, output_is_channels_last
    ):
        y = boo_ops.convolution_replacement(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            output_is_channels_last,
        )
        return torch.ops.aten.relu(y)

    return replacement_fn, (
        x,
        weight,
        bias,
        boo_ops.make_tuple(padding, num_spatial_dims),
        boo_ops.make_tuple(stride, num_spatial_dims),
        boo_ops.make_tuple(dilation, num_spatial_dims),
        groups,
        output_is_channels_last,
    )


def replace_aten_convolution(args: tuple[Argument, ...], meta: dict[str, object]):
    "Replace 'torch.ops.aten.convolution' with custom BOO implementation."
    (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        _output_padding,
        groups,
    ) = args

    # BOO convolution doesn't support transpose. 'output_padding' is ignored
    # in non-transpose cases.
    if transposed is not False:
        return None

    example_out = meta["val"]
    assert isinstance(example_out, torch.Tensor)
    output_is_channels_last = example_out.is_contiguous(
        memory_format=torch.channels_last
    )
    return boo_ops.convolution_replacement, (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        output_is_channels_last,
    )
