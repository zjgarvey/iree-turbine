# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from operator import getitem
from typing import Callable, Dict, Sequence


import torch
from torch.fx.node import Target, Node

from .replacement import ReplacementSchema, replace_aten_convolution, replace_conv_relu


@dataclass
class OpFusionSpec:
    recursive: bool = True
    """Whether to recursively fuse in producers and consumers."""
    make_single_dispatch: bool = False
    """Whether to force compile a fused op into a single dispatch."""
    match_filters: Sequence[Callable[[Node], bool]] = ()
    """A sequence of filter functions to restrict what gets included in the subgraph.
    If any of these return false, the node is not included in the fused subgraph."""
    producers: Sequence[Target] = ()
    """Producer nodes we want to fuse with."""
    consumers: Sequence[Target] = ()
    """Consumer nodes we want to fuse with."""

    def check_filters(self, node: Node):
        return all([f(node) for f in self.match_filters])

    def is_fusable_producer(self, node: Node):
        """Checks if `node` is a fusable producer.

        This currently doesn't allow `getitem` unless explicitly indicated by `self.producers`.
        """
        if node.op != "call_function" or not self.check_filters(node):
            return False
        # TODO: add support for capturing all `getitem` ops for multi-output producers.
        return node.target in self.producers or str(node.target) in self.producers

    def is_fusable_consumer(self, node: Node):
        """Checks if `node` is a fusable consumer.

        Any multi-output nodes also get their consuming `getitem` calls fused into the graph.
        """
        if node.op != "call_function" or not self.check_filters(node):
            return False
        return (
            node.target in self.consumers
            or str(node.target) in self.consumers
            or node.target == getitem
        )


FusionSchema = Dict[str, OpFusionSpec]


def _all_unit_spatial_dims_filter(node: Node) -> bool:
    if str(node.target) not in ["aten.convolution.default", "conv_relu._fwd.default"]:
        return True
    x, w, *_ = node.all_input_nodes
    input_meta = x.meta.get("val", None)
    weight_meta = w.meta.get("val", None)
    if input_meta is None or weight_meta is None:
        return False
    return all([dim == 1 for dim in input_meta.shape[-2:] + weight_meta.shape[-2:]])


def _conv_transpose_filter(node: Node) -> bool:
    if node.target != torch.ops.aten.convolution.default:
        return True
    transposed = node.args[-3]
    return transposed == False


# TODO: extend this
DEFAULT_SUPPORTED_BOO_FUSIONS: FusionSchema = {
    "aten.convolution.default": OpFusionSpec(
        recursive=True,
        make_single_dispatch=True,
        match_filters=(_conv_transpose_filter, _all_unit_spatial_dims_filter),
        consumers=(
            "aten.relu.default",
            "aten.sigmoid.default",
        ),
    ),
    "conv_relu._fwd.default": OpFusionSpec(
        make_single_dispatch=True,
        match_filters=(_all_unit_spatial_dims_filter,),
    ),
    "aten.native_layer_norm.default": OpFusionSpec(
        make_single_dispatch=True,
    ),
}

DEFAULT_POST_FUSION_REPLACEMENTS: ReplacementSchema = {
    "aten.convolution.default": replace_aten_convolution,
    "conv_relu._fwd.default": replace_conv_relu,
}
