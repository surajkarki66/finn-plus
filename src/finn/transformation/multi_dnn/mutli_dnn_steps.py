"""Build-flow steps for multi-DNN model construction and collapsing."""
import json
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.attention_heads import InferSplitIntoSplitMultiHeads
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.multi_dnn.multi_dnn_pr import ApplyPartialReconfiguration
from finn.transformation.multi_dnn.multi_dnn_selectable import ExtractSelectableWeights
from finn.transformation.multi_dnn.multi_dnn_wrapper_transformations import (
    CollapseModels,
    CombineInputsChannelwise,
    CombineOutputsChannelwise,
    MultiDNNWrapperExposeIO,
)
from finn.transformation.multi_dnn.nodecontainer_transformations import NameNodeContainerNodes


def _resolve_multi_dnn_mode(cfg: DataflowBuildConfig):
    """Read the generation mode and kwargs from the multi-DNN config JSON."""
    with open(cfg.multi_dnn_config_path, "r") as fp_json:
        multi_dnn_config = json.load(fp_json)
    gen = multi_dnn_config.get("Generation")
    return gen["mode"], gen.get("kwargs", None)


def step_apply_multi_dnn(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Apply the appropriate multi-DNN transformation (Parallel, SelectableWeights, or PR)."""
    mode, kwargs = _resolve_multi_dnn_mode(cfg)
    if mode == "Parallel":
        if kwargs is not None:
            combine_inputs_channelwise = kwargs.get("combine_inputs_channelwise", None)
            combine_outputs_channelwise = kwargs.get("combine_outputs_channelwise", None)
        model = model.transform(MultiDNNWrapperExposeIO())
        model = model.transform(CombineInputsChannelwise()) if combine_inputs_channelwise else model
        model = (
            model.transform(CombineOutputsChannelwise()) if combine_outputs_channelwise else model
        )
    elif mode == "SelectableWeights":
        model = model.transform(ExtractSelectableWeights(**kwargs))
        model = model.transform(MultiDNNWrapperExposeIO())
    elif mode == "PartialReconfiguration":
        model = model.transform(ApplyPartialReconfiguration(**kwargs))
        model = model.transform(MultiDNNWrapperExposeIO())
    else:
        raise Exception("This Mode is not implemented")

    return model


def step_collapse_multi_dnn(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Collapse all DNNContainer subgraphs and specialize the resulting concat/split nodes."""
    model = model.transform(CollapseModels())
    model = model.transform(InferSplitIntoSplitMultiHeads())
    model = model.transform(to_hw.InferConcatLayer())
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))  # For Concat and Split
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(NameNodeContainerNodes())
    return model


def step_maximize_concat_split_simd(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Maximize SIMD on StreamingConcat_hls nodes after collapse.

    The Parallel multi-DNN flow inserts SplitMultiHeads_hls (already fully
    parallel, no SIMD attribute) and StreamingConcat_hls (initialized with
    SIMD=1). This step raises SIMD on StreamingConcat_hls nodes to the largest
    common divisor of ChannelsPerStream so their cycle count is as close as
    possible to the surrounding operators.
    """
    from qonnx.custom_op.registry import getCustomOp

    from finn.transformation.fpgadataflow.set_folding import common_divisors

    for node in model.graph.node:
        if node.op_type == "StreamingConcat_hls":
            node_inst = getCustomOp(node)
            channels_per_stream = node_inst.get_nodeattr("ChannelsPerStream")
            max_simd = int(max(common_divisors(channels_per_stream)))
            node_inst.set_nodeattr("SIMD", max_simd)
    return model
