# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from qonnx.util.basic import get_by_name

from finn.util.exception import FINNInternalError


def is_fpgadataflow_node(node):
    """Returns True if given node is fpgadataflow node. Otherwise False."""
    is_node = False
    if node is not None:
        if is_custom_op(node.domain):
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node


def is_hls_node(node):
    """Returns True if given node is hls node. Otherwise False."""
    is_node = False
    if node is not None:
        if node.domain == "finn.custom_op.fpgadataflow.hls":
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node


def is_rtl_node(node):
    """Returns True if given node is rtl node. Otherwise False."""
    is_node = False
    if node is not None:
        if node.domain == "finn.custom_op.fpgadataflow.rtl":
            n_backend = get_by_name(node.attribute, "backend")
            if n_backend is not None:
                backend_value = n_backend.s.decode("UTF-8")
                if backend_value == "fpgadataflow":
                    is_node = True

    return is_node


def get_submodel(node: NodeProto) -> tuple[ModelWrapper, Path]:
    """Try to retrieve the submodel (and its path) of a StreamingDataflowPartition
    node. If the node is not an SDP or the `model` metadata prop does not exist,
    or the path does not point to a file, an error is raised.
    """
    if node.op_type != "StreamingDataflowPartition":
        raise FINNInternalError(f"Cannot get model of non-SDP node: {node.name}")
    p = getCustomOp(node).get_nodeattr("model")
    if p is None:
        raise FINNInternalError(
            f"SDP node {node.name} has no 'model' metadata prop. " f"Cannot get model."
        )
    p = Path(str(p))
    if not p.exists():
        raise FINNInternalError(
            f"Cannot open model of SDP node {node.name}: " f"No file found at path: {p}"
        )
    return ModelWrapper(str(p)), p
