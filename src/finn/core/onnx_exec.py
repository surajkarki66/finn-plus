# Copyright (c) 2022, Xilinx, Inc.
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
# * Neither the name of FINN nor the names of its
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

"""ONNX model execution utilities for FINN.

This module provides enhanced ONNX model execution capabilities that extend
the base QONNX execution functionality with FINN-specific features, including
RTL simulation support and model debugging utilities.

Key Functions:
- execute_onnx: Enhanced ONNX execution with RTL simulation support
- execute_onnx_and_make_model: Create debug models with intermediate activations
- compare_execution: Compare outputs between two models
"""

import copy
import numpy as np
from collections.abc import Callable
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx as execute_onnx_base


def execute_onnx(
    model: "ModelWrapper",
    input_dict: dict[str, np.ndarray],
    return_full_exec_context: bool = False,
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
) -> dict[str, np.ndarray]:
    """Execute given ONNX ModelWrapper with given named inputs.
    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.
    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict.
    When start_node and end_node are set to None, the whole graph is executed.
    If they are set to particular ONNX nodes, only the subgraph between (and
    including) those nodes is executed.
    """
    return execute_onnx_base(model, input_dict, return_full_exec_context, start_node, end_node)


def execute_onnx_and_make_model(
    model: "ModelWrapper", input_dict: dict[str, np.ndarray]
) -> ModelWrapper:
    """Execute given ONNX ModelWrapper with given named inputs and return a new
    ModelWrapper where an initializer is provided for each tensor as taken from
    the execution. This new model is useful for debugging, since it contains
    all the intermediate activation values."""
    # retrieve the full execution context
    execution_context = execute_onnx(model, input_dict, True)
    new_model = copy.deepcopy(model)
    # create value_info entries and initializers for everything
    for i in execution_context.keys():
        if i != "" and execution_context[i] is not None:
            new_model.set_initializer(i, execution_context[i])
    for vi in new_model.graph.value_info:
        new_model.graph.output.append(vi)
    return new_model


def compare_execution(
    model_a: "ModelWrapper",
    model_b: "ModelWrapper",
    input_dict: dict[str, np.ndarray],
    compare_fxn: Callable[
        [list | np.ndarray, list | np.ndarray], bool | np.bool_
    ] = lambda x, y: np.isclose(x, y, atol=1e-3).all(),
) -> bool | np.bool_:
    """Execute two ONNX models and compare their outputs using given function.

    compare_fxn should take in two tensors and return a Boolean"""
    # compare values from first output tensors produced
    res_a = next(iter(execute_onnx(model_a, input_dict).items()))[1]
    res_b = next(iter(execute_onnx(model_b, input_dict).items()))[1]
    return compare_fxn(res_a, res_b)
