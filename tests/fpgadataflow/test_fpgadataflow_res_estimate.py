# Copyright (c) 2020, Xilinx
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

import pytest

from functools import partial
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.res_estimation import (
    res_estimation,
    res_estimation_complete,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

test_fpga_part = "xczu3eg-sbva484-1-e"


def check_two_dict_for_equality(dict1, dict2):
    for key in dict1:
        assert key in dict2, "Key: {} is not in both dictionaries".format(key)
        assert (
            dict1[key] == dict2[key]
        ), """Values for key {} are not the same
        in both dictionaries""".format(
            key
        )

    return True


def _make_single_hls_node_model(node, inputs, outputs, tensor_dtypes):
    graph = helper.make_graph(
        nodes=[node],
        name=node.name + "_graph",
        inputs=inputs,
        outputs=outputs,
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name=node.name + "-model"))
    for tensor_name, tensor_dtype in tensor_dtypes.items():
        model.set_tensor_datatype(tensor_name, tensor_dtype)
    return model


@pytest.mark.fpgadataflow
def test_res_estimate():
    mw = mh = 4
    simd = 1
    pe = 1
    idt = DataType["INT2"]
    wdt = DataType["INT2"]
    odt = DataType["INT2"]
    actval = odt.min()

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    node_inp_list = ["inp", "weights", "thresh"]

    FCLayer_node = helper.make_node(
        "MVAU",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=actval,
        binaryXnorMode=0,
        noActivation=0,
    )
    graph = helper.make_graph(
        nodes=[FCLayer_node], name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)

    model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    prod_resource_estimation = model.analysis(partial(res_estimation, fpgapart=test_fpga_part))
    expect_resource_estimation = {
        "MVAU_hls_0": {
            "BRAM_18K": 0,
            "BRAM_efficiency": 1,
            "LUT": 317,
            "DSP": 0,
            "URAM_efficiency": 1,
            "URAM": 0,
        }
    }

    assert check_two_dict_for_equality(
        prod_resource_estimation, expect_resource_estimation
    ), """The produced output of
    the res_estimation analysis pass is not equal to the expected one"""

    prod_resource_estimation = model.analysis(
        partial(res_estimation_complete, fpgapart=test_fpga_part)
    )
    expect_resource_estimation = {
        "MVAU_hls_0": [
            {
                "BRAM_18K": 0,
                "BRAM_efficiency": 1,
                "LUT": 313,
                "DSP": 1,
                "URAM": 0,
                "URAM_efficiency": 1,
            },
            {
                "BRAM_18K": 0,
                "BRAM_efficiency": 1,
                "LUT": 317,
                "DSP": 0,
                "URAM": 0,
                "URAM_efficiency": 1,
            },
        ]
    }

    assert check_two_dict_for_equality(
        prod_resource_estimation, expect_resource_estimation
    ), """The produced output of
    the res_estimation_complete analysis pass is not equal to the expected one"""


@pytest.mark.fpgadataflow
def test_lookup_uram_res_estimate():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1])
    emb = helper.make_tensor_value_info("emb", TensorProto.FLOAT, [4096, 9])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 9])
    lookup_node = helper.make_node(
        "Lookup_hls",
        ["inp", "emb"],
        ["outp"],
        name="lookup",
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        NumEmbeddings=4096,
        EmbeddingDim=9,
        EmbeddingType=DataType["INT8"].name,
        InputType=DataType["UINT16"].name,
        InputShape=[1],
        mem_mode="internal_embedded",
        ram_style="ultra",
    )
    model = _make_single_hls_node_model(
        lookup_node,
        [inp, emb],
        [outp],
        {"inp": DataType["UINT16"], "emb": DataType["INT8"], "outp": DataType["INT8"]},
    )

    resources = model.analysis(partial(res_estimation, fpgapart=test_fpga_part))
    assert resources["lookup"]["BRAM_18K"] == 0
    assert resources["lookup"]["URAM"] == 1
    assert resources["lookup"]["URAM_efficiency"] == 1

    complete_resources = model.analysis(partial(res_estimation_complete, fpgapart=test_fpga_part))
    alternatives = complete_resources["lookup"]
    assert any(alt["BRAM_18K"] > 0 and alt["URAM"] == 0 for alt in alternatives)
    assert any(alt["LUT"] > 0 and alt["BRAM_18K"] == 0 for alt in alternatives)
    assert any(alt["URAM"] == 1 and alt["BRAM_18K"] == 0 for alt in alternatives)


@pytest.mark.fpgadataflow
def test_thresholding_mlo_uram_res_estimate():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4096])
    thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [4096, 1])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4096])
    thresholding_node = helper.make_node(
        "Thresholding_hls",
        ["inp", "thresh"],
        ["outp"],
        name="thresholding",
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        PE=1,
        NumChannels=4096,
        numSteps=1,
        inputDataType=DataType["INT8"].name,
        weightDataType=DataType["INT8"].name,
        outputDataType=DataType["UINT1"].name,
        numInputVectors=[1],
        mem_mode="internal_decoupled",
        ram_style="ultra",
        mlo_max_iter=2,
    )
    model = _make_single_hls_node_model(
        thresholding_node,
        [inp, thresh],
        [outp],
        {"inp": DataType["INT8"], "thresh": DataType["INT8"], "outp": DataType["UINT1"]},
    )

    resources = model.analysis(partial(res_estimation, fpgapart=test_fpga_part))
    assert resources["thresholding"]["BRAM_18K"] == 0
    assert resources["thresholding"]["URAM"] == 2
    assert resources["thresholding"]["URAM_efficiency"] == 8 * 8192 / (2 * 72 * 4096)


@pytest.mark.fpgadataflow
def test_elementwise_mlo_uram_res_estimate():
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4096])
    param = helper.make_tensor_value_info("param", TensorProto.FLOAT, [1, 4096])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4096])
    elementwise_node = helper.make_node(
        "ElementwiseAdd_hls",
        ["inp", "param"],
        ["outp"],
        name="elementwise",
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        lhs_dtype=DataType["INT8"].name,
        rhs_dtype=DataType["INT8"].name,
        out_dtype=DataType["INT9"].name,
        lhs_shape=[1, 4096],
        rhs_shape=[1, 4096],
        out_shape=[1, 4096],
        lhs_style="input",
        rhs_style="input",
        PE=1,
        ram_style="ultra",
        mlo_max_iter=2,
    )
    model = _make_single_hls_node_model(
        elementwise_node,
        [inp, param],
        [outp],
        {"inp": DataType["INT8"], "param": DataType["INT8"], "outp": DataType["INT9"]},
    )

    resources = model.analysis(partial(res_estimation, fpgapart=test_fpga_part))
    assert resources["elementwise"]["BRAM_18K"] == 0
    assert resources["elementwise"]["URAM"] == 2
    assert resources["elementwise"]["URAM_efficiency"] == 8 * 8192 / (2 * 72 * 4096)
