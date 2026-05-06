# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for build configuration checks."""

import pytest

import json
import os
from onnx import TensorProto, helper
from unittest.mock import patch

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
)
from finn.util.basic import make_build_dir


def make_test_model(build_dir):
    """Create minimal ONNX model for testing."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["inp"], ["out"])
    graph = helper.make_graph([node], "test", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model_path = os.path.join(build_dir, "model.onnx")
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())
    return model_path


def cfg(output_dir, **kw):
    """Create config that stops immediately after first step."""
    return DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=5.0,
        stop_step="step_qonnx_to_finn",
        generate_outputs=kw.pop("generate_outputs", [DataflowOutputType.ESTIMATE_REPORTS]),
        **kw
    )


class TestConfigCheckIntegration:
    def test_report_files_created(self):
        """Config check report should be saved to output_dir."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            build_dataflow_cfg(
                model_path,
                cfg(output_dir, board="Pynq-Z1", shell_flow_type=ShellFlowType.VIVADO_ZYNQ),
            )

        assert os.path.exists(os.path.join(output_dir, "config_check_report.txt"))
        assert os.path.exists(os.path.join(output_dir, "config_check_report.json"))

    def test_invalid_config_raises(self):
        """Invalid config should raise AssertionError."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with pytest.raises(AssertionError, match="Configuration check failed"):
            build_dataflow_cfg(
                model_path, cfg(output_dir, board="V80", shell_flow_type=ShellFlowType.VITIS_ALVEO)
            )

    def test_muted_config_proceeds(self):
        """Invalid config with mute_config_assertions=True should not raise."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        # This would normally fail (V80 needs SLASH_ALVEO), but muting allows it to proceed
        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2025.1"}):
            try:
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        board="V80",
                        shell_flow_type=ShellFlowType.VITIS_ALVEO,
                        mute_config_assertions=True,
                    ),
                )
            except AssertionError as e:
                assert "Configuration check failed" not in str(e)
            except Exception:
                pass  # Other errors are fine

        assert os.path.exists(os.path.join(output_dir, "config_check_report.txt"))

    def test_report_contains_errors(self):
        """Report JSON should contain the detected errors."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2025.1"}):
            try:
                build_dataflow_cfg(
                    model_path,
                    cfg(
                        output_dir,
                        board="V80",
                        shell_flow_type=ShellFlowType.VITIS_ALVEO,
                        mute_config_assertions=True,
                    ),
                )
            except Exception:
                pass

        with open(os.path.join(output_dir, "config_check_report.json")) as f:
            report = json.load(f)

        assert report["summary"]["errors"] > 0
        error_names = [
            c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "ERROR"
        ]
        assert "v80_shell" in error_names

    @pytest.mark.parametrize(
        "board,flow,should_error",
        [
            ("Pynq-Z1", ShellFlowType.VIVADO_ZYNQ, False),
            ("U250", ShellFlowType.VITIS_ALVEO, False),
            ("Pynq-Z1", ShellFlowType.VITIS_ALVEO, True),
            ("U250", ShellFlowType.VIVADO_ZYNQ, True),
        ],
    )
    def test_board_shell_compatibility(self, board, flow, should_error):
        """Test various board/shell flow combinations."""
        build_dir = make_build_dir("test_config_check_")
        model_path = make_test_model(build_dir)
        output_dir = os.path.join(build_dir, "output")

        with patch.dict("os.environ", {"XILINX_VIVADO": "/tools/Vivado/2024.2"}):
            if should_error:
                with pytest.raises(AssertionError, match="Configuration check failed"):
                    build_dataflow_cfg(
                        model_path, cfg(output_dir, board=board, shell_flow_type=flow)
                    )
            else:
                build_dataflow_cfg(model_path, cfg(output_dir, board=board, shell_flow_type=flow))
                assert os.path.exists(os.path.join(output_dir, "config_check_report.txt"))
