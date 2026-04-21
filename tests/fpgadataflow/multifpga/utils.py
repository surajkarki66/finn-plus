"""Utils for Multi-FPGA testing."""
import pytest

import brevitas.nn as qnn
import configparser
import numpy as np
import torch
from brevitas.export import export_qonnx
from brevitas_examples.bnn_pynq.models.resnet import quant_resnet18
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper

from finn.util.basic import make_build_dir


class RN18(torch.nn.Module):
    """A simple ResNet-18 with an input quantizer."""

    def __init__(self, cfg: configparser.ConfigParser) -> None:
        """Create a simple ResNet-18 from scratch with the given config."""
        super().__init__()
        self.inpQuantizer = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.resnet = quant_resnet18(cfg)

    def forward(self, x):  # noqa
        x = self.inpQuantizer(x)
        return self.resnet(x)


def create_rn18_model(w: int, a: int, classes: int = 100) -> RN18:
    """Create a ResNet-18 (Brevitas model) with the given weight and activation bitwidths."""
    cfg = configparser.ConfigParser()
    cfg["MODEL"] = {"NUM_CLASSES": str(classes)}
    cfg["QUANT"] = {"WEIGHT_BIT_WIDTH": str(w), "ACT_BIT_WIDTH": str(a)}
    return RN18(cfg)


def create_rn18_onnx(path: Path, w: int, a: int, classes: int = 100) -> None:
    """Create a ResNet-18 and export as a QONNX model to the given path."""
    model = create_rn18_model(w, a, classes)
    model.eval()
    inp = torch.zeros((1, 3, 32, 32))
    _ = model(inp)
    export_qonnx(model, (inp,), str(path.absolute()))


@pytest.mark.multifpga
def test_rn18_onnx_creation() -> None:
    """Test that the RN18 is created correctly and can be loaded by a
    QONNX modelwrapper.
    """
    rn18_path = Path(make_build_dir("TEST-RN18")) / "rn18.onnx"
    create_rn18_onnx(rn18_path, 4, 4, 100)
    assert rn18_path.exists()
    _ = ModelWrapper(str(rn18_path))
