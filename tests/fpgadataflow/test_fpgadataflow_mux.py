import pytest

import jinja2
import onnx.helper as oh
import shlex
import shutil
import subprocess as sp
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.custom_op.registry import getCustomOp
from typing import cast

from finn.custom_op.fpgadataflow.hls.multiplexer_hls import Multiplexer_hls
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.transformation.fpgadataflow.prepare_ip import _codegen_single_node
from finn.util.settings import get_settings


def render_testbench_file(template: Path, destination: Path, **kwargs) -> None:  # noqa
    """Search in finn/custom_hls/mux for the template and render it with the given arguments."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(f"{get_settings().finn_custom_hls}/mux"),
        undefined=jinja2.StrictUndefined,
    )
    loaded_template = env.get_template(str(template))
    destination.write_text(loaded_template.render(**kwargs))


def create_header(node: HLSBackend, file: Path) -> None:
    """Create a header for this HLS operator."""
    node.blackboxfunction()
    guard = file.name.replace(".", "_").upper()
    signature = "\n".join(node.code_gen_dict["$BLACKBOXFUNCTION$"])  # type: ignore
    file.write_text(
        f"#ifndef {guard}\n#define {guard}\n#include "
        f'"bnn-library.h"\n{signature};\n#endif // {guard}'
    )


@pytest.mark.parametrize("mux_type", ["round_robin"])
@pytest.mark.parametrize("part", ["xcu280-fsvh2892-2L-e"])
@pytest.mark.parametrize("clk", [5.0])
@pytest.mark.parametrize("idts", [["UINT2", "INT10"]])
@pytest.mark.parametrize("iterations", [10])
def test_mux(mux_type: str, part: str, clk: float, idts: list[str], iterations: int) -> None:
    """Test code generation and testbenches for mux nodes."""
    node_name = "Multiplexer_top"
    topfile_name = f"top_{node_name}.cpp"

    # Make an onnx node to test
    node = oh.make_node(
        "Multiplexer_hls",
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        name=node_name,
        inputs=[],
        outputs=[],
        muxStrategy=mux_type,
        inStreams=["in0", "in1"],
        inStreamWidths=[10, 20],
        inStreamDataTypes=idts,
        inStreamFoldedOutputShapes=["3,4,1", "10,10"],
        inStreamNormalOutputShapes=["3,4,1", "10,10"],
        outStream="out",
    )

    # Generate the IP source code
    _codegen_single_node(node, None, part, 5.0)

    # Make sure code was generated
    topfile = Path(cast("str", getCustomOp(node).get_nodeattr("code_gen_dir_ipgen"))) / topfile_name
    header_topfile = Path(str(topfile.absolute()).replace("cpp", "h"))
    assert topfile.exists()

    # Create a header. This is required for proper linking,
    # but FINN by default only generates the source file
    create_header(cast("HLSBackend", getCustomOp(node)), header_topfile)

    # Place the Tcl script for executing the testbench next to the topfile
    tb_tcl = topfile.parent / "test_tb.tcl"
    tb_source = topfile.parent / "tb.cpp"
    render_testbench_file(
        Path("test_mux.tcl.jinja"),
        tb_tcl,
        project=f"{mux_type}_test",
        sources=[topfile],
        testbench_source=tb_source,
        top_function=node_name,
        part=part,
        clk=clk,
        finn_hlslib=get_settings().finn_deps / "finn-hlslib",
        solution="sol1",
    )

    # Place the testbench itself also next to the topfile and the Tcl script and the header
    out_dtype = cast("Multiplexer_hls", getCustomOp(node)).get_output_datatype()
    render_testbench_file(
        Path(mux_type) / "tb.cpp.jinja",
        tb_source,
        header=header_topfile,
        hls_streamtypes=[f"hls::stream<{DataType[idt].get_hls_datatype_str()}>" for idt in idts],
        out_streamtype=f"hls::stream<{out_dtype.get_hls_datatype_str()}>",
        out_dtype=out_dtype.get_hls_datatype_str(),
        top_function=node_name,
        iterations=iterations,
    )

    # Execute the testbench Tcl script
    result = sp.run(
        shlex.split(f"vitis_hls -f {tb_tcl}"), cwd=tb_tcl.parent, capture_output=True, text=True
    )
    assert (
        result.returncode == 0
    ), f"Error during testbench execution: \nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
