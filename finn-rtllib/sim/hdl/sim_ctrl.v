/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Simulation control triggering $finish upon asserting sim_finish.
 * @author	Shane T. Fleming <shane.fleming@amd.com>
 ***************************************************************************/
module sim_ctrl(input ap_clk, input sim_finish);
`ifdef FINN_SIMULATION
	initial @(posedge sim_finish) $finish;
	// This ensures there is always a pending #delay in the event queue,
	// preventing the kernel from concluding that the simulation is ending.
	initial forever #1_000_000_000;
`endif
endmodule
