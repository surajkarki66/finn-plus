module sim_ctrl(input ap_clk, input sim_finish);
`ifdef FINN_SIMULATION
	always @(posedge sim_finish) $finish;
	// This ensures there is always a pending #delay in the event queue,
	// preventing the kernel from concluding that the simulation is ending.
	initial forever #1_000_000_000;
`endif
endmodule
