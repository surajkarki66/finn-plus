module $TOP_MODULE_NAME$(
//- Global Control ------------------
(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET = ap_rst_n" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
input   ap_clk,
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
input   ap_rst_n,

//- AXI4 Slave - Write Address -----
input  [$ADDR_WIDTH$-1:0] s_axi_awaddr,
input                   s_axi_awvalid,
output                  s_axi_awready,

//- AXI4 Slave - Write Data --------
input  [$DATA_WIDTH$-1:0] s_axi_wdata,
input  [$DATA_BYTES$-1:0] s_axi_wstrb,
input                    s_axi_wvalid,
input                    s_axi_wlast,
output                   s_axi_wready,

//- AXI4 Slave - Write Response ----
output reg [1:0]         s_axi_bresp,
output reg               s_axi_bvalid,
input                    s_axi_bready,

//- AXI4 Slave - Read Address ------
input  [$ADDR_WIDTH$-1:0] s_axi_araddr,
input                    s_axi_arvalid,
output                   s_axi_arready,

//- AXI4 Slave - Read Data ---------
output reg [$DATA_WIDTH$-1:0] s_axi_rdata,
output reg [1:0]             s_axi_rresp,
output reg                   s_axi_rvalid,
output reg                   s_axi_rlast,
input                        s_axi_rready
);

parameter integer LATENCY = 100;

// Internal flags and counters
reg aw_received;
reg w_received;
reg ar_received;
reg [$clog2(LATENCY+1)-1:0] write_cnt;
reg [$clog2(LATENCY+1)-1:0] read_cnt;
reg busy_write;
reg busy_read;

// Ready signals: accept new addr/data when not busy and not already pending
assign s_axi_awready = !busy_write && !aw_received;
assign s_axi_wready  = !busy_write && !w_received;
assign s_axi_arready = !busy_read && !ar_received;

// Default response values
always @(posedge ap_clk) begin
	if (!ap_rst_n) begin
		aw_received <= 1'b0;
		w_received <= 1'b0;
		ar_received <= 1'b0;
		busy_write <= 1'b0;
		busy_read  <= 1'b0;
		write_cnt  <= {($clog2(LATENCY+1)){1'b0}};
		read_cnt   <= {($clog2(LATENCY+1)){1'b0}};
		s_axi_bvalid <= 1'b0;
		s_axi_bresp  <= 2'b00;
		s_axi_rvalid <= 1'b0;
		s_axi_rresp  <= 2'b00;
		s_axi_rdata  <= {${DATA_WIDTH}${1'b0}};
		s_axi_rlast  <= 1'b0;
	end else begin
		// Capture write address
		if (s_axi_awvalid && s_axi_awready) begin
			aw_received <= 1'b1;
		end

		// Capture write data (we only need to see WLAST to consider a complete write)
		if (s_axi_wvalid && s_axi_wready) begin
			if (s_axi_wlast) begin
				w_received <= 1'b1;
			end
		end

		// Start write transaction when both address and data received
		if (aw_received && w_received && !busy_write) begin
			busy_write <= 1'b1;
			write_cnt  <= LATENCY - 1; // will count down
			aw_received <= 1'b0;
			w_received  <= 1'b0;
		end

		// Decrement write counter
		if (busy_write) begin
			if (write_cnt != 0) begin
				write_cnt <= write_cnt - 1;
			end else begin
				busy_write <= 1'b0;
				s_axi_bvalid <= 1'b1; // respond OKAY after latency
				s_axi_bresp  <= 2'b00;
			end
		end

		// B channel handshake
		if (s_axi_bvalid && s_axi_bready) begin
			s_axi_bvalid <= 1'b0;
		end

		// Capture read address
		if (s_axi_arvalid && s_axi_arready) begin
			ar_received <= 1'b1;
		end

		// Start read transaction when address captured
		if (ar_received && !busy_read) begin
			busy_read <= 1'b1;
			read_cnt  <= LATENCY - 1;
			ar_received <= 1'b0;
		end

		// Decrement read counter
		if (busy_read) begin
			if (read_cnt != 0) begin
				read_cnt <= read_cnt - 1;
			end else begin
				busy_read <= 1'b0;
				s_axi_rvalid <= 1'b1;
				s_axi_rresp  <= 2'b00;
				s_axi_rdata  <= {${DATA_WIDTH}${1'b0}}; // return zeros for reads
				s_axi_rlast  <= 1'b1;
			end
		end

		// R channel handshake
		if (s_axi_rvalid && s_axi_rready) begin
			s_axi_rvalid <= 1'b0;
			s_axi_rlast  <= 1'b0;
		end
	end
end

endmodule
