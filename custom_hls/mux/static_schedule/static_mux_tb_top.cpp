#include "static_mux_tb_top.h"

void Mux(S1 &a, S2 &b, S3 &c, S2 &d, S2 &e, S3 &out) {
#pragma HLS INTERFACE axis port=a
#pragma HLS INTERFACE axis port=b
#pragma HLS INTERFACE axis port=c
#pragma HLS INTERFACE axis port=d
#pragma HLS INTERFACE axis port=e
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE ap_ctrl_none port=return
    static_mux(std::index_sequence<0, 1, 2, 3, 4>{}, out, a, b, c, d, e);
}
