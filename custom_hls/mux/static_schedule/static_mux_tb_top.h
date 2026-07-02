#ifndef STATIC_MUX_TOP_H
#define STATIC_MUX_TOP_H

#include "static_mux.hpp"

using T1 = ap_uint<2>;
using T2 = ap_uint<10>;
using T3 = ap_int<14>;
using S1 = hls::stream<T1, 20>;
using S2 = hls::stream<T2, 20>;
using S3 = hls::stream<T3, 20>;

void Mux(S1 &a, S2 &b, S3 &c, S2 &d, S2 &e, S3 &out);

#endif
