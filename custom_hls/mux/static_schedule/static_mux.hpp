#ifndef STATIC_MUX_HPP
#define STATIC_MUX_HPP

#include "ap_int.h"
#include "hls_stream.h"
#include <type_traits>
#include <tuple>
#include <initializer_list>
#include <functional>
#include <utility>

constexpr std::size_t bitwidth(std::size_t i) {
    // Every bit halves the number of choices addressed, so the
    // next call only deals with i/2 (i>>1)
    return (i < 2 ? i : 1 + bitwidth(i >> 1));
}
static_assert(bitwidth(2) == 2, "Bitwidth for 2 should be 2");
static_assert(bitwidth(11) == 4, "Bitwidth for 11 should be 4");


inline void _static_mux() {}

template<std::size_t idx = 0, typename TOut, int FOut, typename T, int F>
void _static_mux(hls::stream<TOut, FOut>& out, std::reference_wrapper<hls::stream<T, F>> &stream) {
    if (!stream.get().empty()) { out.write(stream.get().read()); }
}

template<std::size_t idx = 0, typename TOut, int FOut, typename T, int F, typename ...Ts, int ...Fs>
void _static_mux(hls::stream<TOut, FOut>& out, std::reference_wrapper<hls::stream<T, F>> &stream, std::reference_wrapper<hls::stream<Ts, Fs>>& ...others) {
    // TODO: ADD HEADER
    if (!stream.get().empty()) { out.write(stream.get().read()); }
    _static_mux<idx+1>(out, others...);
}

template<
    int N,
    int FOut,
    typename ...Ts,
    int ...Fs,
    std::size_t ...I
    >
void static_mux(
    std::index_sequence<I...>,
    hls::stream<ap_uint<N>, FOut> &out,
    hls::stream<Ts, Fs>& ...others
) {
    std::tuple<std::reference_wrapper<hls::stream<Ts, Fs>>...> streams = std::make_tuple(std::ref(others)... );
    _static_mux(out, std::get<I>(streams)...);
}

#endif
