#ifndef STATIC_MUX_H
#define STATIC_MUX_H

#include "ap_int.h"
#include "hls_stream.h"
#include <type_traits>
#include <tuple>
#include <initializer_list>
#include <functional>
#include <utility>

inline void _static_mux() {}

template<typename TOut, int FOut, typename T, int F>
void _static_mux(hls::stream<TOut, FOut>& out, std::reference_wrapper<hls::stream<T, F>> &stream) {
    if (!stream.get().empty()) {
        out.write(stream.get().read());
    }
}

template<typename TOut, int FOut, typename T, int F, typename ...Ts, int ...Fs>
void _static_mux(hls::stream<TOut, FOut>& out, std::reference_wrapper<hls::stream<T, F>> &stream, std::reference_wrapper<hls::stream<Ts, Fs>>& ...others) {
    if (!stream.get().empty()) {
        out.write(stream.get().read());
    }
    _static_mux(out, others...);
}

template<
    typename TOut,
    int FOut,
    typename ...Ts,
    int ...Fs,
    std::size_t ...I
    >
void static_mux(
    std::index_sequence<I...>,
    hls::stream<TOut, FOut> &out,
    hls::stream<Ts, Fs>& ...others
) {
    std::tuple<std::reference_wrapper<hls::stream<Ts, Fs>>...> streams = std::make_tuple(std::ref(others)... );
    _static_mux(out, std::get<I>(streams)...);
}

#endif
