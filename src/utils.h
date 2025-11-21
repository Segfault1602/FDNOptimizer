#pragma once

#include <armadillo>

#include <sffdn/sffdn.h>

#include <cstdint>
#include <vector>

namespace utils
{
void PrintParams(const arma::mat& params, uint32_t fdn_order);

sfFDN::FDNConfig ParametersToFDNConfig(const arma::mat& params, uint32_t fdn_order);

float RMS(std::span<const float> signal);

template <typename T>
T Db2Mag(T db)
{
    return std::pow(static_cast<T>(10), db / static_cast<T>(20));
}

template <typename T>
T RT602Slope(T t60, uint16_t sr)
{
    return static_cast<T>(-60) / (t60 * static_cast<T>(sr));
}

} // namespace utils