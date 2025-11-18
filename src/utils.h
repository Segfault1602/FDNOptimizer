#pragma once

#include <armadillo>

#include <sffdn/sffdn.h>

#include <cstdint>
#include <vector>

namespace utils
{

void PrintParams(const arma::mat& params, uint32_t fdn_order);

void SetupFDNFromParameters(const arma::mat& params, sfFDN::FDN& fdn);

std::vector<float> GenerateIR(sfFDN::FDN& fdn, uint32_t ir_size);

float EvaluateWithSpectralFlatness(const arma::mat& params, uint32_t fdn_order, uint32_t ir_size);

arma::mat EvaluateWithSpectralFlatness_Gradient(const arma::mat& params, uint32_t fdn_order, uint32_t ir_size);

} // namespace utils