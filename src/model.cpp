#include "model.h"

#include <armadillo>
#include <sffdn/sffdn.h>

#include "utils.h"

#include <cstdint>
#include <iostream>

namespace
{
constexpr uint32_t kDefaultFdnBlockSize = 256;
} // namespace

BaseModel::BaseModel(uint32_t fdn_order)
    : fdn_(fdn_order, kDefaultFdnBlockSize, false)
{
    fdn_.SetDirectGain(0.0f);

    auto feedback_matrix =
        std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_.GetOrder(), sfFDN::ScalarMatrixType::Hadamard);
    fdn_.SetFeedbackMatrix(std::move(feedback_matrix));
}

void BaseModel::SetDelays(std::span<const uint32_t> delays)
{
    fdn_.SetDelays(delays);
}

void BaseModel::Setup(const arma::mat& params)
{
    utils::SetupFDNFromParameters(params, fdn_);
}

std::span<const float> BaseModel::GenerateIR(uint32_t ir_size)
{
    if (response_buffer_.size() < ir_size)
    {
        response_buffer_.resize(ir_size);
    }

    if (impulse_buffer_.size() < ir_size)
    {
        impulse_buffer_.resize(ir_size);
    }

    std::ranges::fill(impulse_buffer_, 0.0f);
    std::ranges::fill(response_buffer_, 0.0f);

    impulse_buffer_[0] = 1.0f; // Delta impulse

    sfFDN::AudioBuffer in_buffer(impulse_buffer_);
    sfFDN::AudioBuffer out_buffer(response_buffer_);
    fdn_.Clear();
    fdn_.Process(in_buffer, out_buffer);

    return std::span<const float>(response_buffer_);
}

FDNModel::FDNModel(uint32_t fdn_order, uint32_t ir_size, loss::LossFunction loss_function)
    : base_model_(fdn_order)
    , ir_size_(ir_size)
    , loss_function_(loss_function)
{
}

void FDNModel::SetDelays(std::span<const uint32_t> delays)
{
    base_model_.SetDelays(delays);
}

double FDNModel::Evaluate(const arma::mat& params)
{
    base_model_.Setup(params);
    std::span<const float> ir = base_model_.GenerateIR(ir_size_);
    return static_cast<double>(loss_function_(ir));
}