#pragma once

#include <armadillo>
#include <cstdint>
#include <sffdn/sffdn.h>

#include "loss.h"

class BaseModel
{
  public:
    BaseModel(uint32_t fdn_order);

    void SetDelays(std::span<const uint32_t> delays);

    void Setup(const arma::mat& params);

    std::span<const float> GenerateIR(uint32_t ir_size);

  private:
    sfFDN::FDN fdn_;
    std::vector<float> impulse_buffer_;
    std::vector<float> response_buffer_;
};

class FDNModel
{
  public:
    FDNModel(uint32_t fdn_order, uint32_t ir_size, loss::LossFunction loss_function);

    void SetDelays(std::span<const uint32_t> delays);

    double Evaluate(const arma::mat& params);

    std::span<const float> GetImpulseResponse()
    {
        return base_model_.GenerateIR(ir_size_);
    }

  private:
    BaseModel base_model_;
    uint32_t ir_size_;
    loss::LossFunction loss_function_;
};