#pragma once

#include <armadillo>
#include <audio_utils/fft.h>
#include <sffdn/sffdn.h>

#include "loss.h"

#include <cstdint>

class BaseModel
{
  public:
    BaseModel(uint32_t fdn_order);

    std::span<const float> GenerateIR(uint32_t ir_size);

  protected:
    sfFDN::FDN fdn_;
    std::vector<float> impulse_buffer_;
    std::vector<float> response_buffer_;
};

enum class ParamType : uint8_t
{
    Gains,
    Matrix,
};

class FDNModel_InputOutputGain : public BaseModel
{
  public:
    FDNModel_InputOutputGain(uint32_t fdn_order, uint32_t ir_size, std::span<const ParamType> param_types);

    uint32_t GetParamCount() const;

    arma::mat GetInitialParams() const;

    void Setup(const arma::mat& params);

    double Evaluate(const arma::mat& params);
    double Evaluate(const arma::mat& params, const size_t i, const size_t batch_size);

    void Gradient(const arma::mat& x, arma::mat& g);
    void Gradient(const arma::mat& x, const size_t i, arma::mat& g, const size_t batchSize);

    std::span<const float> GetImpulseResponse()
    {
        return GenerateIR(ir_size_);
    }

    sfFDN::FDNConfig GetFDNConfig(const arma::mat& params) const;

    void PrintFDNConfig(const arma::mat& params) const;

    size_t NumFunctions() const
    {
        return 1;
    }

    void Shuffle()
    {
        // No-op
    }

  private:
    uint32_t ir_size_;
    audio_utils::FFT fft_;

    std::vector<float> matrix_coeffs_;
    std::vector<uint32_t> delays_;
    std::vector<ParamType> param_types_;
};