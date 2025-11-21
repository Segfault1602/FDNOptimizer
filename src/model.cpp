#include "model.h"

#include <armadillo>
#include <sffdn/sffdn.h>

#include "utils.h"

#include <cassert>
#include <cstdint>
#include <iostream>

namespace
{
constexpr uint32_t kDefaultFdnBlockSize = 256;
constexpr uint32_t kSampleRate = 48000;

arma::mat ParamToGain(const arma::mat& params)
{
    arma::mat gains = params;
    // gains /= gains.n_cols;
    return gains;
}

arma::mat ParamToGains(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.N;
    assert(params.n_cols >= 2 * fdn_order);

    arma::mat input_gains_arma = ParamToGain(params.cols(0, fdn_order - 1));
    arma::mat output_gains_arma = ParamToGain(params.cols(fdn_order, (2 * fdn_order) - 1));

    config.input_gains.resize(fdn_order);
    config.output_gains.resize(fdn_order);

    for (uint32_t i = 0; i < fdn_order; ++i)
    {
        config.input_gains[i] = static_cast<float>(input_gains_arma(i));
        config.output_gains[i] = static_cast<float>(output_gains_arma(i));
    }

    const size_t start_offset = 2 * fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(2 * fdn_order, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamToMatrix(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.N;
    assert(params.n_cols >= fdn_order * fdn_order);

    arma::mat M = params.cols(0, (fdn_order * fdn_order) - 1);
    M.reshape(fdn_order, fdn_order);

    arma::mat Q, R;
    arma::qr_econ(Q, R, M);
    Q = Q * arma::diagmat(arma::sign(R.diag()));

    // arma::mat test = Q.t() * Q;
    // test.print("Q^T * Q:");

    std::vector<float> matrix_coeffs(fdn_order * fdn_order);
    for (uint32_t r = 0; r < fdn_order; ++r)
    {
        for (uint32_t c = 0; c < fdn_order; ++c)
        {
            matrix_coeffs[r * fdn_order + c] = static_cast<float>(Q(r, c));
        }
    }

    config.matrix_info = std::move(matrix_coeffs);

    const size_t start_offset = fdn_order * fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(start_offset, params.n_cols - 1);
    return leftover_params;
}
} // namespace

BaseModel::BaseModel(uint32_t fdn_order)
    : fdn_(fdn_order, kDefaultFdnBlockSize, false)
{
    fdn_.SetDirectGain(0.0f);
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

FDNModel_InputOutputGain::FDNModel_InputOutputGain(uint32_t fdn_order, uint32_t ir_size,
                                                   std::span<const ParamType> param_types)
    : BaseModel(fdn_order)
    , ir_size_(ir_size)
    , fft_(audio_utils::FFT::NextSupportedFFTSize(ir_size))
    , param_types_(param_types.begin(), param_types.end())
{
    constexpr uint32_t kRandomSeed = 42;
    matrix_coeffs_ = sfFDN::GenerateMatrix(fdn_order, sfFDN::ScalarMatrixType::Random, kRandomSeed);
    auto feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_order);
    feedback_matrix->SetMatrix(matrix_coeffs_);
    fdn_.SetFeedbackMatrix(std::move(feedback_matrix));

    // Following delays are from [1]
    // [1] G. D. Santo, K. Prawda, S. J. Schlecht, and V. Välimäki, “Efficient Optimization of Feedback Delay Networks
    // for Smooth Reverberation,” Aug. 28, 2024, arXiv: arXiv:2402.11216. doi: 10.48550/arXiv.2402.11216.
    if (fdn_order == 4)
    {
        delays_ = {1499, 1889, 2381, 2999};
    }
    else if (fdn_order == 6)
    {
        delays_ = {997, 1153, 1327, 1559, 1801, 2099};
    }
    else if (fdn_order == 8)
    {
        delays_ = {809, 877, 937, 1049, 1151, 1249, 1373, 1499};
    }
    else
    {
        delays_ = sfFDN::GetDelayLengths(fdn_order, 512, 3000, sfFDN::DelayLengthType::Random, kRandomSeed);
    }
    fdn_.SetDelays(delays_);

    constexpr std::array<float, 1> t60s = {10.0f};
    auto attenuation_filter = sfFDN::CreateAttenuationFilterBank(t60s, delays_, kSampleRate);
    fdn_.SetFilterBank(std::move(attenuation_filter));
}

uint32_t FDNModel_InputOutputGain::GetParamCount() const
{
    uint32_t count = 0;
    const uint32_t fdn_order = fdn_.GetOrder();
    for (const auto& type : param_types_)
    {
        switch (type)
        {
        case ParamType::Gains:
            count += 2 * fdn_order;
            break;
        case ParamType::Matrix:
            count += fdn_order * fdn_order;
            break;
        default:
            throw std::runtime_error("Unknown ParamType in GetParamCount");
        }
    }
    return count;
}

arma::mat FDNModel_InputOutputGain::GetInitialParams() const
{
    arma::mat params(1, GetParamCount(), arma::fill::randn);
    return params;
}

void FDNModel_InputOutputGain::Setup(const arma::mat& params)
{
    arma::mat params_to_process = params;
    sfFDN::FDNConfig config;
    config.N = fdn_.GetOrder();
    for (const auto& type : param_types_)
    {
        switch (type)
        {
        case ParamType::Gains:
        {
            params_to_process = ParamToGains(config, params_to_process);
            fdn_.SetInputGains(config.input_gains);
            fdn_.SetOutputGains(config.output_gains);
        }
        break;
        case ParamType::Matrix:
        {
            params_to_process = ParamToMatrix(config, params_to_process);
            auto feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_.GetOrder());
            feedback_matrix->SetMatrix(std::get<std::vector<float>>(config.matrix_info));
            fdn_.SetFeedbackMatrix(std::move(feedback_matrix));
        }
        break;
        default:
            throw std::runtime_error("Unknown ParamType in Setup");
        }
    }
}

double FDNModel_InputOutputGain::Evaluate(const arma::mat& params)
{
    Setup(params);
    std::span<const float> ir = GenerateIR(ir_size_);

    constexpr uint32_t kRMSSampleCount = 1024;
    constexpr float kRMSTarget = 0.01f;
    constexpr float kRMSWeight = 0.1f;

    double spectral_loss = loss::SpectralFlatnessLoss(ir, fft_);
    double rms_loss = loss::RMSLoss(ir.subspan(ir.size() - kRMSSampleCount), kRMSTarget);

    double total_loss = spectral_loss + (kRMSWeight * rms_loss);
    return total_loss;
}

double FDNModel_InputOutputGain::Evaluate(const arma::mat& params, const size_t i, const size_t batch_size)
{
    assert(i == 0 && batch_size == 1);
    return Evaluate(params);
}

void FDNModel_InputOutputGain::Gradient(const arma::mat& x, arma::mat& g)
{
    constexpr float kEpsilon = 1e-1f;

    g.zeros(x.n_rows, x.n_cols);
    size_t col;
#pragma omp parallel for private(col) shared(g)
    for (col = 0; col < x.n_cols; ++col)
    {
        arma::mat x_plus = x;
        x_plus(0, col) += kEpsilon;
        // Creating a whole new model to avoid threading issues
        FDNModel_InputOutputGain plus_model(fdn_.GetOrder(), ir_size_, param_types_);
        double plus_value = plus_model.Evaluate(x_plus);

        arma::mat x_minus = x;
        x_minus(0, col) -= kEpsilon;
        double minus_value = plus_model.Evaluate(x_minus);

        g(0, col) = (plus_value - minus_value) / (2 * kEpsilon);
    }
}

void FDNModel_InputOutputGain::Gradient(const arma::mat& x, const size_t i, arma::mat& g, const size_t batchSize)
{
    assert(i == 0 && batchSize == 1);
    Gradient(x, g);
}

sfFDN::FDNConfig FDNModel_InputOutputGain::GetFDNConfig(const arma::mat& params) const
{
    arma::mat params_to_process = params;
    sfFDN::FDNConfig config;
    config.N = fdn_.GetOrder();
    config.delays = delays_;
    config.matrix_info = matrix_coeffs_;

    for (const auto& type : param_types_)
    {
        switch (type)
        {
        case ParamType::Gains:
        {
            params_to_process = ParamToGains(config, params_to_process);
        }
        break;
        case ParamType::Matrix:
        {
            params_to_process = ParamToMatrix(config, params_to_process);
        }
        break;
        default:
            throw std::runtime_error("Unknown ParamType in Setup");
        }
    }

    return config;
}

void FDNModel_InputOutputGain::PrintFDNConfig(const arma::mat& params) const
{
    sfFDN::FDNConfig config = GetFDNConfig(params);

    arma::fvec input_gains_arma(config.input_gains.data(), config.N);
    arma::fvec output_gains_arma(config.output_gains.data(), config.N);

    std::cout << "FDN Configuration:----------------------" << std::endl;
    // std::cout << "    Input Gains: [" << input_gains_arma.t();
    // std::cout << "    Output Gains: [" << output_gains_arma.t();
    input_gains_arma.t().print("Input Gains:");
    output_gains_arma.t().print("Output Gains:");
    std::cout << "Delays: [";
    for (const auto& delay : delays_)
    {
        std::cout << delay << " ";
    }
    std::cout << "]" << std::endl;

    std::vector<float> matrix_data = std::get<std::vector<float>>(config.matrix_info);
    arma::fmat matrix_data_arma(matrix_data.data(), config.N, config.N);

    matrix_data_arma.print("Feedback Matrix:");
    std::cout << "----------------------------------------" << std::endl;
}