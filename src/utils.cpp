#include "utils.h"

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>
#include <sffdn/sffdn.h>

namespace utils
{

float RMS(std::span<const float> signal)
{
    float sum_squares = 0.0f;
    for (const auto& sample : signal)
    {
        sum_squares += sample * sample;
    }

    float rms = std::sqrt(sum_squares / static_cast<float>(signal.size()));
    return rms;
}

// float EvaluateWithSpectralFlatness(const arma::mat& params, uint32_t fdn_order, uint32_t ir_size)
// {
//     sfFDN::FDN fdn(fdn_order, 256, false);

//     utils::SetupFDNFromParameters(params, fdn);

//     auto feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_order,
//     sfFDN::ScalarMatrixType::Hadamard); fdn.SetFeedbackMatrix(std::move(feedback_matrix));

//     auto delays = sfFDN::GetDelayLengths(fdn_order, 512, 3000, sfFDN::DelayLengthType::Random);
//     fdn.SetDelays(delays);

//     auto output_buffer = utils::GenerateIR(fdn, ir_size);

//     const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(output_buffer.size());
//     audio_utils::FFT fft(fft_size);
//     std::vector<float> spectrum((fft_size / 2) + 1, 0.0f);
//     fft.ForwardAbs(std::span(output_buffer), std::span(spectrum), false, false);

//     float flatness = audio_utils::analysis::SpectralFlatness(spectrum);
//     return flatness;
// }

// arma::mat EvaluateWithSpectralFlatness_Gradient(const arma::mat& params, uint32_t fdn_order, uint32_t ir_size)
// {
//     constexpr float kEpsilon = 1e-6f;

//     arma::mat gradient(1, params.n_cols, arma::fill::zeros);

//     size_t col;
// #pragma omp parallel for private(col) shared(gradient)
//     for (col = 0; col < params.n_cols; ++col)
//     {
//         arma::mat perturbed_params = params;
//         perturbed_params(0, col) += kEpsilon;

//         float value_plus = EvaluateWithSpectralFlatness(perturbed_params, fdn_order, ir_size);

//         perturbed_params = params;
//         perturbed_params(0, col) -= kEpsilon;
//         float value_minus = EvaluateWithSpectralFlatness(perturbed_params, fdn_order, ir_size);

//         float grad = (value_plus - value_minus) / (2 * kEpsilon);
//         gradient(0, col) = grad;
//     }

//     return gradient;
// }
} // namespace utils