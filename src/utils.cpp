#include "utils.h"

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>
#include <sffdn/sffdn.h>

namespace
{

constexpr uint32_t kMinDelay = 256;
constexpr uint32_t kDelayRange = 1000;

void SetupFDNFromParameters_InputOutputGains(sfFDN::FDN& fdn, const arma::mat& params)
{
    const uint32_t fdn_order = fdn.GetOrder();
    if (params.size() != fdn_order * 2)
    {
        throw std::invalid_argument("Not enough parameters to set input and output gains");
    }

    const arma::mat input_gains_mat = params.cols(0, fdn_order - 1);
    const arma::mat output_gains_mat = params.cols(fdn_order, (2 * fdn_order) - 1);

    std::vector<float> input_gains(fdn_order);
    std::vector<float> output_gains(fdn_order);

    for (uint32_t i = 0; i < fdn_order; ++i)
    {
        input_gains[i] = static_cast<float>(input_gains_mat(i));
        output_gains[i] = static_cast<float>(output_gains_mat(i));
    }

    fdn.SetInputGains(input_gains);
    fdn.SetOutputGains(output_gains);
}

double Sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

uint32_t ParamToDelayLength(double param)
{
    return static_cast<uint32_t>(std::round(kMinDelay + ((param * param) * kDelayRange)));
}

void SetupFDNFromParameters_Delays(sfFDN::FDN& fdn, arma::mat params)
{
    const uint32_t fdn_order = fdn.GetOrder();
    if (params.size() != fdn_order)
    {
        throw std::invalid_argument("Not enough parameters to set delays");
    }
    std::vector<uint32_t> delay_lengths(fdn_order);
    for (uint32_t i = 0; i < fdn_order; ++i)
    {
        delay_lengths[i] = ParamToDelayLength(params(i));
    }
    fdn.SetDelays(delay_lengths);
}

} // namespace

namespace utils
{

void PrintParams(const arma::mat& params, uint32_t fdn_order)
{
    std::cout << "FDN Parameters:" << std::endl;

    if (params.n_cols >= fdn_order * 2)
    {
        std::cout << "Input Gains: ";
        for (uint32_t i = 0; i < fdn_order; ++i)
        {
            std::cout << params(i) << " ";
        }
        std::cout << std::endl;

        std::cout << "Output Gains: ";
        for (uint32_t i = 0; i < fdn_order; ++i)
        {
            std::cout << params(i + fdn_order) << " ";
        }
        std::cout << std::endl;
    }

    if (params.n_cols >= fdn_order * 3)
    {
        const arma::mat delay_params = params.cols(fdn_order * 2, (fdn_order * 3) - 1);
        std::cout << "Delays: ";
        for (uint32_t i = 0; i < fdn_order; ++i)
        {
            std::cout << ParamToDelayLength(delay_params(i)) << " ";
        }
        std::cout << std::endl;
    }
}

void SetupFDNFromParameters(const arma::mat& params, sfFDN::FDN& fdn)
{
    const uint32_t fdn_order = fdn.GetOrder();
    if (params.n_cols >= fdn_order * 2)
    {
        const arma::mat inout_gains = params.cols(0, (fdn_order * 2) - 1);
        SetupFDNFromParameters_InputOutputGains(fdn, inout_gains);
    }

    if (params.n_cols >= fdn_order * 3)
    {
        const arma::mat delay_params = params.cols(fdn_order * 2, (fdn_order * 3) - 1);
        SetupFDNFromParameters_Delays(fdn, delay_params);
    }
}

std::vector<float> GenerateIR(sfFDN::FDN& fdn, uint32_t ir_size)
{
    std::vector<float> input_buffer(ir_size, 0.0f);
    input_buffer[0] = 1.0f; // Delta impulse

    std::vector<float> output_buffer(ir_size, 0.0f);

    sfFDN::AudioBuffer input_audio_buffer(input_buffer);
    sfFDN::AudioBuffer output_audio_buffer(output_buffer);

    fdn.Process(input_audio_buffer, output_audio_buffer);

    return output_buffer;
}

float EvaluateWithSpectralFlatness(const arma::mat& params, uint32_t fdn_order, uint32_t ir_size)
{
    sfFDN::FDN fdn(fdn_order, 256, false);

    utils::SetupFDNFromParameters(params, fdn);

    auto feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_order, sfFDN::ScalarMatrixType::Hadamard);
    fdn.SetFeedbackMatrix(std::move(feedback_matrix));

    auto delays = sfFDN::GetDelayLengths(fdn_order, 512, 3000, sfFDN::DelayLengthType::Random);
    fdn.SetDelays(delays);

    auto output_buffer = utils::GenerateIR(fdn, ir_size);

    const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(output_buffer.size());
    audio_utils::FFT fft(fft_size);
    std::vector<float> spectrum((fft_size / 2) + 1, 0.0f);
    fft.ForwardAbs(std::span(output_buffer), std::span(spectrum), false, false);

    float flatness = audio_utils::analysis::SpectralFlatness(spectrum);
    return flatness;
}

arma::mat EvaluateWithSpectralFlatness_Gradient(const arma::mat& params, uint32_t fdn_order, uint32_t ir_size)
{
    constexpr float kEpsilon = 1e-6f;

    arma::mat gradient(1, params.n_cols, arma::fill::zeros);

    size_t col;
#pragma omp parallel for private(col) shared(gradient)
    for (col = 0; col < params.n_cols; ++col)
    {
        arma::mat perturbed_params = params;
        perturbed_params(0, col) += kEpsilon;

        float value_plus = EvaluateWithSpectralFlatness(perturbed_params, fdn_order, ir_size);

        perturbed_params = params;
        perturbed_params(0, col) -= kEpsilon;
        float value_minus = EvaluateWithSpectralFlatness(perturbed_params, fdn_order, ir_size);

        float grad = (value_plus - value_minus) / (2 * kEpsilon);
        gradient(0, col) = grad;
    }

    return gradient;
}
} // namespace utils