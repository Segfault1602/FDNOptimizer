#include "loss.h"

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

#include "echo_density.h"
#include "utils.h"

namespace loss
{
float SpectralFlatnessLoss(std::span<const float> signal, audio_utils::FFT& fft)
{
    // const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(signal.size());
    // audio_utils::FFT fft(fft_size);
    std::vector<float> spectrum((fft.GetFFTSize() / 2) + 1, 0.0f);
    fft.ForwardAbs(signal, std::span(spectrum), false, false);

    float flatness = audio_utils::analysis::SpectralFlatness(spectrum);
    return 1.f - flatness;
}

float RMSLoss(std::span<const float> signal, float target_rms)
{
    float rms = utils::RMS(signal);
    return std::abs(rms - target_rms);
}

float MixingTimeLoss(std::span<const float> signal, uint32_t sample_rate, float target_mixing_time)
{
    constexpr float kWindowSizeMs = 50.0f;
    constexpr float kHopSizeMs = 10.0f;
    const uint32_t window_size = static_cast<uint32_t>((kWindowSizeMs / 1000.0f) * static_cast<float>(sample_rate));
    const uint32_t hop_size = static_cast<uint32_t>((kHopSizeMs / 1000.0f) * static_cast<float>(sample_rate));
    auto results = EchoDensity(signal, window_size, sample_rate, hop_size);

    return std::abs(results.mixing_time - target_mixing_time);
}

} // namespace loss