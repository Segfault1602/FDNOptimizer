#include "loss.h"

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

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
    float sum_squares = 0.0f;
    for (const auto& sample : signal)
    {
        sum_squares += sample * sample;
    }

    float rms = std::sqrt(sum_squares / static_cast<float>(signal.size()));
    return std::abs(rms - target_rms);
}
} // namespace loss