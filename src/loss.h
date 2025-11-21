#pragma once

#include <audio_utils/fft.h>

#include <cstdint>
#include <functional>
#include <span>

namespace loss
{

using LossFunction = std::function<float(std::span<const float>)>;

float SpectralFlatnessLoss(std::span<const float> signal, audio_utils::FFT& fft);

float RMSLoss(std::span<const float> signal, float target_rms);

float MixingTimeLoss(std::span<const float> signal, uint32_t sample_rate, float target_mixing_time);

} // namespace loss