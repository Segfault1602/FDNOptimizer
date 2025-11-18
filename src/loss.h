#pragma once

#include <audio_utils/fft.h>

#include <functional>
#include <span>

namespace loss
{

using LossFunction = std::function<float(std::span<const float>)>;

float SpectralFlatnessLoss(std::span<const float> signal, audio_utils::FFT& fft);

float RMSLoss(std::span<const float> signal, float target_rms);

} // namespace loss