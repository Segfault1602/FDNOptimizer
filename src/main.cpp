// #define ENS_PRINT_INFO
// #define ENS_PRINT_WARN
#include <ensmallen.hpp>

#include <audio_utils/audio_file_manager.h>

#include "loss.h"
#include "model.h"
#include "utils.h"

#include <iostream>

int main()
{
    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kIrSize = 1 << 16;
    constexpr uint32_t kFdnOrder = 8;
    constexpr uint32_t kRMSSampleCount = 8192;

    // arma::arma_rng::set_seed_random();
    constexpr uint32_t kParamCount = kFdnOrder * 3;
    arma::mat params(1, kParamCount, arma::fill::randn);

    std::cout << "Params: " << params << std::endl;
    utils::PrintParams(params, kFdnOrder);

    const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(kIrSize);
    audio_utils::FFT fft(fft_size);

    auto loss_function = [&fft](std::span<const float> signal) -> float {
        return loss::SpectralFlatnessLoss(signal, fft) +
               0.1 * loss::RMSLoss(signal.subspan(signal.size() - kRMSSampleCount), 0.1f);
    };

    FDNModel model(kFdnOrder, kIrSize, loss_function);

    // constexpr std::array<uint32_t, kFdnOrder> kDelays = {2712, 2981, 1580, 1023, 2992, 1371, 1161, 1578};
    // model.SetDelays(kDelays);

    model.Evaluate(params);
    auto initial_ir = model.GetImpulseResponse();
    audio_utils::audio_file::WriteWavFile("initial_ir.wav", initial_ir, kSampleRate);

    auto initial_flatness = loss::SpectralFlatnessLoss(initial_ir, fft);
    auto initial_rms = loss::RMSLoss(initial_ir.subspan(initial_ir.size() - kRMSSampleCount), 0.1f);

    // Simulated Annealing Optimizer
    // ens::SA<> optimizer(ens::ExponentialSchedule(), 1e6, 10000.0, 1000, 100, 1e-3, 3, 1, 0.03, 0.5);

    // CNE
    // ens::CNE optimizer(1000, 50000, 0.2, 0.05, 0.3, 1e-6);

    // DE
    // ens::DE optimizer(1000, 2000, 0.6, 0.8, 1e-5);

    // PSO
    // ens::LBestPSO optimizer(128, 1, 1, 3000, 350, 1e-5, 2.05, 2.05);

    // SPSA
    ens::SPSA optimizer(0.0546487, 0.608178, 0.99438, 0.40484, 1e6, 1e-5);

    optimizer.Optimize(model, params, ens::Report());

    std::cout << std::endl;
    std::cout << "Optimized Params: " << params << std::endl;
    utils::PrintParams(params, kFdnOrder);

    double final_loss = model.Evaluate(params);
    auto final_ir = model.GetImpulseResponse();
    audio_utils::audio_file::WriteWavFile("final_ir.wav", final_ir, kSampleRate);

    auto final_flatness = loss::SpectralFlatnessLoss(final_ir, fft);
    auto final_rms = loss::RMSLoss(final_ir.subspan(final_ir.size() - kRMSSampleCount), 0.1f);

    std::cout << "Final loss: " << final_loss << std::endl;
    std::cout << "Spectral Flatness init: " << 1.0f - initial_flatness << " final: " << 1.0 - final_flatness
              << std::endl;
    std::cout << "IR RMS init: " << initial_rms << " final: " << final_rms << std::endl;
}