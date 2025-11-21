// #define ENS_PRINT_INFO
// #define ENS_PRINT_WARN
#include <ensmallen.hpp>

#include <audio_utils/audio_file_manager.h>
#include <nlohmann/json.hpp>
#include <sffdn/sffdn.h>

#include "loss.h"
#include "model.h"
#include "utils.h"

#include <cstdint>
#include <fstream>
#include <iostream>

void WriteFDNConfigToJson(const sfFDN::FDNConfig& fdn_config, const std::string& filename)
{
    nlohmann::json j;
    to_json(j, fdn_config);

    std::ofstream file(filename);
    if (!file)
    {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    file << j.dump(4);
    file.close();
}

int main()
{
    arma::arma_rng::set_seed_random();

    constexpr uint32_t kSampleRate = 48000;
    constexpr uint32_t kIrSize = 1 << 15;
    constexpr uint32_t kFFTSize = kIrSize;
    constexpr uint32_t kFdnOrder = 6;
    constexpr uint32_t kRMSSampleCount = 1024;

    const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(kFFTSize);
    audio_utils::FFT fft(fft_size);

    constexpr std::array param_types = {ParamType::Gains, ParamType::Matrix};
    FDNModel_InputOutputGain model(kFdnOrder, kIrSize, param_types);

    arma::mat params = model.GetInitialParams();

    std::cout << "Initial Params: " << params << std::endl;
    model.PrintFDNConfig(params);

    WriteFDNConfigToJson(model.GetFDNConfig(params), "init_fdn_config.json");

    model.Evaluate(params);
    auto initial_ir = model.GetImpulseResponse();
    audio_utils::audio_file::WriteWavFile("initial_ir.wav", initial_ir, kSampleRate);

    auto initial_flatness = loss::SpectralFlatnessLoss(initial_ir, fft);
    auto initial_rms = utils::RMS(initial_ir.subspan(initial_ir.size() - kRMSSampleCount));

    // Simulated Annealing Optimizer
    // ens::SA<> optimizer(ens::ExponentialSchedule(), 1e6, 10000.0, 1000, 100, 1e-3, 3, 1, 0.03, 0.5);

    // CNE
    // ens::CNE optimizer(1000, 50000, 0.2, 0.05, 0.3, 1e-6);

    // DE
    // ens::DE optimizer(1000, 2000, 0.6, 0.8, 1e-5);

    // PSO
    // ens::LBestPSO optimizer(128, 1, 1, 3000, 350, 1e-5, 2.05, 2.05);

    // SPSA
    // ens::SPSA optimizer(0.0389942, 0.171879, 0.95201, 0.36284, 1e6, 1e-5);

    // ens::L_BFGS optimizer;

    // ens::BoundaryBoxConstraint b(-1.0, 1.0);
    // ens::CMAES optimizer(0, b, 1, 1e6, 1e-5, ens::FullSelection(), 0.1);

    ens::Adam optimizer(0.5, 1, 0.9, 0.999, 1e-8, 1e6, 1e-5, false, true, true);

    optimizer.Optimize(model, params, ens::Report());

    std::cout << std::endl;
    std::cout << "Optimized Params: " << params << std::endl;
    model.PrintFDNConfig(params);

    double final_loss = model.Evaluate(params);
    auto final_ir = model.GetImpulseResponse();
    audio_utils::audio_file::WriteWavFile("final_ir.wav", final_ir, kSampleRate);

    auto final_flatness = loss::SpectralFlatnessLoss(final_ir, fft);
    auto final_rms = utils::RMS(final_ir.subspan(final_ir.size() - kRMSSampleCount));

    std::cout << "Final loss: " << final_loss << std::endl;
    std::cout << "Spectral Flatness init: " << 1.0f - initial_flatness << " final: " << 1.0 - final_flatness
              << std::endl;
    std::cout << "IR RMS init: " << initial_rms << " final: " << final_rms << std::endl;

    sfFDN::FDNConfig fdn_config = model.GetFDNConfig(params);
    WriteFDNConfigToJson(fdn_config, "final_fdn_config.json");
}