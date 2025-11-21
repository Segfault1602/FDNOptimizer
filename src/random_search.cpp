#include <armadillo>
#include <audio_utils/audio_file_manager.h>

#include "loss.h"
#include "model.h"
#include "utils.h"

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kIrSize = 1 << 15;
constexpr uint32_t kFdnOrder = 8;
constexpr uint32_t kRMSSampleCount = 1024;

constexpr std::chrono::seconds kOptimizationDuration = 10s;
constexpr std::array kParamTypes = {ParamType::Gains};

void RandomSearch(arma::mat& params, size_t& eval_count, double& best_loss);

int main()
{

    const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(kIrSize);
    audio_utils::FFT fft(fft_size);

    FDNModel_InputOutputGain model(kFdnOrder, kIrSize, kParamTypes);

    arma::arma_rng::set_seed_random();
    arma::mat params = model.GetInitialParams();

    std::cout << "Params: " << params << std::endl;
    model.PrintFDNConfig(params);

    auto initial_loss = model.Evaluate(params);
    std::cout << "Initial Loss: " << initial_loss << std::endl;
    auto initial_ir = model.GetImpulseResponse();
    audio_utils::audio_file::WriteWavFile("initial_ir.wav", initial_ir, kSampleRate);

    auto initial_flatness = loss::SpectralFlatnessLoss(initial_ir, fft);
    auto initial_rms = utils::RMS(initial_ir.subspan(initial_ir.size() - kRMSSampleCount));

    arma::mat best_params = params;
    double best_loss = initial_loss;
    size_t eval_count = 0;

    std::vector<arma::mat> best_params_vector;
    std::vector<double> best_loss_vector;
    std::vector<size_t> eval_count_vector;

    const size_t kNumThreads = std::thread::hardware_concurrency();
    best_params_vector.resize(kNumThreads, best_params);
    best_loss_vector.resize(kNumThreads, best_loss);
    eval_count_vector.resize(kNumThreads, 0);

    std::cout << "Starting optimization for " << kOptimizationDuration.count() << " seconds..." << std::endl;
    std::vector<std::jthread> threads;

    for (size_t i = 0; i < kNumThreads; ++i)
    {
        threads.emplace_back(RandomSearch, std::ref(best_params_vector[i]), std::ref(eval_count_vector[i]),
                             std::ref(best_loss_vector[i]));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    // Find overall best result
    for (size_t i = 0; i < kNumThreads; ++i)
    {
        eval_count += eval_count_vector[i];
        if (best_loss_vector[i] < best_loss)
        {
            best_loss = best_loss_vector[i];
            best_params = best_params_vector[i];
        }
    }

    std::cout << "Optimization complete." << std::endl;
    std::cout << "Total evaluations: " << eval_count << std::endl;

    std::cout << std::endl;
    std::cout << "Optimized Params: " << best_params << std::endl;
    std::cout << "Best Loss: " << best_loss << std::endl;
    model.PrintFDNConfig(best_params);

    model.Evaluate(best_params);
    auto final_ir = model.GetImpulseResponse();
    audio_utils::audio_file::WriteWavFile("final_ir.wav", final_ir, kSampleRate);

    auto final_flatness = loss::SpectralFlatnessLoss(final_ir, fft);
    auto final_rms = utils::RMS(final_ir.subspan(final_ir.size() - kRMSSampleCount));

    std::cout << "Spectral Flatness went from " << 1.0f - initial_flatness << " to " << 1.0f - final_flatness
              << std::endl;
    std::cout << "RMS went from " << initial_rms << " to " << final_rms << std::endl;
}

void RandomSearch(arma::mat& params, size_t& eval_count, double& best_loss)
{
    const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(kIrSize);
    audio_utils::FFT fft(fft_size);

    FDNModel_InputOutputGain model(kFdnOrder, kIrSize, kParamTypes);

    auto start_time = std::chrono::steady_clock::now();

    arma::mat best_params = params;

    while (start_time + kOptimizationDuration > std::chrono::steady_clock::now())
    {
        params = arma::mat(1, model.GetParamCount(), arma::fill::randn);

        auto loss = model.Evaluate(params);
        if (loss < best_loss)
        {
            best_loss = loss;
            best_params = params;
        }
        ++eval_count;
    }

    params = best_params;
}