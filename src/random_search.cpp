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
constexpr uint32_t kParamCount = kFdnOrder * 3;

constexpr std::chrono::seconds kOptimizationDuration = 10s;

void RandomSearch(arma::mat& params, size_t& eval_count, double& best_loss);

int main()
{
    arma::arma_rng::set_seed_random();
    arma::mat params(1, kParamCount, arma::fill::randn);

    // params /= arma::norm(params);

    std::cout << "Params: " << params << std::endl;
    utils::PrintParams(params, kFdnOrder);

    const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(kIrSize);
    audio_utils::FFT fft(fft_size);

    auto loss_function = [&fft](std::span<const float> signal) -> float {
        return loss::SpectralFlatnessLoss(signal, fft) +
               0.01 * loss::RMSLoss(signal.subspan(signal.size() - 1024), 0.1f);
    };

    FDNModel model(kFdnOrder, kIrSize, loss_function);

    auto initial_loss = model.Evaluate(params);
    std::cout << "Initial Loss: " << initial_loss << std::endl;
    auto initial_ir = model.GetImpulseResponse();
    audio_utils::audio_file::WriteWavFile("initial_ir.wav", initial_ir, kSampleRate);

    auto initial_flatness = loss::SpectralFlatnessLoss(initial_ir, fft);
    auto initial_rms = loss::RMSLoss(initial_ir.subspan(initial_ir.size() - 1024), 0.1f);

    auto start_time = std::chrono::steady_clock::now();

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
    utils::PrintParams(best_params, kFdnOrder);

    model.Evaluate(best_params);
    auto final_ir = model.GetImpulseResponse();
    audio_utils::audio_file::WriteWavFile("final_ir.wav", final_ir, kSampleRate);

    auto final_flatness = loss::SpectralFlatnessLoss(final_ir, fft);
    auto final_rms = loss::RMSLoss(final_ir.subspan(final_ir.size() - 1024), 0.1f);

    std::cout << "Spectral Flatness went from " << 1.0f - initial_flatness << " to " << 1.0f - final_flatness
              << std::endl;
    std::cout << "RMS went from " << initial_rms << " to " << final_rms << std::endl;
}

void RandomSearch(arma::mat& params, size_t& eval_count, double& best_loss)
{
    const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(kIrSize);
    audio_utils::FFT fft(fft_size);

    auto loss_function = [&fft](std::span<const float> signal) -> float {
        return loss::SpectralFlatnessLoss(signal, fft) +
               0.01 * loss::RMSLoss(signal.subspan(signal.size() - 1024), 0.1f);
    };

    FDNModel model(kFdnOrder, kIrSize, loss_function);

    auto initial_loss = model.Evaluate(params);
    auto start_time = std::chrono::steady_clock::now();

    arma::mat best_params = params;

    while (start_time + kOptimizationDuration > std::chrono::steady_clock::now())
    {
        params = arma::mat(1, kParamCount, arma::fill::randn);
        // params = arma::mat(1, kFdnOrder * 2, arma::fill::randu) * 2.0 - 1.0;

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