#include <armadillo>
#include <ensmallen.hpp>

#include <audio_utils/audio_file_manager.h>

#include "loss.h"
#include "model.h"
#include "utils.h"

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

using namespace std::chrono_literals;

constexpr uint32_t kIrSize = 1 << 15;
constexpr uint32_t kFdnOrder = 8;

constexpr std::chrono::seconds kOptimizationDuration = 10s;
constexpr std::array kParamTypes = {ParamType::Gains};

std::mutex io_mutex;

struct ParameterTuner
{
    void Loop()
    {
        FDNModel_InputOutputGain model(kFdnOrder, kIrSize, kParamTypes);

        arma::mat params = starting_params;
        double initial_loss = model.Evaluate(params);
        best_loss = initial_loss;

        {
            std::lock_guard lock(io_mutex);
            std::cout << "Thread " << std::this_thread::get_id()
                      << " starting optimization with initial loss: " << initial_loss << std::endl;
        }

        evaluations = 0;
        auto start_time = std::chrono::steady_clock::now();
        while (start_time + kOptimizationDuration > std::chrono::steady_clock::now())
        {
            arma::mat hyper_params = arma::mat(1, 4, arma::fill::randu);

            // hyper_params(0, 2) *= 2.0; // step size [0,2]
            // hyper_params(0, 3) *= 2.0; // eval step size [0,2]

            ens::SPSA optimizer(
                /* alpha */ hyper_params(0, 0),
                /* gamma */ hyper_params(0, 1),
                /* step size */ hyper_params(0, 2),
                /* eval step size */ hyper_params(0, 3),
                /* max iterations */ 1e6,
                /* tolerance */ 1e-5);

            params = starting_params;
            double loss = optimizer.Optimize(model, params);

            if (loss < best_loss)
            {
                best_loss = loss;
                alpha = hyper_params(0, 0);
                gamma = hyper_params(0, 1);
                step_size = hyper_params(0, 2);
                eval_step_size = hyper_params(0, 3);

                // {
                //     std::lock_guard lock(io_mutex);
                //     std::cout << "Thread " << std::this_thread::get_id()
                //               << " found new best SPSA Params - alpha: " << alpha << ", gamma: " << gamma
                //               << ", step size: " << step_size << ", eval step size: " << eval_step_size
                //               << ", loss: " << best_loss << std::endl;
                // }
            }
            ++evaluations;
        }

        {
            std::lock_guard lock(io_mutex);
            std::cout << "Thread " << std::this_thread::get_id() << " best SPSA Params - alpha: " << alpha
                      << ", gamma: " << gamma << ", step size: " << step_size << ", eval step size: " << eval_step_size
                      << ", loss: " << best_loss << std::endl;
        }
    }

    arma::mat starting_params;

    double best_loss;
    double alpha;
    double gamma;
    double step_size;
    double eval_step_size;
    size_t evaluations;
};

int main()
{
    FDNModel_InputOutputGain dummy_model(kFdnOrder, 0, kParamTypes);

    arma::arma_rng::set_seed_random();
    arma::mat starting_params = dummy_model.GetInitialParams();

    std::cout << "Starting optimization for " << kOptimizationDuration.count() << " seconds..." << std::endl;
    const auto kNumThreads = std::thread::hardware_concurrency();
    std::vector<std::jthread> threads;
    std::vector<ParameterTuner> tuners;
    tuners.resize(kNumThreads);

    for (auto i = 0u; i < kNumThreads; ++i)
    {
        tuners[i].starting_params = starting_params;
        threads.emplace_back([&tuners, i]() { tuners[i].Loop(); });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    size_t total_eval_count = 0;
    double best_loss = std::numeric_limits<double>::max();
    size_t best_tuner_index = 0;
    // Find overall best result
    for (size_t i = 0; i < kNumThreads; ++i)
    {
        total_eval_count += tuners[i].evaluations;
        if (tuners[i].best_loss < best_loss)
        {
            best_loss = tuners[i].best_loss;
            best_tuner_index = i;
        }
    }

    const auto& best_tuner = tuners[best_tuner_index];
    std::cout << "Optimization complete." << std::endl;
    std::cout << "Total evaluations: " << total_eval_count << std::endl;
    std::cout << "Best SPSA Params - alpha: " << best_tuner.alpha << ", gamma: " << best_tuner.gamma
              << ", step size: " << best_tuner.step_size << ", eval step size: " << best_tuner.eval_step_size
              << ", loss: " << best_tuner.best_loss << std::endl;
}
