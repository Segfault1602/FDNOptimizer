#include "echo_density.h"

#include <Eigen/Core>
#include <audio_utils/fft_utils.h>

#include <cassert>
#include <cmath>
#include <numbers>
#include <numeric>
#include <span>
#include <vector>

EchoDensityResults EchoDensity(std::span<const float> signal, uint32_t window_size, uint32_t sample_rate,
                               uint32_t hop_size)
{
    if (signal.empty() || window_size == 0 || sample_rate == 0)
    {
        return {};
    }

    EchoDensityResults results;

    std::vector<float> win(window_size, 0.0f);
    GetWindow(audio_utils::FFTWindowType::Hann, win);
    float win_sum = std::accumulate(win.begin(), win.end(), 0.0f);
    for (auto& w : win)
    {
        w /= win_sum;
    }

    const int half_win = window_size / 2;
    results.echo_densities.reserve((signal.size() + hop_size - 1) / hop_size);
    results.sparse_indices.reserve((signal.size() + hop_size - 1) / hop_size);

    results.mixing_time = std::numeric_limits<float>::infinity();

    for (int n = 0; n < signal.size(); n += hop_size)
    {
        std::span<const float> hTau;
        std::span<const float> wT;

        if (n < half_win)
        {
            hTau = signal.subspan(0, n + half_win);
            wT = std::span(win).subspan(win.size() - n - half_win, n + half_win);
        }
        else if (n > signal.size() - half_win)
        {
            hTau = signal.subspan(n - half_win, signal.size() - n + half_win);
            wT = std::span(win).subspan(0, signal.size() - n + half_win);
        }
        else
        {
            hTau = signal.subspan(n - half_win, window_size);
            wT = win;
        }

        assert(hTau.size() == wT.size());

        Eigen::Map<const Eigen::ArrayXf> hTau_map(hTau.data(), hTau.size());
        Eigen::Map<const Eigen::ArrayXf> wT_map(wT.data(), wT.size());

        float std = std::sqrt((hTau_map.square() * wT_map).sum());

        // Use Eigen for vectorized computation of echo_density
        Eigen::ArrayXf abs_hTau = hTau_map.abs();
        Eigen::ArrayXf mask = (abs_hTau > std).cast<float>();
        float echo_density = (mask * wT_map).sum();

        // normalize
        const float kErfc = std::erfc(1.0f / std::numbers::sqrt2_v<float>);
        echo_density /= kErfc;

        results.sparse_indices.push_back(n);
        results.echo_densities.push_back(echo_density);

        // Estimate mixing time as the time when echo density first exceeds 0.9
        if (results.mixing_time == std::numeric_limits<float>::infinity() && echo_density >= 0.9f)
        {
            results.mixing_time = static_cast<float>(n) / static_cast<float>(sample_rate);
        }
    }

    return results;
}