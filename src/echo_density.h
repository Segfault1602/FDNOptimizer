#pragma once

#include <cstdint>
#include <span>
#include <vector>

struct EchoDensityResults
{
    std::vector<float> echo_densities;
    std::vector<int> sparse_indices;
    float mixing_time;
};

EchoDensityResults EchoDensity(std::span<const float> signal, uint32_t window_size, uint32_t sample_rate,
                               uint32_t hop_size);