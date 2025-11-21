#include <ensmallen.hpp>
#include <nanobench.h>

#include <audio_utils/audio_analysis.h>
#include <audio_utils/audio_file_manager.h>
#include <audio_utils/fft.h>
#include <sffdn/sffdn.h>

#include "utils.h"

#include <chrono>
#include <iostream>

using namespace ankerl;
using namespace std::chrono_literals;

class SquaredFunction
{
  public:
    // This returns f(x) = 2 |x|^2.
    double Evaluate(const arma::mat& x)
    {
        return 2 * std::pow(arma::norm(x), 2.0);
    }
};

int main()
{
    // The minimum is at x = [0 0 0].  Our initial point is chosen to be
    // [1.0, -1.0, 1.0].
    arma::mat x("1.0 -1.0 1.0");

    // Create simulated annealing optimizer with default options.
    // The ens::SA<> type can be replaced with any suitable ensmallen optimizer
    // that is able to handle arbitrary functions.
    ens::SA<> optimizer;
    SquaredFunction f; // Create function to be optimized.
    optimizer.Optimize(f, x);

    std::cout << "Minimum of squared function found with simulated annealing is " << x;

    constexpr uint32_t kIrSize = 1 << 15;
    constexpr uint32_t kFdnOrder = 8;
    arma::mat params(1, kFdnOrder * 2, arma::fill::randn);
    std::cout << "Params: " << params << std::endl;

    float flatness = 0.0f;
}