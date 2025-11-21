find_package(OpenMP REQUIRED)

add_library(
    fdn_optimizer STATIC
    echo_density.cpp
    loss.cpp
    model.cpp
    utils.cpp
)

target_link_libraries(
    fdn_optimizer
    PUBLIC Armadillo::Armadillo
           ensmallen
           audio_utils::audio_utils
           sfFDN::sfFDN
           OpenMP::OpenMP_CXX
           nlohmann_json::nlohmann_json
           Eigen3::Eigen
)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE fdn_optimizer)

add_executable(perf main.cpp)
target_link_libraries(perf PRIVATE fdn_optimizer)

add_executable(rng_search random_search.cpp)
target_link_libraries(rng_search PRIVATE fdn_optimizer)

add_executable(spsa_tuning spsa_tuning.cpp)
target_link_libraries(spsa_tuning PRIVATE fdn_optimizer)
