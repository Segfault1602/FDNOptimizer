find_package(OpenMP REQUIRED)

add_executable(
    main
    loss.cpp
    main.cpp
    model.cpp
    utils.cpp
)
target_link_libraries(
    main
    PRIVATE Armadillo::Armadillo
            ensmallen
            audio_utils::audio_utils
            sfFDN::sfFDN
            OpenMP::OpenMP_CXX
)

add_executable(
    perf
    loss.cpp
    main.cpp
    model.cpp
    utils.cpp
)
target_link_libraries(
    perf
    PRIVATE Armadillo::Armadillo
            ensmallen
            audio_utils::audio_utils
            sfFDN::sfFDN
            nanobench::nanobench
            OpenMP::OpenMP_CXX
)

add_executable(
    rng_search
    loss.cpp
    model.cpp
    random_search.cpp
    utils.cpp
)
target_link_libraries(
    rng_search
    PRIVATE Armadillo::Armadillo
            audio_utils::audio_utils
            sfFDN::sfFDN
            OpenMP::OpenMP_CXX
)

add_executable(
    spsa_tuning
    loss.cpp
    model.cpp
    spsa_tuning.cpp
    utils.cpp
)
target_link_libraries(
    spsa_tuning
    PRIVATE Armadillo::Armadillo
            ensmallen
            audio_utils::audio_utils
            sfFDN::sfFDN
            OpenMP::OpenMP_CXX
)
