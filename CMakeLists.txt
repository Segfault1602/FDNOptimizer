cmake_minimum_required(VERSION 4.1.1)
set(CMAKE_CXX_STANDARD 20)

project(FDNOptimizer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 26.0)
endif()

set(AUDIO_UTILS_USE_SNDFILE ON)
set(AUDIO_UTILS_USE_RTAUDIO OFF)
add_subdirectory(externals/audio_utils)

set(SFFDN_BUILD_TESTS OFF)
add_subdirectory(externals/sffdn)

include(FetchContent)

# FetchContent_Declare( armadillo GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git GIT_TAG 15.2.x )
# FetchContent_MakeAvailable(armadillo) find_package(openblas CONFIG REQUIRED)
find_package(Armadillo CONFIG REQUIRED)
add_library(Armadillo::Armadillo ALIAS armadillo)

FetchContent_Declare(ensmallen GIT_REPOSITORY https://github.com/mlpack/ensmallen.git GIT_TAG master)
set(USE_BANDICOOT OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ensmallen)

find_package(nanobench CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_subdirectory(src)
