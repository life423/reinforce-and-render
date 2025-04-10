cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpu_game_environment LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA configuration
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=sm_70;--expt-relaxed-constexpr)

# Find PyBind11
find_package(pybind11 REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${pybind11_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/entity.cpp
    src/environment.cpp
    src/physics.cu
    src/reward.cpp
)

# Pybind11 module
pybind11_add_module(gpu_environment 
    pybind/bindings.cpp
    ${SOURCES}
)

# Link libraries
target_link_libraries(gpu_environment PRIVATE ${CUDA_LIBRARIES})

# Set output directory
set_target_properties(gpu_environment PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
)
