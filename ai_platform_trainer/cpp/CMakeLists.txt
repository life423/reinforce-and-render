cmake_minimum_required(VERSION 3.18)

# Start the project with C++ and check for CUDA later
project(gpu_game_environment LANGUAGES CXX)

# Modern CMake CUDA detection and support
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER_VERSION}")
  # Set CUDA architecture based on capability - include RTX 5070 (sm_120)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90 "120")
  set(CUDA_FOUND TRUE)
else()
  message(STATUS "CUDA not found, using CPU-only build")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Using CUDA implementation for GPU acceleration
# Our physics.cu implementation handles physics on the GPU
option(USE_CUDA "Build with CUDA support if available" ON)
set(CUDA_ENABLED TRUE)
message(STATUS "Using CUDA-accelerated implementation for enhanced physics")

# Find PyBind11 - try multiple approaches
include(FetchContent QUIET)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
  # First try the find_package approach
  find_package(pybind11 QUIET)

  # If that didn't work, try to fetch it
  if(NOT pybind11_FOUND)
    message(STATUS "PyBind11 not found via find_package, attempting to fetch it...")
    # Try to fetch pybind11 if find_package failed
    FetchContent_Declare(
      pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.0.tar.gz
      # Remove hash verification since the tarball may have changed
    )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "PyBind11 fetched successfully")
  endif()
else()
  # For older CMake versions, just try the find_package and hope for the best
  find_package(pybind11 REQUIRED)
endif()

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
    src/physics_cpu.cpp
    src/reward.cpp
)

# Conditionally add CUDA-specific files
if(CUDA_FOUND)
    # Set preprocessor definition to enable CUDA code
    add_definitions(-DUSE_CUDA)
    set(CUDA_ENABLED TRUE)
    
    # Add CUDA-specific files
    set(CUDA_SOURCES
        src/physics.cu
    )
    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

# Pybind11 module
pybind11_add_module(gpu_environment 
    pybind/bindings.cpp
    ${SOURCES}
)

# Link libraries - with modern CMake, we don't need to explicitly specify CUDA_LIBRARIES
if(CUDA_FOUND)
    target_link_libraries(gpu_environment PRIVATE CUDA::cudart)
endif()

# Set output directory
set_target_properties(gpu_environment PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
)
