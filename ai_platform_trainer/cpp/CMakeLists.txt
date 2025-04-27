cmake_minimum_required(VERSION 3.18)

# Start the project with C++ and check for CUDA later
project(gpu_game_environment LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python first since we'll need it for architecture detection
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Modern CMake CUDA detection and support
include(CheckLanguage)
check_language(CUDA)

# CUDA configuration
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER_VERSION}")
  
  # Detect supported CUDA architectures from current system
  message(STATUS "Detecting supported CUDA architectures...")
  execute_process(
    COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/detect_gpu_arch.py"
    OUTPUT_VARIABLE DETECTED_ARCHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE DETECT_RESULT
  )
  
  if(DETECT_RESULT EQUAL 0 AND DETECTED_ARCHS)
    # Successfully detected architectures
    string(REPLACE ";" " " ARCH_LIST "${DETECTED_ARCHS}")
    message(STATUS "Using detected CUDA architectures: ${ARCH_LIST}")
    set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCHS})
  else()
    # Fall back to defaults if detection failed
    message(STATUS "Architecture detection failed, using defaults")
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
  endif()
  
  # Find CUDA package - this gives us access to CUDA libraries
  find_package(CUDA REQUIRED)
  
  # Debug info
  message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
  message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
  message(STATUS "CUDA_CUDART_LIBRARY: ${CUDA_CUDART_LIBRARY}")
  
  # Set variables for use later
  set(CUDA_FOUND TRUE)
  set(CUDA_ENABLED TRUE)
  
  # Set preprocessor definition to enable CUDA code
  add_definitions(-DUSE_CUDA)
  
  message(STATUS "Using CUDA-accelerated implementation for enhanced physics")
else()
  message(STATUS "CUDA not found, using CPU-only build")
  set(CUDA_FOUND FALSE)
  set(CUDA_ENABLED FALSE)
endif()

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

# Python is already found above

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${pybind11_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Add CUDA includes if available
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Source files
set(SOURCES
    src/entity.cpp
    src/environment.cpp
    src/physics_cpu.cpp
    src/reward.cpp
)

# Conditionally add CUDA-specific files
if(CUDA_FOUND)
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

# Link libraries - explicitly using CUDA libraries
if(CUDA_FOUND)
    # For Visual Studio / Windows, we need to link directly
    if(MSVC)
        if(CUDA_CUDART_LIBRARY)
            target_link_libraries(gpu_environment PRIVATE ${CUDA_CUDART_LIBRARY})
        else()
            # Fallback for older CUDA versions
            target_link_libraries(gpu_environment PRIVATE ${CUDA_LIBRARIES})
        endif()
    else()
        # Linux / macOS linking approach
        target_link_libraries(gpu_environment PRIVATE ${CUDA_LIBRARIES})
    endif()
endif()

# Set output directory
set_target_properties(gpu_environment PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
)
