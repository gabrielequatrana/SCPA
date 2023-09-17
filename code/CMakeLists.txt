cmake_minimum_required(VERSION 3.20)
project(Code LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "-O3 -fopenmp")
    set(CMAKE_CXX_FLAGS "-O3 -fopenmp")
endif()

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    set(CMAKE_CUDA_FLAGS "-O3")
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
endif()

include_directories(lib/mmio
        logic/CPU
        logic/GPU
        utility
        utility/GPU
        utility/matrix/CSR
        utility/matrix/ELL
        kernel)

add_executable(spmmCPU
        main.c
        lib/mmio/mmio.c
        lib/mmio/mmio.h
        utility/matrix/ELL/ELL.h
        utility/matrix/CSR/CSR.h
        utility/matrix/CSR/CSR.c
        utility/util.h
        utility/util.c
        utility/matrix/ELL/ELL.c
        logic/CPU/logicCPU.h
        logic/CPU/logicCPU.c)

target_link_libraries(spmmCPU OpenMP::OpenMP_C)

add_executable(spmmGPU
        main.cu
        lib/mmio/mmio.c
        lib/mmio/mmio.h
        utility/matrix/ELL/ELL.h
        utility/matrix/CSR/CSR.h
        utility/matrix/CSR/CSR.c
        utility/util.h
        utility/util.c
        utility/matrix/ELL/ELL.c
        logic/CPU/logicCPU.c
        logic/CPU/logicCPU.h
        logic/GPU/logicGPU.cuh
        logic/GPU/logicGPU.cu
        utility/GPU/utilGPU.cpp
        utility/GPU/utilGPU.h
        kernel/kernel.cu
        kernel/kernel.cuh)

target_link_libraries(spmmGPU ${CUDA_LIBRARIES})
set_target_properties(spmmGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)