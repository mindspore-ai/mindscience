if(${ENABLE_D} STREQUAL "ON")
    set(backend cpu)
    ADD_SUBDIRECTORY(cpu)
endif()

if(${ENABLE_GPU} STREQUAL "ON")
    set(backend gpu)
    find_package(CUDA)
    if(NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
    endif()

    if(CUDA_VERSION VERSION_LESS "11.0")
      set(CUDA_NVCC_FLAGS -O3; -std=c++11; -arch=sm_50; -Xcompiler -fPIC; --use_fast_math)
    else()
      set(CUDA_NVCC_FLAGS -O3; -std=c++14; -arch=sm_75; -Xcompiler -fPIC; --use_fast_math)
    endif()

    message(STATUS "${CUDA_NVCC_FLAGS}")

    ADD_SUBDIRECTORY(gpu)
endif()

