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
    ADD_SUBDIRECTORY(gpu)
endif()

