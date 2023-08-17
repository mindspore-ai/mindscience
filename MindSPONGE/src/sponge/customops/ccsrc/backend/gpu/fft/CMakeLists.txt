file(GLOB_RECURSE CURRENT_HEADERS *.cuh)
file(GLOB CURRENT_SOURCES *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

message("File is ${CURRENT_SOURCES}")
cuda_add_library(fft SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
CUDA_ADD_CUFFT_TO_TARGET(fft)
