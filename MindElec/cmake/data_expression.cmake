set(SECURE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

include(cmake/utils.cmake)
find_package(Python3)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/pybind11.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/glog.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/external_libs/json.cmake)

message("========== External libs built successfully ==========")

include_directories(${Python3_INCLUDE_DIRS})
