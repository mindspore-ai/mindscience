cmake_minimum_required(VERSION 3.14.1)
project(MindElec)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(FATAL_ERROR "mindelec can not support Windows")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

# set build options for project target
include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)

# set cmake config for debug build
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer \
    -Wl,--allow-shlib-undefined -D_LIBCPP_INLINE_VISIBILITY='' -D_LIBCPP_DISABLE_EXTERN_TEMPLATE=1 \
    -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2 -Wno-cpp")

# set cmake args
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/include -std=c++17 \
    -Werror -Wall -Wno-deprecated-declarations -fPIC")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PYBIND11_CPP_STANDARD -std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION_CXX_FLAGS}")

include(${CMAKE_SOURCE_DIR}/cmake/data_expression.cmake)

find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()

message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

include_directories(${CMAKE_SOURCE_DIR})
add_subdirectory(${CMAKE_SOURCE_DIR}/mindelec/ccsrc)

# packaging tool
include(${CMAKE_SOURCE_DIR}/cmake/package.cmake)
