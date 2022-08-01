cmake_minimum_required(VERSION 3.10.2)
project(MindSPONGE)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(FATAL_ERROR "MindSPONGE can not support Windows")
endif()

option(ENABLE_D "Enable d" OFF)
option(ENABLE_GPU "Enable gpu" OFF)
option(ENABLE_MD "Enable MD" OFF)

message("========ARGS==========")
message("BACKEND GPU    : " ${ENABLE_GPU})
message("BACKEND ASCEND : " ${ENABLE_D})
message("ENABLE  MD     : " ${ENABLE_MD})
message("========ARGS==========")
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/mindsponge/libs)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/mindsponge/bin)
message("CMAKE_SOURCE_DIR is ${CMAKE_SOURCE_DIR}")

# ADD_SUBDIRECTORY(mindsponge/ccsrc/backend/)
if(ENABLE_MD MATCHES "ON")
ADD_SUBDIRECTORY(mindsponge/ccsrc/molecular_dynamics/)
endif()