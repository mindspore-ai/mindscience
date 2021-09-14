option(DEBUG_MODE "Debug mode, default off" OFF)
option(ENABLE_D "Enable d" OFF)
option(ENABLE_SCIENTIFIC "Enable scientific computing module" ON)
option(ENABLE_GLIBCXX "enable_glibcxx" OFF)


if(NOT ENABLE_D)
    set(ENABLE_GLIBCXX ON)
endif()

if(NOT ENABLE_GLIBCXX)
    message(STATUS "Set _GLIBCXX_USE_CXX11_ABI 0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(DEBUG_MODE)
    set(CMAKE_BUILD_TYPE "Debug")
else()
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT BUILD_PATH)
    set(BUILD_PATH "${CMAKE_SOURCE_DIR}/build")
endif()
