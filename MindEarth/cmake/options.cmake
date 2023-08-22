option(ENABLE_D "Enable d" OFF)
option(ENABLE_SCIENTIFIC "Enable scientific computing module" ON)

if(NOT BUILD_PATH)
    set(BUILD_PATH "${CMAKE_SOURCE_DIR}/build")
endif()
