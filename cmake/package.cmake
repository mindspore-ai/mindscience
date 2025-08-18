# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_GENERATOR "External")
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/mindscience)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/mindscience)

if(ENABLE_D)
    set(CPACK_MS_PACKAGE_NAME "mindscience")
elseif(ENABLE_GPU)
    set(CPACK_MS_PACKAGE_NAME "mindscience_gpu")
else()
    set(CPACK_MS_PACKAGE_NAME "mindscience")
endif()
include(CPack)

set(INSTALL_BASE_DIR ".")
set(INSTALL_PY_DIR ".")

# copy python files
install(
        FILES
        ${CMAKE_SOURCE_DIR}/mindscience/__init__.py
            ${CMAKE_SOURCE_DIR}/setup.py
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindscience
)

install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindscience/ccsrc
        ${CMAKE_SOURCE_DIR}/mindscience/common
        ${CMAKE_SOURCE_DIR}/mindscience/data
        ${CMAKE_SOURCE_DIR}/mindscience/distributed
        ${CMAKE_SOURCE_DIR}/mindscience/e3nn
        ${CMAKE_SOURCE_DIR}/mindscience/gnn
        ${CMAKE_SOURCE_DIR}/mindscience/models
        ${CMAKE_SOURCE_DIR}/mindscience/pde
        ${CMAKE_SOURCE_DIR}/mindscience/sciops
        ${CMAKE_SOURCE_DIR}/mindscience/solvers
        ${CMAKE_SOURCE_DIR}/mindscience/utils
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindscience
)


