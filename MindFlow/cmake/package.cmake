# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_GENERATOR "External")
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/mindflow)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/mindflow)

if(ENABLE_D)
    set(CPACK_MS_PACKAGE_NAME "mindflow_ascend")
elseif(ENABLE_GPU)
    set(CPACK_MS_PACKAGE_NAME "mindflow_gpu")
else()
    set(CPACK_MS_PACKAGE_NAME "mindflow_ascend")
endif()
include(CPack)

set(INSTALL_BASE_DIR ".")
set(INSTALL_PY_DIR ".")

# copy python files
install(
        FILES
            ${CMAKE_SOURCE_DIR}/mindflow/__init__.py
            ${CMAKE_SOURCE_DIR}/setup.py
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindflow
)

install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindflow/cell
        ${CMAKE_SOURCE_DIR}/mindflow/cfd
        ${CMAKE_SOURCE_DIR}/mindflow/common
        ${CMAKE_SOURCE_DIR}/mindflow/data
        ${CMAKE_SOURCE_DIR}/mindflow/geometry
        ${CMAKE_SOURCE_DIR}/mindflow/loss
        ${CMAKE_SOURCE_DIR}/mindflow/operators
        ${CMAKE_SOURCE_DIR}/mindflow/pde
        ${CMAKE_SOURCE_DIR}/mindflow/utils
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindflow
)

