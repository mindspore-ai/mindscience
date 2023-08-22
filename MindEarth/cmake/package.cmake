# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_GENERATOR "External")
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/mindearth)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/mindearth)

if(ENABLE_D)
    set(CPACK_MS_PACKAGE_NAME "mindearth_ascend")
elseif(ENABLE_GPU)
    set(CPACK_MS_PACKAGE_NAME "mindearth_gpu")
else()
    set(CPACK_MS_PACKAGE_NAME "mindearth_ascend")
endif()
include(CPack)

set(INSTALL_BASE_DIR ".")
set(INSTALL_PY_DIR ".")

# copy python files
install(
        FILES
            ${CMAKE_SOURCE_DIR}/mindearth/__init__.py
            ${CMAKE_SOURCE_DIR}/setup.py
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindearth
)

install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindearth/cell
        ${CMAKE_SOURCE_DIR}/mindearth/core
        ${CMAKE_SOURCE_DIR}/mindearth/data
        ${CMAKE_SOURCE_DIR}/mindearth/module
        ${CMAKE_SOURCE_DIR}/mindearth/utils
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindearth
)

