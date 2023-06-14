# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_GENERATOR "External")
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/mindchemistry)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/mindchemistry)

if(ENABLE_D)
    set(CPACK_MS_PACKAGE_NAME "mindchemistry_ascend")
elseif(ENABLE_GPU)
    set(CPACK_MS_PACKAGE_NAME "mindchemistry_gpu")
else()
    set(CPACK_MS_PACKAGE_NAME "mindchemistry_ascend")
endif()
include(CPack)

set(INSTALL_BASE_DIR ".")
set(INSTALL_PY_DIR ".")

# copy python files
install(
        FILES
            ${CMAKE_SOURCE_DIR}/mindchemistry/__init__.py
            ${CMAKE_SOURCE_DIR}/setup.py
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindchemistry
)

install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindchemistry/cell
        ${CMAKE_SOURCE_DIR}/mindchemistry/e3
        ${CMAKE_SOURCE_DIR}/mindchemistry/utils
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindchemistry
)

