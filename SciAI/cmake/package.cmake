# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_GENERATOR "External")
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/sciai)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/sciai)

set(CPACK_MS_PACKAGE_NAME "sciai")
include(CPack)

set(INSTALL_BASE_DIR ".")
set(INSTALL_PY_DIR ".")

# copy python files
install(
        FILES
            ${CMAKE_SOURCE_DIR}/sciai/__init__.py
            ${CMAKE_SOURCE_DIR}/sciai/version.py
            ${CMAKE_SOURCE_DIR}/setup.py
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT sciai
)

install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/sciai/architecture
        ${CMAKE_SOURCE_DIR}/sciai/common
        ${CMAKE_SOURCE_DIR}/sciai/context
        ${CMAKE_SOURCE_DIR}/sciai/utils
        ${CMAKE_SOURCE_DIR}/sciai/operators
        ${CMAKE_SOURCE_DIR}/sciai/model
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT sciai
)

