# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_GENERATOR "External")
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/mindelec)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/mindelec)

if(ENABLE_D)
    set(CPACK_MS_PACKAGE_NAME "mindelec_ascend")
elseif(ENABLE_GPU)
    set(CPACK_MS_PACKAGE_NAME "mindelec_gpu")
else()
    set(CPACK_MS_PACKAGE_NAME "mindelec_ascend")
endif()
include(CPack)

set(INSTALL_BASE_DIR ".")
set(INSTALL_PY_DIR ".")
set(CMAKE_INSTALL_LIBDIR "./lib")

#install lib_minddata.so
install(
    TARGETS _c_minddata
    DESTINATION ${INSTALL_BASE_DIR}
    COMPONENT mindelec
)

# install lib_glog.so
file(GLOB GLOG_LIB_LIST "${glog_LIBPATH}/libglog.so.0.4.0")

install(
    FILES ${GLOG_LIB_LIST}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RENAME libglog.so.0
    COMPONENT mindelec
)

message("===============${CMAKE_INSTALL_LIBDIR}+++++++++++++++")

# copy python files
install(
        FILES
            ${CMAKE_SOURCE_DIR}/mindelec/__init__.py
            ${CMAKE_SOURCE_DIR}/setup.py
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindelec
)

install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindelec/architecture
        ${CMAKE_SOURCE_DIR}/mindelec/common
        ${CMAKE_SOURCE_DIR}/mindelec/data
        ${CMAKE_SOURCE_DIR}/mindelec/geometry
        ${CMAKE_SOURCE_DIR}/mindelec/loss
        ${CMAKE_SOURCE_DIR}/mindelec/operators
        ${CMAKE_SOURCE_DIR}/mindelec/solver
        ${CMAKE_SOURCE_DIR}/mindelec/vision

    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindelec
)

# copy library to source dir
install(
    TARGETS _c_minddata
    DESTINATION ${CMAKE_SOURCE_DIR}/mindelec
    COMPONENT mindelec
)

install(
    FILES ${GLOG_LIB_LIST}
    DESTINATION ${CMAKE_SOURCE_DIR}/mindelec/lib
    RENAME libglog.so.0
    COMPONENT mindelec
)
