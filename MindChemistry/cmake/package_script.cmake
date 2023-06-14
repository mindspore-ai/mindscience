# find exec
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "No python3 found.")
endif()

set(PYTHON ${Python3_EXECUTABLE})
set(PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

if(NOT (PYTHON_VERSION MATCHES "3.9" OR PYTHON_VERSION MATCHES "3.8" OR PYTHON_VERSION MATCHES "3.7"))
    message(FATAL_ERROR "FIND PYTHON VERSION ${PYTHON_VERSION} BUT CAN NOT MATCH PYTHON VERSION 3.9 OR 3.8 OR 3.7")
endif()

set(ENV{ME_PACKAGE_NAME} ${CPACK_MS_PACKAGE_NAME})
message("start executing setup.py to prepare whl file")

# following block embeds a short msg in the whl package
find_package(Git)
if(NOT GIT_FOUND)
    message("No git found.")
    return()
endif()
set(GIT ${GIT_EXECUTABLE})
set(GIT_COMMIT_ID "")
set(BUILD_DATE_TIME "")
execute_process(
        COMMAND ${GIT} log --format='[sha1]:%h,[branch]:%d' --abbrev=8 -1
        OUTPUT_VARIABLE GIT_COMMIT_ID
        WORKING_DIRECTORY ${CPACK_PACKAGE_DIRECTORY}/../package/mindelec
        ERROR_QUIET)

# set path
set(MS_ROOT_DIR ${CPACK_PACKAGE_DIRECTORY}/../../)
set(MS_PACK_ROOT_DIR ${MS_ROOT_DIR}/build/package)

# set package file name
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(PYTHON_VERSION MATCHES "3.9")
        set(PY_TAGS "cp39-cp39")
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(PY_TAGS "cp38-cp38")
    elseif(PYTHON_VERSION MATCHES "3.7")
        set(PY_TAGS "cp37-cp37m")
    else()
        message("Could not find 'Python 3.9' OR 'Python 3.8' or 'Python 3.7'")
        return()
    endif()
    string(TOLOWER linux_${CMAKE_HOST_SYSTEM_PROCESSOR} PLATFORM_TAG)
else()
    set(PLATFORM_TAG "any")
endif()

# get the current timestamp to be embedded in the built info
string(TIMESTAMP BUILD_DATE_TIME "whl generated on:%Y-%m-%d %H:%M:%S")
file(WRITE ${CPACK_PACKAGE_DIRECTORY}/../package/mindchemistry/build_info.txt "${BUILD_DATE_TIME}\n${GIT_COMMIT_ID}")

# above block embeds a short msg containing commit id and date time in the whl package
execute_process(
        COMMAND ${PYTHON} ${CPACK_PACKAGE_DIRECTORY}/../../setup.py "bdist_wheel"
        WORKING_DIRECTORY ${CPACK_PACKAGE_DIRECTORY}/../package
)

set(PACKAGE_NAME ${CPACK_MS_PACKAGE_NAME})
file(GLOB WHL_FILE ${MS_PACK_ROOT_DIR}/dist/*.whl)
get_filename_component(ORIGIN_FILE_NAME ${WHL_FILE} NAME)
string(REPLACE "-" ";" ORIGIN_FILE_NAME ${ORIGIN_FILE_NAME})
list(GET ORIGIN_FILE_NAME 1 VERSION)
set(NEW_FILE_NAME ${PACKAGE_NAME}-${VERSION}-${PY_TAGS}-${PLATFORM_TAG}.whl)
file(RENAME ${WHL_FILE} ${MS_PACK_ROOT_DIR}/${NEW_FILE_NAME})
