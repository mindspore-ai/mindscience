include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindelec/ccsrc/include)

file(GLOB_RECURSE CORE_SRC_LIST
        "utils/*.cc")

add_library(core OBJECT ${CORE_SRC_LIST})
target_link_libraries(core PRIVATE minddata::glog)
