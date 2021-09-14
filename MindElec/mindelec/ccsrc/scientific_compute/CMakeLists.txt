if(ENABLE_SCIENTIFIC)
    add_subdirectory(pointcloud)
endif()

set(SRC_FILES_LIST
        thread_safe_stack.cc
        )

add_library(scientific OBJECT ${SRC_FILES_LIST})

target_link_libraries(scientific PRIVATE ${pybind11_INCLUDE_DIRS})

add_dependencies(scientific scientific-pointcloud)
