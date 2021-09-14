set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib")

if(ENABLE_SCIENTIFIC)
    add_definitions(-D ENABLE_SCIENTIFIC)
    message(STATUS "Scientific module is enabled")
endif()

########### Set up the include directories ###########################
include_directories(${CMAKE_SOURCE_DIR}/mindelec/ccsrc/include)

######################################################################

################## Include sub-modules ###############################
add_subdirectory(api)
add_subdirectory(core)
if(ENABLE_SCIENTIFIC)
    add_subdirectory(scientific_compute)
endif()
######################################################################
# Rely part


################### Create _c_minddata Library ######################
set(submodules
        $<TARGET_OBJECTS:APItoPython>
        $<TARGET_OBJECTS:core>)

if(ENABLE_SCIENTIFIC)
    set(submodules
            ${submodules}
            $<TARGET_OBJECTS:scientific>
            $<TARGET_OBJECTS:scientific-pointcloud>)
endif()

add_library(_c_minddata SHARED ${submodules})

set_target_properties(_c_minddata PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        )
######################################################################

################# Link with external libraries ########################
target_link_libraries(_c_minddata PRIVATE minddata::pybind11_module -ldl ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY})

target_link_libraries(_c_minddata PRIVATE minddata::json)

target_link_libraries(_c_minddata PRIVATE minddata::glog)
target_link_options(_c_minddata PRIVATE -Wl,-init,minddata_log_init)
