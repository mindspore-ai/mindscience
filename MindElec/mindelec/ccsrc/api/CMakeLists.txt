add_library(APItoPython OBJECT
        python/pybind_conversion.cc
        python/pybind_register.cc
        python/bindings/scientific_compute/pointcloud/bindings.cc
        )
target_include_directories(APItoPython PRIVATE ${pybind11_INCLUDE_DIRS})