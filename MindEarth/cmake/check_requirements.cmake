function(find_required_package pkg_name)
    find_package(${pkg_name})
    if(NOT ${pkg_name}_FOUND)
        message(FATAL_ERROR "Required package ${pkg_name} not found, "
                "please install the package and try building MindEarth again.")
    endif()
endfunction()

## find python, quit if the found python is static
set(Python3_USE_STATIC_LIBS FALSE)
set(Python3_FIND_VIRTUALENV ONLY)
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    message("Python3 found, version: ${Python3_VERSION}")
    message("Python3 library path: ${Python3_LIBRARY}")
    message("Python3 interpreter: ${Python3_EXECUTABLE}")
elseif(Python3_LIBRARY AND Python3_EXECUTABLE AND
        ${Python3_VERSION} VERSION_GREATER_EQUAL "3.7.0" AND ${Python3_VERSION} VERSION_LESS "3.9.9")
    message(WARNING "Maybe python3 environment is broken.")
    message("Python3 library path: ${Python3_LIBRARY}")
    message("Python3 interpreter: ${Python3_EXECUTABLE}")
else()
    message(FATAL_ERROR "Python3 not found, please install Python>=3.7.5, and set --enable-shared "
            "if you are building Python locally")
endif()
