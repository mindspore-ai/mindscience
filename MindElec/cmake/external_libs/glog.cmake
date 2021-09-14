set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 ${SECURE_CXX_FLAGS}")
set(glog_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if(NOT ENABLE_GLIBCXX)
    set(glog_CXXFLAGS "${glog_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/glog/repository/archive/v0.4.0.tar.gz")
    set(MD5 "22fe340ddc231e6c8e46bc295320f8ee")
else()
    set(REQ_URL "https://github.com/google/glog/archive/v0.4.0.tar.gz")
    set(MD5 "0daea8785e6df922d7887755c3d100d0")
endif()

set(glog_lib glog)

set(glog_option -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -DWITH_GFLAGS=OFF)

mindspore_add_pkg(glog
        VER 0.4.0
        LIBS ${glog_lib}
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION ${glog_option})
include_directories(${glog_INC})
find_package(glog 0.4.0 REQUIRED)
add_library(minddata::glog ALIAS glog::${glog_lib})
