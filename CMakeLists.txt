cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0135 NEW)
set(BUILD_SHARED_LIBS OFF)
set (CMAKE_CXX_STANDARD 23)
add_compile_options(-fimplicit-constexpr -march=native)

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")

if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif(CCACHE_FOUND)



project(sfml_rt
    VERSION 0.1
    LANGUAGES CXX)

add_subdirectory(dependencies)
add_subdirectory(src)

