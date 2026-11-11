cmake_minimum_required(VERSION 3.10)
project(Project VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)



include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(libigl)
if(NOT libigl_POPULATED)
    FetchContent_Populate(libigl)
    # Enable the OpenGL/GLFW components explicitly
    set(LIBIGL_WITH_OPENGL ON CACHE BOOL "" FORCE)
    set(LIBIGL_WITH_OPENGL_GLFW ON CACHE BOOL "" FORCE)
    add_subdirectory(${libigl_SOURCE_DIR} ${libigl_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable(program garland_heckbert_surface_simplification.cpp)

target_link_libraries(program PUBLIC
    igl::core
    igl::opengl_glfw
)

include_directories(
    ${libigl_SOURCE_DIR}/include
    ${libigl_SOURCE_DIR}/external/eigen
)