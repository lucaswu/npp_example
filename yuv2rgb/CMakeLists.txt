cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(vsdnn LANGUAGES CXX CUDA)

macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()


set(CMAKE_CXX_STANDARD 11)
set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

set_ifndef(CUDA_INSTALL_DIR "/home/lucas/opt/cuda11.3")

find_library(CUDART_LIB    cudart         HINTS ${CUDA_INSTALL_DIR}       PATH_SUFFIXES lib lib64 lib/x64)
find_library(NPP_LIB    nppicc         HINTS ${CUDA_INSTALL_DIR}       PATH_SUFFIXES lib lib64 lib/x64)
message("*****(NPP_LIB)*****=${NPP_LIB}")

include_directories(${CUDA_INSTALL_DIR}/include)

set(SRC_CODE "${ROOT_PATH}/yuv2rgb.cc")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

ADD_EXECUTABLE(yuv2rgb ${SRC_CODE} )
target_link_libraries(yuv2rgb 
        ${OpenCV_LIBS}
        ${CUDART_LIB}
        ${NPP_LIB}
    )

