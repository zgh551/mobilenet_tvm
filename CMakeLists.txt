cmake_minimum_required(VERSION 3.05)

project(mobilenet_module)

#include(cmake/FindOpenCL.cmake)
#include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake) 

#tvm_option(USE_OPENCL "Build With OpenCL" OFF)
## Set current version and minimal executable version.
set(MAJOR_VERSION 1)
set(MISOR_VERSION 0)
set(REVISION_VERSION 0)

set(SOFTWARE_VERSION ${MAJOR_VERSION}.${MISOR_VERSION}.${REVISION_VERSION})

add_definitions(-DSOFTWARE_VERSION=\"${SOFTWARE_VERSION}\")
message(STATUS "Software Version: ${SOFTWARE_VERSION}")

set(TARGET_DEVICE_TYPE Cpu CACHE STRING "the target device type")
set_property(CACHE TARGET_DEVICE_TYPE PROPERTY STRINGS Cpu OpenCL)

if(TARGET_DEVICE_TYPE MATCHES Cpu)
    add_definitions(-DCPU)
    #add_definitions(-DCPU=ON)
    #add_definitions(-DOPENCL=OFF)
    message(STATUS "Target Device Type: CPU")
elseif(TARGET_DEVICE_TYPE MATCHES OpenCL)
    #add_definitions(-DCPU=OFF)
    #add_definitions(-DOPENCL=ON)
    add_definitions(-DOPENCL)
    message(STATUS "Target Device Type: OpenCL")
endif()

find_package(OpenCV REQUIRED)

#find_opencl(USE_OPENCL)

include_directories(
    $ENV{TVM_HOME}/3rdparty/dlpack/include
    $ENV{TVM_HOME}/3rdparty/dmlc-core/include
    $ENV{TVM_HOME}/include
    ${OpenCV_INCLUDE_DIRS}
)
message(STATUS ${OpenCV_INCLUDE_DIRS})
#-Wall -Werror
add_compile_options(-std=c++14 -O2)

#set(OPENCL_LIB_DIRS /home/zgh/workspace/github/powervr_opencl/lib64)
## set the link lib location
## link the dynamic lib
if (CMAKE_SYSTEM_PROCESSOR MATCHES x64)
    message(STATUS "link the x64 lib")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/x64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES armv8)
    message(STATUS "link the armv8 lib")
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/armv8)
endif()

add_executable(mobilenet_test main.cc)

target_link_libraries(mobilenet_test tvm_runtime ${OpenCV_LIBS} pthread dl)
