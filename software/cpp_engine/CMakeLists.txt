cmake_minimum_required(VERSION 3.10)
project(vision_engine)

# Find packages
find_package(OpenCV REQUIRED)

# PyBind11 can be found in different ways, we'll try the python module first
execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# If the command failed, fallback to find_package
if (NOT PYBIND11_INCLUDES)
    find_package(pybind11 REQUIRED)
endif()

# We might also need the python executable and include paths
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# If PYBIND11_INCLUDES contains string like "-I/path -I/other/path"
string(REPLACE "-I" "" PYBIND11_INCLUDES_CLEAN "${PYBIND11_INCLUDES}")
separate_arguments(PYBIND11_INCLUDE_DIRS UNIX_COMMAND "${PYBIND11_INCLUDES_CLEAN}")

# Define the library
add_library(vision_cpp SHARED 
    bindings.cpp 
    color_tracker.cpp 
    physics_engine.cpp
)

# Set prefix to empty so it compiles as vision_cpp.so instead of libvision_cpp.so
set_target_properties(vision_cpp PROPERTIES PREFIX "")

# Include directories
target_include_directories(vision_cpp PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${PYBIND11_INCLUDE_DIRS}
    .
)

# Link libraries
target_link_libraries(vision_cpp PRIVATE 
    ${OpenCV_LIBS}
)

# Extension standard (optional but recommended)
set_target_properties(vision_cpp PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
