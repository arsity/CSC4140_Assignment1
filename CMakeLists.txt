cmake_minimum_required(VERSION 3.24)
project(CSC4140_Proj1)

set(CMAKE_CXX_STANDARD 20)

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CSC4140_Proj1 main.cpp)

target_link_libraries(CSC4140_Proj1 ${OpenCV_LIBS})
