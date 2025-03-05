cmake_minimum_required(VERSION 3.9)
project(lab2)

find_package(OpenMP)

add_executable(lab2 main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab2 PUBLIC OpenMP::OpenMP_CXX)
endif()