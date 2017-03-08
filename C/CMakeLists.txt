project(Integral)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# find_package(OpenCV 3.1 REQUIRED)

find_package(OpenMP)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

if (CMAKE_VERSION VERSION_LESS 3.1)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
endif()

add_library(integral-c SHARED "integral-c.cpp")
# target_link_libraries(integral-c ${OpenCV_LIBS})