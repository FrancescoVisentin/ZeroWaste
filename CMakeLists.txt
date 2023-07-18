cmake_minimum_required(VERSION 2.8)
project(ZeroWaste)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(main
    src/main.cpp
    src/segmentation.cpp
    src/classifier.cpp
    src/segmentation_utils.cpp
)

target_link_libraries(main ${OpenCV_LIBS})