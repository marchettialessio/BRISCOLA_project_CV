cmake_minimum_required(VERSION 3.10)

project(Briscola)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp)

target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(main PRIVATE ${OpenCV_LIBS})

target_compile_definitions(main PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")