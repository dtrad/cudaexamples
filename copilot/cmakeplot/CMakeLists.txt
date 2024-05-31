cmake_minimum_required(VERSION 3.10)
project(cuda_matplotlib_example)
# tell cmake to use nvc++ as the compiler
set(CMAKE_C_COMPILER=nvc++)
set(CMAKE_CXX_COMPILER=nvc++)
# specify the language is CUDA
enable_language(CUDA)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})
add_executable(cuda_matplotlib_example main.cu)
target_link_libraries(cuda_matplotlib_example ${Python3_LIBRARIES} ${pybind11_LIBRARIES} cuda)
