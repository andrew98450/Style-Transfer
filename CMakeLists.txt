cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(StyleTransfer)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


add_executable(StyleTransfer main.cpp)
target_link_libraries(StyleTransfer ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET StyleTransfer PROPERTY CXX_STANDARD 17)