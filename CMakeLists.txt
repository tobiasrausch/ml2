cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project( example-app )
set(CMAKE_CXX_STANDARD 14)
find_package( Torch REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${TORCH_INCLUDE_DIRS} )

add_executable("${CMAKE_PROJECT_NAME}" example-app.cpp)
target_link_libraries("${CMAKE_PROJECT_NAME}" ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET "${CMAKE_PROJECT_NAME}" PROPERTY CXX_STANDARD 14)

