cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

#set(CMAKE_VERBOSE_MAKEFILE ON)

project( ml )
set( htslib_LIB hts )
find_package( Torch REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Boost COMPONENTS program_options iostreams filesystem system date_time REQUIRED )
include_directories( ${CMAKE_SOURCE_DIR}/htslib )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${TORCH_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIR} )
link_directories( ${CMAKE_SOURCE_DIR}/htslib )

add_executable("${CMAKE_PROJECT_NAME}" "${CMAKE_PROJECT_NAME}.cpp")
target_link_libraries("${CMAKE_PROJECT_NAME}" ${htslib_LIB} ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_property(TARGET "${CMAKE_PROJECT_NAME}" PROPERTY CXX_STANDARD 14)
