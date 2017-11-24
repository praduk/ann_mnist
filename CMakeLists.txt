#
# Application Build File
#

cmake_minimum_required( VERSION 3.2 )

# Set Project Name
set( PNAME ann_mnist )

project( ${PNAME}_project CXX )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release"
        CACHE STRING "Choose the type of build, options are: Debug Release
        RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Find all C++ source files
file(GLOB_RECURSE HDRS
    ${PROJECT_SOURCE_DIR}/src/*.h
    ${PROJECT_SOURCE_DIR}/src/*.hh
    ${PROJECT_SOURCE_DIR}/src/*.hpp )
file(GLOB_RECURSE SRCS
    ${PROJECT_SOURCE_DIR}/src/*.c
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Add Include Directories
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_BINARY_DIR}/include")
add_executable( ${PNAME} ${SRCS} )

# Add Libraries
#find_package(qsim REQUIRED)
#include_directories(${qsim_INCLUDE_DIRS})
#target_link_libraries(${PNAME} qsim)

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

install( TARGETS ${PNAME}
    EXPORT ${PNAME} 
    DESTINATION bin )
install(EXPORT ${PNAME} DESTINATION bin)

