cmake_minimum_required ( VERSION 2.8 )

set( USE_LIBLBFGS ON )
set( FORCE_OPENBLAS OFF )
set( USE_GOOGLE OFF )

include( ExternalData )
project( RAE )

# set path modify for you own configuration
set( LBFGSDIR "${CMAKE_CURRENT_LIST_DIR}/../liblbfgs" )
set( DLIBDIR "${CMAKE_CURRENT_LIST_DIR}/../dlib-18.18" )

set( EIGEN_DIRS  "/usr/include/eigen3" "/opt/local/include/eigen3" )

set( INCEIGEN "NOT-FOUND" )
foreach( EDIR ${EIGEN_DIRS} )
    message( "-- Searching eigen3 at ${EDIR}... ")
    if( EXISTS ${EDIR} )
        set( INCEIGEN ${EDIR} )
        message( "-- Found.")
        break()
    else()
        message( "-- Not found.")
    endif()
endforeach()

if( ${INCEIGEN} MATCHES "NOT-FOUND" )
    message( FATAL_ERROR "eigen3 was not found at any of the tested places, either install it or set the right path in the CMakeLists.txt file ( try: locate eigen3 )" )
endif()

message( "INCEIGEN=${INCEIGEN}" )

set( GOOGDIR  "/usr/include/google" )

# where to fetch libraries if need be
set( LBFGSGIT "https://github.com/chokkan/liblbfgs" )
set( DLIBHTML "http://dlib.net/files/dlib-18.18.tar.bz2" )

set( LIB_PATH   /lib/ 
                /lib64/  
                /usr/lib 
                /usr/lib64 
                /usr/local/lib 
                /usr/local/lib64 
                /opt/OpenBLAS/lib 
                /opt/local/lib 
                /usr/lib/openblas-base 
                $ENV{OPENBLAS_HOME}/lib
                )

if( ${CMAKE_EXTRA_GENERATOR} MATCHES "Eclipse CDT4" )
	add_definitions( -DCMAKE_CXX_COMPILER_ARG1=-std=c++11 )
	add_definitions( -DCMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT=TRUE )
endif( ${CMAKE_EXTRA_GENERATOR} MATCHES "Eclipse CDT4" )

if( FORCE_OPENBLAS )
    find_library(OPENBLAS NAMES openblas PATHS ${LIB_PATH})
    if( OPENBLAS )
        set(OPENBLAS_FOUND ON)
        message(STATUS "Found OpenBLAS")
    endif( OPENBLAS )
endif()

find_program( WGET wget )
find_program( GIT git )
find_program( TAR tar )
find_program( BUNZIP2 bunzip2 )

find_package( Armadillo REQUIRED )
find_package( HDF5 REQUIRED )

include_directories( "${HDF5_INCLUDE_DIR}" )
include_directories( "${ARMADILLO_INCLUDE_DIRS}" )
include_directories( ${INCEIGEN} )
include_directories( "src" )

if( NOT USE_LIBLBFGS )
    message( "info: liblBFGS will not be used. If you want libLBFGS change the value USE_LIBLBFGS to ON in CMakeLists.txt\n" )
endif()

if( USE_LIBLBFGS AND NOT EXISTS ${LBFGSDIR} )
    message( "no liblbfgs at location given in CMakeLists.txt trying to download..." )
    if( NOT GIT-NOTFOUND )
        message( "downloading..." )
        execute_process( COMMAND ${GIT} clone ${LBFGSGIT} ${LBFGSDIR} )
        message( "done." )
    else()
        message( "WARNING: git not found, could not download liblbfgs" )
        message( "if you want libLBFGS opmization please try downloading it manually, and setting LBFGSDIR in CMakeLists.txt...\n" )
    endif()
endif()

if( NOT EXISTS ${DLIBDIR} )
    message( "no dlib at location given in CMakeLists.txt trying to download" )
    if( NOT WGET-NOTFOUND AND NOT TAR-NOTFOUND AND NOT BUNZIP2-NOTFOUND )
        message( "downloading..." )
        message( "${WGET} -O ${DLIBDIR}.tar.bz2 ${DLIBHTML}" )
        execute_process( COMMAND ${WGET} -O ${DLIBDIR}.tar.bz2 ${DLIBHTML} )
        execute_process( COMMAND ${BUNZIP2} ${DLIBDIR}.tar.bz2 )
        execute_process( COMMAND mkdir -p ${DLIBDIR} )
        execute_process( COMMAND ${TAR} -xf ${DLIBDIR}.tar -C .. )
        message( "done." )
    else()
        message(
            "wget or tar or bunzip not found, couldn't download dlib, try to download manually and/or change path"
        )
    endif()
endif()

if( USE_LIBLBFGS )
    add_custom_command( 
        OUTPUT ${LBFGSDIR}/lib/lbfgs.o
        COMMAND
            cd ${LBFGSDIR} &&
            ./autogen.sh > ${CMAKE_CURRENT_LIST_DIR}/liblbfgs.log && 
            autoreconf --force --install &&
            ./configure >> ${CMAKE_CURRENT_LIST_DIR}/liblbfgs.log &&
            make clean >> ${CMAKE_CURRENT_LIST_DIR}/liblbfgs.log &&
            make >> ${CMAKE_CURRENT_LIST_DIR}/liblbfgs.log &&
            cd ${CMAKE_CURRENT_LIST_DIR}
    )
    add_executable( run_rae ${LBFGSDIR}/lib/lbfgs.o src/AE.cpp src/PARSE_TREE.cpp src/PT_GROUP.cpp src/PROGRESS_MGR.cpp src/util.cpp src/main.cpp )
    add_definitions( -DUSE_LIBLBFGS="use liblbfgs" )
    include_directories( "${LBFGSDIR}/include" )
else()
    add_executable( run_rae src/AE.cpp src/PARSE_TREE.cpp src/PT_GROUP.cpp src/PROGRESS_MGR.cpp src/util.cpp src/main.cpp )
endif()

if( EXISTS ${DLIBDIR} )
    add_definitions( -DUSE_DLIB="use lbfgslib" )
    include_directories ("${DLIBDIR}/dlib/matrix" "${DLIBDIR}" )
else()
    message( "\nwarning: ${DLIBDIR} is not present and DLIB won't be used.\n")
endif()

if( USE_GOOGLE AND EXISTS ${GOOGDIR}/profiler.h )
    add_definitions( -DUSE_GOOGLE="use google" )
    include_directories ("${GOOGDIR}")
endif()

file( MAKE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/example/data )
file( MAKE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/example/models )
file( MAKE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/example/preds )
file( MAKE_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/example/representations )

target_link_libraries( run_rae ${HDF5_LIBRARIES} )
target_link_libraries( run_rae ${ARMADILLO_LIBRARIES} )
if( FORCE_OPENBLAS )
    target_link_libraries( run_rae ${OPENBLAS} )
endif()


if( "${CMAKE_VERSION}" VERSION_GREATER 3.0  )
    set( CXX_STANDARD_REQUIRED ON )
    set_property(TARGET run_rae PROPERTY CXX_STANDARD 11)
    set( GCC_COVERAGE_COMPILE_FLAGS "-O2 -DARMA_USE_HDF5" )
else()
    if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" )
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION
        )
        if( NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7) )
            message( FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater." )
        endif ()
        set( GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -O2 -DARMA_USE_HDF5" )
    else()
        message(FATAL_ERROR "Your C++ compiler is not GNU OR  does not support C++11.")
    endif()

endif()

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )


install( TARGETS run_rae DESTINATION . )
