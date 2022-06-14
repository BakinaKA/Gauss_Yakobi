cmake_minimum_required(VERSION 3.12)

set(ProjectId Gauss_Yakobi)
project(${ProjectId})

include("${CMAKE_SOURCE_DIR}/cmake/Configure.cmake")

enable_language(CXX C )

############################### GTest ###############################
SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
include_directories("${CMAKE_SOURCE_DIR}/3rdparty/gtest/include")
include_directories("${CMAKE_SOURCE_DIR}/3rdparty/gtest/googletest/include")
add_subdirectory("${CMAKE_SOURCE_DIR}/3rdparty/gtest")

############################### OpenMP ##############################
find_package( OpenMP )
if( OpenMP_FOUND )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif( OpenMP_FOUND )

############################ std::thread ############################
find_package( Threads )

################################ TBB ################################
if( WIN32 )
    include( cmake/TBBGet.cmake )
    tbb_get( TBB_ROOT tbb_root RELEASE_TAG "v2020.3" CONFIG_DIR TBB_DIR )

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(SYSTEM_BIT "intel64")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(SYSTEM_BIT "ia32")
    endif()

    file( GLOB TBB_PRE_LIB  "${tbb_root}/bin/${SYSTEM_BIT}/vc14/*.dll" )
    file( COPY ${TBB_PRE_LIB} DESTINATION "${CMAKE_BINARY_DIR}/bin")

    find_package( TBB )
else( WIN32 )
    include( cmake/FindTBB.cmake )
endif( WIN32 )

if( TBB_FOUND )
    message(STATUS "Found TBB: TRUE (found version \"${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR}\") ")
    include_directories("${tbb_root}/include")
endif( TBB_FOUND )

message(STATUS "TBB_IMPORTED_TARGETS: ${TBB_IMPORTED_TARGETS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(LIBRARY     "lib_${ProjectId}")
set(TESTS       "test_${ProjectId}")

add_subdirectory(src)
add_subdirectory(test)