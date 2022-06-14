set(target ${TESTS})

file(GLOB srcs "*.cpp")
set_source_files_properties(${srcs} PROPERTIES
    LABELS "${ProjectId};Test")

add_executable(${target} ${srcs} ${hdrs})
set_target_properties(${target} PROPERTIES
    LABELS "${ProjectId};Test")

if (UNIX)
  target_link_libraries(${target} gtest ${CMAKE_THREAD_LIBS_INIT} pthread)
endif (UNIX)
target_link_libraries(${target} gtest ${LIBRARY})

# VS2012 doesn't support correctly the tuples yet,
# see http://code.google.com/p/googletest/issues/detail?id=412
if(MSVC)
    target_compile_definitions(${target} PUBLIC _VARIADIC_MAX=10)
endif()

add_test(
    NAME ${ProjectId}_gtest
    COMMAND ${target}
)
set_tests_properties (${ProjectId}_gtest PROPERTIES
    LABELS "${ProjectId}")


if( WIN32 )
    target_link_libraries(${target} ${TBB_IMPORTED_TARGETS})
else( WIN32 )
    target_link_libraries(${target} ${TBB_LIBRARIES})
endif( WIN32 )
