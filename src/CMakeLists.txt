set(target ${LIBRARY})

file(GLOB srcs "*.cpp")
file(GLOB hdrs "*.h")
set_source_files_properties(${srcs} ${hdrs} PROPERTIES
    LABELS "${ProjectId};Library")

add_library(${target} STATIC ${srcs} ${hdrs})
set_target_properties(${target} PROPERTIES
    OUTPUT_NAME ${ProjectId}
    LABELS "${ProjectId};Library")

if (UNIX)
  target_link_libraries(${target} ${CMAKE_THREAD_LIBS_INIT})
endif (UNIX)

target_link_libraries(${target} gtest)