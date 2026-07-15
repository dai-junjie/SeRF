set(CMAKE_CXX_STANDARD 17)

add_executable(generate_groundtruth generate_groundtruth.cc)
target_link_libraries(generate_groundtruth UTIL)
target_compile_options(generate_groundtruth PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(generate_groundtruth ${OpenMP_CXX_FLAGS})
target_link_libraries(generate_groundtruth OpenMP::OpenMP_CXX)


add_executable(serf_halfbound serf_halfbound.cc)
target_link_libraries(serf_halfbound UTIL)
target_compile_options(serf_halfbound PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(serf_halfbound ${OpenMP_CXX_FLAGS})
target_link_libraries(serf_halfbound OpenMP::OpenMP_CXX)

add_executable(serf_arbitrary serf_arbitrary.cc)
target_link_libraries(serf_arbitrary UTIL)
target_compile_options(serf_arbitrary PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(serf_arbitrary ${OpenMP_CXX_FLAGS})
target_link_libraries(serf_arbitrary OpenMP::OpenMP_CXX)

# Save/load and multi-threaded benchmarks used by the reproducibility suite.
foreach(target serf_save_load hnsw_save_load serf_multithread hnsw_multithread)
  add_executable(${target} ${target}.cc)
  target_link_libraries(${target} UTIL OpenMP::OpenMP_CXX)
  target_compile_options(${target} PRIVATE -Wall ${OpenMP_CXX_FLAGS})
  target_link_libraries(${target} ${OpenMP_CXX_FLAGS})
endforeach()
