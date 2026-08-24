add_executable(CMakeProject2 CMakeProject2.cpp)

target_link_libraries(CMakeProject2 PRIVATE llm_core)

target_compile_features(CMakeProject2 PRIVATE cxx_std_20)
