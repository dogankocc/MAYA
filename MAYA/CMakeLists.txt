add_executable(MAYA MAYA.cpp)

target_link_libraries(MAYA PRIVATE llm_core)

target_compile_features(MAYA PRIVATE cxx_std_20)
