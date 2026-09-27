add_library(safelog_storage
  in_memory_repository.cpp
  system_services.cpp
  types.cpp
)
target_include_directories(safelog_storage PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
