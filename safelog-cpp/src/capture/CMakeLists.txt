add_library(safelog_capture capture_service.cpp)
target_include_directories(safelog_capture PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(safelog_capture PUBLIC safelog_storage)
