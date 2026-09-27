add_library(safelog_workflow workflow_service.cpp)
target_include_directories(safelog_workflow PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(safelog_workflow PUBLIC safelog_storage)
