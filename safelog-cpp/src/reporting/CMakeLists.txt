add_library(safelog_reporting report_service.cpp html_renderer.cpp)
target_include_directories(safelog_reporting PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(safelog_reporting PUBLIC safelog_storage)
