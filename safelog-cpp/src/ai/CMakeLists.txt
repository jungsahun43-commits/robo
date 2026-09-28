add_library(safelog_ai ai_safety_service.cpp mock_ai_analyzer.cpp)
target_include_directories(safelog_ai PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(safelog_ai PUBLIC safelog_storage)
