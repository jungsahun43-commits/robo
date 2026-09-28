add_executable(safelog_demo main.cpp)
target_link_libraries(safelog_demo PRIVATE
  safelog_storage safelog_capture safelog_workflow safelog_ai safelog_reporting
)
