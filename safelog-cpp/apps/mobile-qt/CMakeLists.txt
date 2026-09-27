find_package(Qt6 6.5 REQUIRED COMPONENTS Quick QuickControls2)

qt_standard_project_setup(REQUIRES 6.5)
qt_add_executable(safelog_mobile main.cpp app_controller.cpp)
qt_add_qml_module(safelog_mobile
  URI SafeLog
  VERSION 1.0
  QML_FILES qml/Main.qml
)
target_include_directories(safelog_mobile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(safelog_mobile PRIVATE
  Qt6::Quick Qt6::QuickControls2
  safelog_storage safelog_capture safelog_workflow safelog_reporting
)

set_target_properties(safelog_mobile PROPERTIES
  MACOSX_BUNDLE TRUE
  WIN32_EXECUTABLE TRUE
)
