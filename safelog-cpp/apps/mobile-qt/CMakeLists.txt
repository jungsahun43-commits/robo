find_package(Qt6 6.5 REQUIRED COMPONENTS Quick QuickControls2 Concurrent Network)
qt_standard_project_setup(REQUIRES 6.5)
set(SAFELOG_CONTROLLER_SOURCES
  adapters/application_support.hpp adapters/workflow_support.hpp adapters/sharing_adapter.hpp
  adapters/ai/http_ai_safety_analyzer.hpp adapters/ai/http_ai_safety_analyzer.cpp
  controllers/workflow/workflow_controller.hpp controllers/workflow/workflow_controller.cpp
  controllers/reporting/reporting_controller.hpp controllers/reporting/reporting_controller.cpp
  app_controller.hpp app_controller.cpp
  controllers/auth/auth_controller.hpp controllers/auth/auth_controller.cpp
  controllers/ai/ai_controller.hpp controllers/ai/ai_controller.cpp
  controllers/capture/capture_controller.hpp controllers/capture/capture_controller.cpp
)
qt_add_executable(safelog_mobile main.cpp ${SAFELOG_CONTROLLER_SOURCES})
qt_add_qml_module(safelog_mobile
  URI SafeLog
  VERSION 1.0
  QML_FILES
    qml/workflow/AssignedTasksPage.qml
    qml/workflow/ActionSubmitPage.qml
    qml/workflow/AiComparisonReviewPage.qml
    qml/reporting/ReportPreviewPage.qml
    qml/Main.qml
    qml/auth/LoginPage.qml
    qml/home/HomePage.qml
    qml/capture/FindingCapturePage.qml
    qml/capture/AiHazardReviewPage.qml
    qml/capture/FindingSavedPage.qml
    qml/common/ErrorPanel.qml
    qml/common/LoadingOverlay.qml
    qml/common/PhotoPreview.qml
    qml/common/PlaceholderPage.qml
)
target_include_directories(safelog_mobile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(safelog_mobile PRIVATE
  Qt6::Quick Qt6::QuickControls2 Qt6::Concurrent Qt6::Network
  safelog_storage safelog_capture safelog_workflow safelog_ai safelog_reporting
)
set_target_properties(safelog_mobile PROPERTIES MACOSX_BUNDLE TRUE WIN32_EXECUTABLE TRUE)

qt_add_executable(safelog_ai_eval
  tools/ai_evaluator.cpp
  adapters/ai/http_ai_safety_analyzer.hpp
  adapters/ai/http_ai_safety_analyzer.cpp
)
target_include_directories(safelog_ai_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(safelog_ai_eval PRIVATE Qt6::Core Qt6::Network safelog_ai)

if(SAFELOG_BUILD_TESTS)
  find_package(Qt6 6.5 REQUIRED COMPONENTS Test)
  qt_add_executable(safelog_mobile_tests tests/controller_tests.cpp ${SAFELOG_CONTROLLER_SOURCES})
  target_include_directories(safelog_mobile_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(safelog_mobile_tests PRIVATE Qt6::Quick Qt6::Concurrent Qt6::Network Qt6::Test
    safelog_storage safelog_capture safelog_workflow safelog_ai safelog_reporting)
  add_test(NAME safelog_mobile_tests COMMAND safelog_mobile_tests)
  set_tests_properties(safelog_mobile_tests PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  qt_add_executable(safelog_ai_http_tests
    tests/http_ai_safety_analyzer_tests.cpp
    adapters/ai/http_ai_safety_analyzer.hpp
    adapters/ai/http_ai_safety_analyzer.cpp
  )
  target_include_directories(safelog_ai_http_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(safelog_ai_http_tests PRIVATE
    Qt6::Quick Qt6::Network Qt6::Test safelog_ai)
  add_test(NAME safelog_ai_http_tests COMMAND safelog_ai_http_tests)
  set_tests_properties(safelog_ai_http_tests PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
endif()
