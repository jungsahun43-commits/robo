cmake_minimum_required(VERSION 3.24)
project(SafeLog VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SAFELOG_BUILD_TESTS "Build lightweight module tests" ON)
option(SAFELOG_BUILD_QT "Build the Qt 6 mobile shell" OFF)

add_subdirectory(src/storage)
add_subdirectory(src/capture)
add_subdirectory(src/workflow)
add_subdirectory(src/reporting)
add_subdirectory(apps/demo)

if(SAFELOG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SAFELOG_BUILD_QT)
  add_subdirectory(apps/mobile-qt)
endif()
