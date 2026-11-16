add_library(safelog_storage
  in_memory_repository.cpp
  sqlite_codec.cpp
  sqlite_repository.cpp
  system_services.cpp
  types.cpp
  uuid_id_generator.cpp
)
target_include_directories(safelog_storage PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Embed database/schema.sql as a C++ string; re-configures when the file changes.
set(SAFELOG_SCHEMA_FILE ${PROJECT_SOURCE_DIR}/database/schema.sql)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${SAFELOG_SCHEMA_FILE})
file(READ ${SAFELOG_SCHEMA_FILE} SAFELOG_SCHEMA_SQL)
configure_file(schema_sql.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/safelog/storage/schema_sql.hpp @ONLY)
target_include_directories(safelog_storage PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

target_link_libraries(safelog_storage PRIVATE safelog_sqlite3)
