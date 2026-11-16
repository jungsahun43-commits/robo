# SQLite 3.50.4 amalgamation (public domain), downloaded from
# https://www.sqlite.org/2025/sqlite-amalgamation-3500400.zip
enable_language(C)
find_package(Threads REQUIRED)

add_library(safelog_sqlite3 STATIC sqlite3.c)
target_include_directories(safelog_sqlite3 SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(safelog_sqlite3 PRIVATE
  SQLITE_THREADSAFE=1
  SQLITE_DEFAULT_FOREIGN_KEYS=1
  SQLITE_DQS=0
  SQLITE_OMIT_LOAD_EXTENSION
)
# Third-party code: keep the build log focused on SafeLog warnings.
target_compile_options(safelog_sqlite3 PRIVATE $<IF:$<C_COMPILER_ID:MSVC>,/w,-w>)
target_link_libraries(safelog_sqlite3 PRIVATE Threads::Threads)
