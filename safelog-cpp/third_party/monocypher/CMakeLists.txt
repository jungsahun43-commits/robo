# Monocypher 4.0.2 (CC0-1.0 or BSD-2-Clause, see LICENCE.md), downloaded from
# https://monocypher.org/download/monocypher-4.0.2.tar.gz
enable_language(C)

add_library(safelog_monocypher STATIC monocypher.c)
target_include_directories(safelog_monocypher SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Third-party code: keep the build log focused on SafeLog warnings.
target_compile_options(safelog_monocypher PRIVATE $<IF:$<C_COMPILER_ID:MSVC>,/w,-w>)
