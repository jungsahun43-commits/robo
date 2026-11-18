# Header-only port (ICredentialStore, CredentialRecord, errors) shared with storage,
# so storage can implement it without depending on the auth implementation.
add_library(safelog_auth_api INTERFACE)
target_include_directories(safelog_auth_api INTERFACE
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(safelog_auth
  auth_service.cpp
  password_hasher.cpp
  secure_random.cpp
)
target_link_libraries(safelog_auth
  PUBLIC safelog_auth_api
  PRIVATE safelog_monocypher
)
if(WIN32)
  target_link_libraries(safelog_auth PRIVATE bcrypt)
endif()
