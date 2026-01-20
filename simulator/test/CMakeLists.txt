llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  LLVM_BUILD_SHARED_LIBS
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

set(AMD_SIMULATOR_TEST_DEPENDS
  amd-simulator
)

add_lit_testsuite(check-amd-simulator
  "Running AMD simulator tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${AMD_SIMULATOR_TEST_DEPENDS}
)

set_target_properties(check-amd-simulator PROPERTIES FOLDER "Tests")
