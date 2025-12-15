add_custom_target(amd-simulator ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/simulator.py
          ${triton_BINARY_DIR}/bin/simulator.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/simulator.py
  COMMENT "Copying simulator.py to bin/"
)

if(UNIX)
  add_custom_command(
    TARGET amd-simulator POST_BUILD
    COMMAND chmod +x ${triton_BINARY_DIR}/bin/simulator.py
  )
endif()
