cmake_minimum_required(VERSION 3.20)

project(amd_simulator NONE)

if(NOT DEFINED triton_BINARY_DIR)
  set(triton_BINARY_DIR "${CMAKE_BINARY_DIR}")
endif()

set(SIMULATOR_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/simulator.py")
set(SIMULATOR_OUTPUT_DIR "${triton_BINARY_DIR}/bin")

file(MAKE_DIRECTORY "${SIMULATOR_OUTPUT_DIR}")

add_custom_target(amd-simulator ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SIMULATOR_SCRIPT}"
          "${SIMULATOR_OUTPUT_DIR}/simulator.py"
  DEPENDS "${SIMULATOR_SCRIPT}"
  COMMENT "Copying simulator.py to bin/"
)

if(UNIX)
  add_custom_command(
    TARGET amd-simulator POST_BUILD
    COMMAND chmod +x "${SIMULATOR_OUTPUT_DIR}/simulator.py"
  )
endif()

add_subdirectory(test)
