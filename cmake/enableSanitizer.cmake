include(CheckCXXCompilerFlag)

# gcc/clang stuff
set(FLAG_NO_OMIT_FRAMEPOINTER "-fno-omit-frame-pointer")
check_cxx_compiler_flag(${FLAG_NO_OMIT_FRAMEPOINTER} HAS_NO_OMIT_FRAMEPOINTER)
if(NOT ${HAS_NO_OMIT_FRAMEPOINTER})
	unset(FLAG_NO_OMIT_FRAMEPOINTER)
endif()

set(FLAG_SANITIZER "-fsanitize=address")
set(CMAKE_REQUIRED_FLAGS "-Werror ${FLAG_SANITIZER}")
check_cxx_compiler_flag(${FLAG_SANITIZER} HAS_SANITIZE_ADDRESS)
if(NOT HAS_SANITIZE_ADDRESS)
	unset(FLAG_SANITIZER)
endif()
add_library(sanitize_address INTERFACE)
target_compile_options(sanitize_address INTERFACE ${FLAG_SANITIZER} ${FLAG_NO_OMIT_FRAMEPOINTER})
target_link_libraries(sanitize_address INTERFACE ${FLAG_SANITIZER})


set(FLAG_SANITIZER "-fsanitize=thread")
set(CMAKE_REQUIRED_FLAGS "-Werror ${FLAG_SANITIZER}")
check_cxx_compiler_flag(${FLAG_SANITIZER} HAS_SANITIZE_THREAD)
if(NOT HAS_SANITIZE_THREAD)
	unset(FLAG_SANITIZER)
endif()
add_library(sanitize_thread INTERFACE)
target_compile_options(sanitize_thread INTERFACE ${FLAG_SANITIZER} ${FLAG_NO_OMIT_FRAMEPOINTER})
target_link_libraries(sanitize_thread INTERFACE ${FLAG_SANITIZER})


set(FLAG_SANITIZER "-fsanitize=leak")
set(CMAKE_REQUIRED_FLAGS "-Werror ${FLAG_SANITIZER}")
check_cxx_compiler_flag(${FLAG_SANITIZER} HAS_SANITIZE_LEAK)
if(NOT HAS_SANITIZE_LEAK)
	unset(FLAG_SANITIZER)
endif()
add_library(sanitize_leak INTERFACE)
target_compile_options(sanitize_leak INTERFACE ${FLAG_SANITIZER} ${FLAG_NO_OMIT_FRAMEPOINTER})
target_link_libraries(sanitize_leak INTERFACE ${FLAG_SANITIZER})


set(FLAG_SANITIZER "-fsanitize=undefined")
set(CMAKE_REQUIRED_FLAGS "-Werror ${FLAG_SANITIZER}")
check_cxx_compiler_flag(${FLAG_SANITIZER} HAS_SANITIZE_UNDEFINED)
if(NOT HAS_SANITIZE_UNDEFINED)
	unset(FLAG_SANITIZER)
endif()
add_library(sanitize_undefined INTERFACE)
target_compile_options(sanitize_undefined INTERFACE ${FLAG_SANITIZER} ${FLAG_NO_OMIT_FRAMEPOINTER})
target_link_libraries(sanitize_undefined INTERFACE ${FLAG_SANITIZER})

unset(FLAG_SANITIZER)
unset(CMAKE_REQUIRED_FLAGS)
