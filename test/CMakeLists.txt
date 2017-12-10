find_package(Lest REQUIRED)
include_directories(SYSTEM "${Lest_INCLUDE_DIR}")

include(enableSanitizer)
include_directories(".")

add_library(randoms STATIC random_objects.cpp)
target_include_directories(randoms PUBLIC ${PROJECT_SOURCE_DIR})

add_subdirectory(cuda)
