include(LibFindMacros)

find_path(Lest_INCLUDE_DIR NAMES lest.hpp PATH_SUFFIXES "lest")
set(Lest_PROCESS_INCLUDES Lest_INCLUDE_DIR)
libfind_process(Lest)
