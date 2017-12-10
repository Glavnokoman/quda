find_package(CUDA QUIET REQUIRED)

FILE(GLOB_RECURSE cudaHeaderFiles "*.hpp" "*.h" "*.cuh")
add_custom_target(cudaHeaders SOURCES ${cudaHeaderFiles})

file(GLOB cuda_sources "kernel_n.cu")
#file(GLOB cuda_sources "kernerl_[1-2].cu")
#file(GLOB cuda_sources  RELATIVE ${PROJECT_SOURCE_DIR} "kernel_1.cu") # debug

if(${PROJECT_NAME}_stub_cuda) # simulate cuda on cpu
	foreach(cusrc in ${cuda_sources})
		set_source_files_properties(${cusrc} PROPERTIES LANGUAGE CXX)
	endforeach()
	add_library(cuda_kernels SHARED ${cuda_sources})
		target_compile_definitions(cuda_kernels PUBLIC CUDA_STABBED)
		target_include_directories(cuda_kernels PUBLIC ${CUDA_INCLUDE_DIRS})
		set_target_properties(cuda_kernels PROPERTIES LINKER_LANGUAGE CXX)
		target_compile_options(cuda_kernels PRIVATE -x c++)
else() # build real cuda stuff
	list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo
											-use_fast_math -Xptxas -dlcm=cg)
#	list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")
	set(RETARDED_COMPILER_FLAGS -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES) # required with some compiler configurations
	list(APPEND CUDA_NVCC_FLAGS ${RETARDED_COMPILER_FLAGS})

	cuda_add_library(cuda_kernels SHARED ${cuda_sources})
	target_include_directories(cuda_kernels PUBLIC ${CUDA_INCLUDE_DIRS}) # why, why is this needed after cuda_add_library!?!?
endif()
