# test/cuda/ cmake file

function(pq_cuda_test arg_test_name arg_test_src)
	add_executable(${arg_test_name} ${arg_test_src})
	target_link_libraries(${arg_test_name} PUBLIC cuda_kernels randoms)
	if(${PROJECT_NAME}_stub_cuda)
		target_link_libraries(${arg_test_name} PUBLIC sanitize_address sanitize_undefined)
	endif()
	target_include_directories(${arg_test_name} PRIVATE ${PROJECT_SOURCE_DIR})
	add_test(${arg_test_name} ${arg_test_name})
endfunction()

pq_cuda_test(test_cuda_kernel_1  kernel1_t.cpp)
pq_cuda_test(test_cuda_kernel_2  kernel2_t.cpp)
pq_cuda_test(test_cuda_kernel_3  kernel3_t.cpp)
pq_cuda_test(test_cuda_kernel_4  kernel4_t.cpp)
pq_cuda_test(test_cuda_kernel_5  kernel5_t.cpp)
