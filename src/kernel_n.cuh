#pragma once

#include "cuda_cpu_debug.hpp"
#include "cudaRunChecked.hpp"

#include "types.h"
#include "utils.hpp"
#include "matrix_permutation.hpp"
#include "integer_sequence.hpp"

#include <array>
#include <cassert>
#include <cmath>

using namespace quda;
using quda::dev_complex;

template<class T> using raw_ptr = T*;

using dev_complex_ptr = raw_ptr<dev_complex>;

constexpr auto n_states(unsigned n_qubits)-> unsigned { return 1u << n_qubits; }
constexpr auto gate_matrix_size(unsigned n_qubits)-> unsigned {
	return n_states(n_qubits)*n_states(n_qubits);
}

__constant__ char gate_matrix[sizeof(dev_complex)*gate_matrix_size(6u)]; // max 6-qubit gates

///
template<class... Ds>
__global__ void kernel_core(dev_complex phi[], unsigned phi_size, Ds... ds)
{
	const unsigned i = thread_id_to_offset(threadIdx.x + blockDim.x*blockIdx.x, ds...);

	auto matrix = reinterpret_cast<dev_complex_ptr>(gate_matrix);
	if(i < phi_size){
		apply_gate_qid(matrix, phi + i, ds...);
	}
}

///
template<class... Ds>
__global__ void kernel_ctrl_core(dev_complex phi[], unsigned phi_size
											, unsigned ctrlmask, Ds... ds)
{
	const unsigned i = thread_id_to_offset(threadIdx.x + blockDim.x*blockIdx.x, ds...);

	auto matrix = reinterpret_cast<dev_complex_ptr>(gate_matrix);
	if(i < phi_size && (i&ctrlmask) == ctrlmask){
		apply_gate_qid(matrix, phi + i, ds...);
	}
}

///
template<class... Ds>
struct Kernel_impl{
	template<size_t... Is>
	auto operator()(quda::StateVector& phi, quda::Matrix const& m, size_t ctrlmask
	                , index_sequence<Is...>, Ds... ds
	                )-> void
	{
		constexpr size_t NQUBITS = sizeof...(Ds);
		auto ids = std::array<unsigned, NQUBITS>{ds...};
		std::reverse(ids.begin(), ids.end());
		const auto offs_and_m = sorted_offsets_and_matrix(ids, m);
		RUN_CHECKED(cudaMemcpyToSymbol(gate_matrix, (void*)offs_and_m.matrix.data()
		                              , sizeof(dev_complex)*gate_matrix_size(NQUBITS)));

		auto d_phi = dev_complex_ptr{};
		RUN_CHECKED(cudaMalloc((void**)&d_phi, phi.size()*sizeof(dev_complex)));
		RUN_CHECKED(cudaMemcpy(d_phi, phi.data(), phi.size()*sizeof(dev_complex)
		                       , cudaMemcpyHostToDevice));

		const unsigned n_threads = 256;
		const auto n_blocks = divide_up(unsigned(phi.size()), n_states(NQUBITS)*n_threads);
		const auto& offs = offs_and_m.offsets;
		if(ctrlmask == 0){
			kernel_call(kernel_core<Ds...>, KernelParams1D{n_blocks, n_threads}
			            , d_phi, phi.size(), offs[Is]...);
		} else {
			kernel_call(kernel_ctrl_core<Ds...>, KernelParams1D{n_blocks, n_threads}
			            , d_phi, phi.size(), ctrlmask, offs[Is]...);
		}

		RUN_CHECKED(cudaMemcpy(phi.data(), d_phi, phi.size()*sizeof(dev_complex)
		                       , cudaMemcpyDeviceToHost));
		RUN_CHECKED(cudaFree(d_phi));
	}
}; // struct Kernel_impl


///
template<class... Ds>
void kernel_(quda::StateVector& phi, quda::Matrix const& m, size_t ctrlmask, Ds... ds){
	assert(m.size() == n_states(sizeof...(ds)));
	assert(m.size() <= phi.size());

	Kernel_impl<Ds...>{} (phi, m, ctrlmask, index_sequence_for<Ds...>{}, ds...);
}
